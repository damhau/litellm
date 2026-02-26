# LiteLLM Changes Required for Custom Guardrails with Claude Code

This document describes the modifications needed in the LiteLLM codebase to make custom `post_call` guardrails work correctly when Claude Code is the client.

## Context

When Claude Code connects to a LiteLLM proxy via `ANTHROPIC_BASE_URL`, requests go through the `/v1/messages` endpoint using Anthropic's native streaming SSE format. Two issues in the LiteLLM codebase prevent custom guardrails from working properly in this setup.

## Change 1: Preserve Anthropic-native tool format through guardrail processing

**File:** `litellm/llms/anthropic/chat/guardrail_translation/handler.py`

**Commit:** `45ccf15` — _fix: preserve Anthropic-native tool format through guardrail processing_

### Problem

LiteLLM's guardrail translation layer converts Anthropic-format tools (e.g. `type: "bash_20250124"`) to OpenAI format (`type: "function"`) for guardrail inspection. After processing, it wrote the OpenAI-format tools back into the request data. This caused Anthropic's API to reject requests from Claude Code with `"type: function"` errors, because Claude Code uses native Anthropic tool types.

### Fix

Added `_reconcile_guardrailed_tools()` method (lines 158-208) that:

1. Preserves the original Anthropic-format tools before guardrail processing
2. After guardrail inspection, detects which tools the guardrail kept/removed
3. Maps modifications back to the original Anthropic-format tools instead of overwriting with OpenAI-format tools

**Key code** (lines 96-100, 134-143):

```python
# Preserve originals before guardrail processing
original_anthropic_tools: Optional[list] = data.get("tools")

# After guardrail runs, reconcile back to Anthropic format
if guardrailed_tools is not None:
    data["tools"] = self._reconcile_guardrailed_tools(
        original_anthropic_tools=original_anthropic_tools,
        openai_tools_before=tools_to_check,
        openai_tools_after=guardrailed_tools,
    )
```

### Tests

`tests/test_litellm/llms/anthropic/chat/guardrail_translation/test_anthropic_guardrail_handler.py` — 9 tests covering tool format preservation and reconciliation.

---

## Change 2: Return guardrail error details instead of generic 500 on streaming denial

**File:** `litellm/proxy/common_request_processing.py`

### Problem

When a `post_call` streaming guardrail denies a tool call, it raises `HTTPException(status_code=400, detail={...})` from inside the async streaming generator. The `create_response()` function (line 149) catches this while consuming the first chunk:

```python
first_chunk_value = await generator.__anext__()  # line 169
```

The guardrail buffers ALL chunks before evaluating, so the `HTTPException` is raised here. The existing code had a generic `except Exception` handler (line 214) that discarded the actual error and returned:

```
500: {"error": {"message": "Error processing stream start", "code": 500}}
```

Claude Code received this generic error and timed out/retried instead of displaying the guardrail denial message.

### Fix

Added an `except HTTPException` handler **before** the generic `except Exception` block (lines 214-224) that preserves the status code and error detail:

```python
except HTTPException as e:
    # Guardrail or known HTTP error while consuming stream.
    verbose_proxy_logger.warning(
        f"HTTPException consuming first chunk from generator: {e.status_code}: {e.detail}"
    )
    error_detail = e.detail if isinstance(e.detail, dict) else {"message": str(e.detail)}
    return JSONResponse(
        status_code=e.status_code,
        content={"error": error_detail},
        headers=headers,
    )
except Exception as e:
    # ... existing generic handler unchanged ...
```

Now Claude Code receives:

```
400: {"error": {"error": "Violated guardrail policy", "guardrail_name": "external-tool-evaluator", "detection_message": "Pushing a Docker image to a remote registry..."}}
```

### Tests

Existing tests in `tests/test_litellm/proxy/test_common_request_processing.py` cover the generic error case. The `HTTPException` path is tested end-to-end via the guardrail tests.

---

## Request Flow (after changes)

```
Claude Code
  → POST /v1/messages (streaming, with tools)
  → LiteLLM proxy authenticates with master_key
  → Translates to Anthropic API call (preserving native tool format — Change 1)
  → Anthropic returns streaming response with tool_use
  → post_call guardrail buffers all chunks
  → Extracts tool calls from SSE stream
  → Sends each tool call to external evaluation backend
  → If backend returns "deny":
      → GuardrailRaisedException → HTTPException(400)
      → create_response() catches HTTPException → JSONResponse(400) — Change 2
      → Claude Code displays the guardrail error message
  → If backend returns "allow":
      → All buffered chunks are yielded to the client
      → Claude Code executes the tool call
```

## Config Example

```yaml
model_list:
  - model_name: anthropic/*
    litellm_params:
      model: anthropic/*
      api_key: os.environ/ANTHROPIC_API_KEY

guardrails:
  - guardrail_name: "external-tool-evaluator"
    litellm_params:
      guardrail: external_hook_evaluate.ExternalHookEvaluateGuardrail
      mode: post_call
      default_on: true
      api_base: "http://localhost:8000"
      api_key: "os.environ/EVAL_SERVICE_API_KEY"
      evaluate_path: "/api/v1/proxy/evaluate"  # default
      timeout: 10                               # seconds, default 10
      max_retries: 1                            # default 1
      fail_open: false                          # default false

general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
```

## Claude Code Configuration

```bash
export ANTHROPIC_BASE_URL="http://127.0.0.1:4111"
export ANTHROPIC_API_KEY="sk-your-master-key"
```

## Note on LiteLLM UI Logs

When a guardrail denies a streaming response, the LiteLLM spend log still shows the request as "success" because the LLM call to Anthropic did succeed — tokens were consumed and cost was incurred. The guardrail denial happens after the LLM response, during streaming iteration. The authoritative record of guardrail decisions is in the external evaluation backend logs and the proxy server logs (search for `External Hook Evaluate Guardrail: tool '...' denied`).
