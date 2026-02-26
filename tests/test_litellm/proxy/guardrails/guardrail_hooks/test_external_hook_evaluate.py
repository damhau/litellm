"""
Unit tests for External Hook Evaluate Guardrail
"""

import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import HTTPException

sys.path.insert(0, os.path.abspath("../../../../../.."))

from litellm.exceptions import GuardrailRaisedException
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.utils import (
    ChatCompletionMessageToolCall,
    Choices,
    Delta,
    ModelResponse,
    ModelResponseStream,
    StreamingChoices,
)

from external_hook_evaluate import ExternalHookEvaluateGuardrail


def _make_tool_call(name: str, arguments: str, call_id: str = "call_1") -> ChatCompletionMessageToolCall:
    return ChatCompletionMessageToolCall(
        id=call_id,
        function={"name": name, "arguments": arguments},
        type="function",
    )


def _make_response_with_tool_calls(tool_calls):
    return ModelResponse(
        id="resp_1",
        choices=[
            Choices(
                index=0,
                finish_reason="tool_calls",
                message={
                    "role": "assistant",
                    "content": None,
                    "tool_calls": tool_calls,
                },
            )
        ],
        model="claude-sonnet-4-20250514",
    )


def _make_response_no_tool_calls():
    return ModelResponse(
        id="resp_2",
        choices=[
            Choices(
                index=0,
                finish_reason="stop",
                message={
                    "role": "assistant",
                    "content": "Hello, world!",
                },
            )
        ],
        model="claude-sonnet-4-20250514",
    )


def _make_data(model: str = "claude-sonnet-4-20250514", metadata: dict | None = None):
    return {
        "model": model,
        "metadata": metadata or {"user_api_key_user_id": "user-1", "user_api_key_team_id": "team-1"},
    }


def _make_user_api_key_dict():
    return UserAPIKeyAuth(api_key="sk-test")


def _make_guardrail(**overrides):
    defaults = {
        "api_base": "http://eval-service:8080",
        "api_key": "test-key-123",
        "guardrail_name": "external-tool-evaluator",
        "default_on": True,
        "event_hook": "post_call",
    }
    defaults.update(overrides)
    return ExternalHookEvaluateGuardrail(**defaults)


class TestExternalHookEvaluateGuardrail:

    def test_initialization(self):
        guardrail = _make_guardrail()
        assert guardrail.api_base == "http://eval-service:8080"
        assert guardrail.api_key == "test-key-123"
        assert guardrail.guardrail_name == "external-tool-evaluator"
        assert guardrail.timeout == 10.0
        assert guardrail.max_retries == 1
        assert guardrail.fail_open is False
        assert guardrail.evaluate_path == "/api/v1/proxy/evaluate"

    def test_custom_settings(self):
        guardrail = _make_guardrail(
            evaluate_path="/custom/evaluate",
            timeout=5.0,
            max_retries=3,
            fail_open=True,
        )
        assert guardrail.evaluate_path == "/custom/evaluate"
        assert guardrail.timeout == 5.0
        assert guardrail.max_retries == 3
        assert guardrail.fail_open is True

    def test_api_base_required(self):
        with pytest.raises(ValueError, match="requires 'api_base'"):
            _make_guardrail(api_base=None)

    def test_api_base_trailing_slash_stripped(self):
        guardrail = _make_guardrail(api_base="http://eval-service:8080/")
        assert guardrail.api_base == "http://eval-service:8080"

    def test_api_key_from_env(self):
        with patch.dict(os.environ, {"MY_KEY": "secret-from-env"}):
            guardrail = _make_guardrail(api_key="os.environ/MY_KEY")
            assert guardrail.api_key == "secret-from-env"

    def test_api_key_none(self):
        guardrail = _make_guardrail(api_key=None)
        assert guardrail.api_key is None

    @pytest.mark.asyncio
    async def test_tool_call_allowed(self):
        guardrail = _make_guardrail()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"decision": "allow"}
        guardrail.async_handler.post = AsyncMock(return_value=mock_response)

        tool_calls = [_make_tool_call("Bash", json.dumps({"command": "ls"}))]
        response = _make_response_with_tool_calls(tool_calls)
        data = _make_data()

        result = await guardrail.async_post_call_success_hook(
            data=data,
            user_api_key_dict=_make_user_api_key_dict(),
            response=response,
        )

        assert result is not None
        guardrail.async_handler.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_tool_call_denied(self):
        guardrail = _make_guardrail()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "decision": "deny",
            "message": "docker push is not allowed",
        }
        guardrail.async_handler.post = AsyncMock(return_value=mock_response)

        tool_calls = [_make_tool_call("Bash", json.dumps({"command": "docker push myimage"}))]
        response = _make_response_with_tool_calls(tool_calls)
        data = _make_data()

        with pytest.raises(GuardrailRaisedException) as exc_info:
            await guardrail.async_post_call_success_hook(
                data=data,
                user_api_key_dict=_make_user_api_key_dict(),
                response=response,
            )

        assert "docker push is not allowed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_no_tool_calls_passthrough(self):
        guardrail = _make_guardrail()
        guardrail.async_handler.post = AsyncMock()

        response = _make_response_no_tool_calls()
        data = _make_data()

        result = await guardrail.async_post_call_success_hook(
            data=data,
            user_api_key_dict=_make_user_api_key_dict(),
            response=response,
        )

        assert result is not None
        guardrail.async_handler.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_endpoint_unreachable_fail_closed(self):
        guardrail = _make_guardrail(max_retries=0)
        guardrail.async_handler.post = AsyncMock(
            side_effect=httpx.ConnectError("Connection refused")
        )

        tool_calls = [_make_tool_call("Bash", json.dumps({"command": "ls"}))]
        response = _make_response_with_tool_calls(tool_calls)
        data = _make_data()

        with pytest.raises(GuardrailRaisedException) as exc_info:
            await guardrail.async_post_call_success_hook(
                data=data,
                user_api_key_dict=_make_user_api_key_dict(),
                response=response,
            )

        assert "unavailable" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_endpoint_unreachable_fail_open(self):
        guardrail = _make_guardrail(fail_open=True, max_retries=0)
        guardrail.async_handler.post = AsyncMock(
            side_effect=httpx.ConnectError("Connection refused")
        )

        tool_calls = [_make_tool_call("Bash", json.dumps({"command": "ls"}))]
        response = _make_response_with_tool_calls(tool_calls)
        data = _make_data()

        result = await guardrail.async_post_call_success_hook(
            data=data,
            user_api_key_dict=_make_user_api_key_dict(),
            response=response,
        )

        assert result is not None

    @pytest.mark.asyncio
    async def test_retry_on_connection_error(self):
        guardrail = _make_guardrail(max_retries=1)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"decision": "allow"}

        guardrail.async_handler.post = AsyncMock(
            side_effect=[httpx.ConnectError("Connection refused"), mock_response]
        )

        tool_calls = [_make_tool_call("Bash", json.dumps({"command": "ls"}))]
        response = _make_response_with_tool_calls(tool_calls)
        data = _make_data()

        result = await guardrail.async_post_call_success_hook(
            data=data,
            user_api_key_dict=_make_user_api_key_dict(),
            response=response,
        )

        assert result is not None
        assert guardrail.async_handler.post.call_count == 2

    @pytest.mark.asyncio
    async def test_retry_on_5xx(self):
        guardrail = _make_guardrail(max_retries=1)

        error_response = MagicMock()
        error_response.status_code = 503
        error_response.request = MagicMock()
        error_response.text = "Service Unavailable"

        ok_response = MagicMock()
        ok_response.status_code = 200
        ok_response.json.return_value = {"decision": "allow"}

        guardrail.async_handler.post = AsyncMock(
            side_effect=[error_response, ok_response]
        )

        tool_calls = [_make_tool_call("Bash", json.dumps({"command": "ls"}))]
        response = _make_response_with_tool_calls(tool_calls)
        data = _make_data()

        result = await guardrail.async_post_call_success_hook(
            data=data,
            user_api_key_dict=_make_user_api_key_dict(),
            response=response,
        )

        assert result is not None
        assert guardrail.async_handler.post.call_count == 2

    @pytest.mark.asyncio
    async def test_4xx_no_retry(self):
        guardrail = _make_guardrail(max_retries=2)

        error_response = MagicMock()
        error_response.status_code = 422
        error_response.text = "Unprocessable Entity"

        guardrail.async_handler.post = AsyncMock(return_value=error_response)

        tool_calls = [_make_tool_call("Bash", json.dumps({"command": "ls"}))]
        response = _make_response_with_tool_calls(tool_calls)
        data = _make_data()

        with pytest.raises(GuardrailRaisedException) as exc_info:
            await guardrail.async_post_call_success_hook(
                data=data,
                user_api_key_dict=_make_user_api_key_dict(),
                response=response,
            )

        # 4xx should not retry — only 1 call
        assert guardrail.async_handler.post.call_count == 1
        assert "error (422)" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_request_payload_format(self):
        guardrail = _make_guardrail()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"decision": "allow"}
        guardrail.async_handler.post = AsyncMock(return_value=mock_response)

        tool_calls = [_make_tool_call("Bash", json.dumps({"command": "echo hello"}))]
        response = _make_response_with_tool_calls(tool_calls)
        data = _make_data()

        await guardrail.async_post_call_success_hook(
            data=data,
            user_api_key_dict=_make_user_api_key_dict(),
            response=response,
        )

        call_args = guardrail.async_handler.post.call_args
        assert call_args.kwargs["url"] == "http://eval-service:8080/api/v1/proxy/evaluate"

        payload = call_args.kwargs["json"]
        assert payload["tool_name"] == "Bash"
        assert payload["tool_input"] == {"command": "echo hello"}
        assert payload["model"] == "claude-sonnet-4-20250514"
        assert payload["metadata"]["user_id"] == "user-1"
        assert payload["metadata"]["team_id"] == "team-1"
        assert "request_id" in payload["metadata"]

    @pytest.mark.asyncio
    async def test_custom_evaluate_path(self):
        guardrail = _make_guardrail(evaluate_path="/custom/eval")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"decision": "allow"}
        guardrail.async_handler.post = AsyncMock(return_value=mock_response)

        tool_calls = [_make_tool_call("Bash", json.dumps({"command": "ls"}))]
        response = _make_response_with_tool_calls(tool_calls)
        data = _make_data()

        await guardrail.async_post_call_success_hook(
            data=data,
            user_api_key_dict=_make_user_api_key_dict(),
            response=response,
        )

        call_args = guardrail.async_handler.post.call_args
        assert call_args.kwargs["url"] == "http://eval-service:8080/custom/eval"

    @pytest.mark.asyncio
    async def test_api_key_sent_as_bearer_token(self):
        guardrail = _make_guardrail(api_key="my-secret-key")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"decision": "allow"}
        guardrail.async_handler.post = AsyncMock(return_value=mock_response)

        tool_calls = [_make_tool_call("Bash", json.dumps({"command": "ls"}))]
        response = _make_response_with_tool_calls(tool_calls)
        data = _make_data()

        await guardrail.async_post_call_success_hook(
            data=data,
            user_api_key_dict=_make_user_api_key_dict(),
            response=response,
        )

        call_args = guardrail.async_handler.post.call_args
        headers = call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer my-secret-key"

    @pytest.mark.asyncio
    async def test_no_api_key_no_auth_header(self):
        guardrail = _make_guardrail(api_key=None)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"decision": "allow"}
        guardrail.async_handler.post = AsyncMock(return_value=mock_response)

        tool_calls = [_make_tool_call("Bash", json.dumps({"command": "ls"}))]
        response = _make_response_with_tool_calls(tool_calls)
        data = _make_data()

        await guardrail.async_post_call_success_hook(
            data=data,
            user_api_key_dict=_make_user_api_key_dict(),
            response=response,
        )

        call_args = guardrail.async_handler.post.call_args
        headers = call_args.kwargs["headers"]
        assert "Authorization" not in headers

    @pytest.mark.asyncio
    async def test_parallel_evaluation_all_allowed(self):
        guardrail = _make_guardrail()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"decision": "allow"}
        guardrail.async_handler.post = AsyncMock(return_value=mock_response)

        tool_calls = [
            _make_tool_call("Read", json.dumps({"path": "/tmp/a"}), "call_1"),
            _make_tool_call("Bash", json.dumps({"command": "ls"}), "call_2"),
            _make_tool_call("Write", json.dumps({"path": "/tmp/b"}), "call_3"),
        ]
        response = _make_response_with_tool_calls(tool_calls)
        data = _make_data()

        result = await guardrail.async_post_call_success_hook(
            data=data,
            user_api_key_dict=_make_user_api_key_dict(),
            response=response,
        )

        assert result is not None
        assert guardrail.async_handler.post.call_count == 3

    @pytest.mark.asyncio
    async def test_parallel_evaluation_one_denied(self):
        guardrail = _make_guardrail()

        def mock_post(**kwargs):
            resp = MagicMock()
            resp.status_code = 200
            payload = kwargs.get("json", {})
            if payload.get("tool_name") == "Bash":
                resp.json.return_value = {"decision": "deny", "message": "blocked"}
            else:
                resp.json.return_value = {"decision": "allow"}
            return resp

        guardrail.async_handler.post = AsyncMock(side_effect=mock_post)

        tool_calls = [
            _make_tool_call("Read", json.dumps({"path": "/etc/passwd"}), "call_1"),
            _make_tool_call("Bash", json.dumps({"command": "rm -rf /"}), "call_2"),
            _make_tool_call("Write", json.dumps({"path": "/tmp/x"}), "call_3"),
        ]
        response = _make_response_with_tool_calls(tool_calls)
        data = _make_data()

        with pytest.raises(GuardrailRaisedException):
            await guardrail.async_post_call_success_hook(
                data=data,
                user_api_key_dict=_make_user_api_key_dict(),
                response=response,
            )

        # All 3 should be evaluated in parallel
        assert guardrail.async_handler.post.call_count == 3

    @pytest.mark.asyncio
    async def test_deny_default_message(self):
        guardrail = _make_guardrail()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"decision": "deny"}
        guardrail.async_handler.post = AsyncMock(return_value=mock_response)

        tool_calls = [_make_tool_call("Bash", json.dumps({"command": "ls"}))]
        response = _make_response_with_tool_calls(tool_calls)
        data = _make_data()

        with pytest.raises(GuardrailRaisedException) as exc_info:
            await guardrail.async_post_call_success_hook(
                data=data,
                user_api_key_dict=_make_user_api_key_dict(),
                response=response,
            )

        assert "Bash" in str(exc_info.value)
        assert "denied" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_streaming_openai_allowed(self):
        guardrail = _make_guardrail()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"decision": "allow"}
        guardrail.async_handler.post = AsyncMock(return_value=mock_response)

        chunks = [
            ModelResponseStream(
                id="chunk_1",
                choices=[
                    StreamingChoices(
                        index=0,
                        delta=Delta(
                            tool_calls=[
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "function": {"name": "Bash", "arguments": ""},
                                    "type": "function",
                                }
                            ]
                        ),
                    )
                ],
                model="claude-sonnet-4-20250514",
            ),
            ModelResponseStream(
                id="chunk_2",
                choices=[
                    StreamingChoices(
                        index=0,
                        delta=Delta(
                            tool_calls=[
                                {
                                    "index": 0,
                                    "function": {"arguments": '{"command":"ls"}'},
                                }
                            ]
                        ),
                    )
                ],
                model="claude-sonnet-4-20250514",
            ),
        ]

        async def mock_response_iter():
            for c in chunks:
                yield c

        collected = []
        async for chunk in guardrail.async_post_call_streaming_iterator_hook(
            user_api_key_dict=_make_user_api_key_dict(),
            response=mock_response_iter(),
            request_data=_make_data(),
        ):
            collected.append(chunk)

        assert len(collected) == 2
        guardrail.async_handler.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_streaming_openai_denied(self):
        guardrail = _make_guardrail()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"decision": "deny", "message": "not allowed"}
        guardrail.async_handler.post = AsyncMock(return_value=mock_response)

        chunks = [
            ModelResponseStream(
                id="chunk_1",
                choices=[
                    StreamingChoices(
                        index=0,
                        delta=Delta(
                            tool_calls=[
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "function": {"name": "Bash", "arguments": '{"command":"rm -rf /"}'},
                                    "type": "function",
                                }
                            ]
                        ),
                    )
                ],
                model="claude-sonnet-4-20250514",
            ),
        ]

        async def mock_response_iter():
            for c in chunks:
                yield c

        with pytest.raises(HTTPException) as exc_info:
            async for _ in guardrail.async_post_call_streaming_iterator_hook(
                user_api_key_dict=_make_user_api_key_dict(),
                response=mock_response_iter(),
                request_data=_make_data(),
            ):
                pass

        assert exc_info.value.status_code == 400
        assert "not allowed" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_streaming_anthropic_sse_allowed(self):
        guardrail = _make_guardrail()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"decision": "allow"}
        guardrail.async_handler.post = AsyncMock(return_value=mock_response)

        sse_chunks = [
            b'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"Bash"}}\n\n',
            b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\\"command\\":\\"ls\\"}"}}\n\n',
        ]

        async def mock_response_iter():
            for c in sse_chunks:
                yield c

        collected = []
        async for chunk in guardrail.async_post_call_streaming_iterator_hook(
            user_api_key_dict=_make_user_api_key_dict(),
            response=mock_response_iter(),
            request_data=_make_data(),
        ):
            collected.append(chunk)

        assert len(collected) == 2
        guardrail.async_handler.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_streaming_anthropic_sse_denied(self):
        guardrail = _make_guardrail()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"decision": "deny", "message": "blocked by policy"}
        guardrail.async_handler.post = AsyncMock(return_value=mock_response)

        sse_chunks = [
            b'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"Bash"}}\n\n',
            b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\\"command\\":\\"rm -rf /\\"}"}}\n\n',
        ]

        async def mock_response_iter():
            for c in sse_chunks:
                yield c

        with pytest.raises(HTTPException) as exc_info:
            async for _ in guardrail.async_post_call_streaming_iterator_hook(
                user_api_key_dict=_make_user_api_key_dict(),
                response=mock_response_iter(),
                request_data=_make_data(),
            ):
                pass

        assert exc_info.value.status_code == 400
        assert "blocked by policy" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_streaming_no_tool_calls_passthrough(self):
        guardrail = _make_guardrail()
        guardrail.async_handler.post = AsyncMock()

        chunks = [
            ModelResponseStream(
                id="chunk_1",
                choices=[
                    StreamingChoices(
                        index=0,
                        delta=Delta(content="Hello"),
                    )
                ],
                model="claude-sonnet-4-20250514",
            ),
            ModelResponseStream(
                id="chunk_2",
                choices=[
                    StreamingChoices(
                        index=0,
                        delta=Delta(content=" world"),
                    )
                ],
                model="claude-sonnet-4-20250514",
            ),
        ]

        async def mock_response_iter():
            for c in chunks:
                yield c

        collected = []
        async for chunk in guardrail.async_post_call_streaming_iterator_hook(
            user_api_key_dict=_make_user_api_key_dict(),
            response=mock_response_iter(),
            request_data=_make_data(),
        ):
            collected.append(chunk)

        assert len(collected) == 2
        guardrail.async_handler.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_timeout_passed_to_handler(self):
        guardrail = _make_guardrail(timeout=5.0)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"decision": "allow"}
        guardrail.async_handler.post = AsyncMock(return_value=mock_response)

        tool_calls = [_make_tool_call("Bash", json.dumps({"command": "ls"}))]
        response = _make_response_with_tool_calls(tool_calls)
        data = _make_data()

        await guardrail.async_post_call_success_hook(
            data=data,
            user_api_key_dict=_make_user_api_key_dict(),
            response=response,
        )

        call_args = guardrail.async_handler.post.call_args
        assert call_args.kwargs["timeout"] == 5.0
