"""
External Hook Evaluate Guardrail

A custom LiteLLM guardrail that intercepts LLM responses containing tool_use/tool_calls,
sends each to an external evaluation endpoint, and blocks or allows based on the response.

YAML config example:

    guardrails:
      - guardrail_name: "external-tool-evaluator"
        litellm_params:
          guardrail: external_hook_evaluate.ExternalHookEvaluateGuardrail
          mode: "post_call"
          default_on: true
          api_base: "http://my-eval-service:8080"
          api_key: "os.environ/EVAL_SERVICE_API_KEY"
          # Optional settings:
          # evaluate_path: "/api/v1/proxy/evaluate"  # default
          # timeout: 10                               # seconds, default 10
          # max_retries: 1                            # default 1 (0 to disable)
          # fail_open: false                          # default false (deny on error)
"""

import asyncio
import json
import os
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx
from fastapi import HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.exceptions import GuardrailRaisedException
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.callback_utils import (
    add_guardrail_to_applied_guardrails_header,
)
from litellm.proxy.guardrails.guardrail_hooks.tool_permission import (
    ToolPermissionGuardrail,
)
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import (
    ChatCompletionMessageToolCall,
    Choices,
    LLMResponseTypes,
    ModelResponse,
    ModelResponseStream,
)

GUARDRAIL_NAME = "external-tool-evaluator"


class ExternalHookEvaluateGuardrail(CustomGuardrail):
    def __init__(
        self,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        evaluate_path: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        fail_open: Optional[bool] = None,
        **kwargs,
    ):
        if "supported_event_hooks" not in kwargs:
            kwargs["supported_event_hooks"] = [
                GuardrailEventHooks.post_call,
            ]

        super().__init__(**kwargs)

        if not api_base:
            raise ValueError(
                "ExternalHookEvaluateGuardrail requires 'api_base' in litellm_params"
            )

        self.api_base = api_base.rstrip("/")
        self.api_key = self._resolve_api_key(api_key)
        self.evaluate_path = evaluate_path or "/api/v1/proxy/evaluate"
        self.timeout = timeout if timeout is not None else 10.0
        self.max_retries = max_retries if max_retries is not None else 1
        self.fail_open = fail_open if fail_open is not None else False
        self.async_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback
        )

    @staticmethod
    def _resolve_api_key(api_key: Optional[str]) -> Optional[str]:
        if api_key is None:
            return None
        if api_key.startswith("os.environ/"):
            env_var = api_key.replace("os.environ/", "")
            return os.environ.get(env_var)
        return api_key

    async def _evaluate_tool_call(
        self,
        tool_name: str,
        tool_input: Any,
        model: str,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        url = f"{self.api_base}{self.evaluate_path}"

        payload = {
            "tool_name": tool_name,
            "tool_input": tool_input,
            "model": model,
            "metadata": metadata,
        }

        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last_exception: Optional[Exception] = None
        attempts = 1 + self.max_retries

        for attempt in range(attempts):
            try:
                response = await self.async_handler.post(
                    url=url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                )

                if response.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        f"Evaluation service returned {response.status_code}",
                        request=response.request,
                        response=response,
                    )

                if response.status_code >= 400:
                    verbose_proxy_logger.error(
                        "External Hook Evaluate Guardrail: evaluation service returned %d for tool '%s': %s",
                        response.status_code,
                        tool_name,
                        response.text,
                    )
                    if self.fail_open:
                        return {"decision": "allow"}
                    return {
                        "decision": "deny",
                        "message": f"Evaluation service error ({response.status_code}) for tool '{tool_name}'",
                    }

                return response.json()

            except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as e:
                last_exception = e
                if attempt < attempts - 1:
                    wait = 0.5 * (attempt + 1)
                    verbose_proxy_logger.warning(
                        "External Hook Evaluate Guardrail: attempt %d/%d failed for tool '%s': %s. Retrying in %.1fs",
                        attempt + 1,
                        attempts,
                        tool_name,
                        str(e),
                        wait,
                    )
                    await asyncio.sleep(wait)

        verbose_proxy_logger.error(
            "External Hook Evaluate Guardrail: all %d attempts failed for tool '%s': %s",
            attempts,
            tool_name,
            str(last_exception),
        )
        if self.fail_open:
            return {"decision": "allow"}
        return {
            "decision": "deny",
            "message": f"Evaluation service unavailable for tool '{tool_name}'",
        }

    async def _evaluate_all_tool_calls(
        self,
        tool_calls: List[ChatCompletionMessageToolCall],
        data: dict,
    ) -> None:
        model = data.get("model", "unknown")
        metadata = data.get("metadata") or data.get("litellm_metadata") or {}
        request_id = metadata.get("litellm_call_id") or str(uuid.uuid4())
        eval_metadata = {
            "request_id": request_id,
            "user_id": metadata.get("user_api_key_user_id") or metadata.get("user_id", ""),
            "team_id": metadata.get("user_api_key_team_id") or metadata.get("team_id", ""),
        }

        tasks = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name if tool_call.function else "unknown"
            arguments = getattr(tool_call.function, "arguments", None)
            if isinstance(arguments, str):
                try:
                    tool_input = json.loads(arguments)
                except json.JSONDecodeError:
                    tool_input = arguments
            elif isinstance(arguments, dict):
                tool_input = arguments
            else:
                tool_input = {}

            tasks.append((tool_name, tool_input))

        results = await asyncio.gather(
            *(
                self._evaluate_tool_call(
                    tool_name=name,
                    tool_input=inp,
                    model=model,
                    metadata=eval_metadata,
                )
                for name, inp in tasks
            )
        )

        for (tool_name, _), result in zip(tasks, results):
            decision = result.get("decision", "deny")
            if decision == "deny":
                message = result.get("message", f"Tool '{tool_name}' was denied by external evaluator")
                verbose_proxy_logger.warning(
                    "External Hook Evaluate Guardrail: tool '%s' denied: %s",
                    tool_name,
                    message,
                )
                raise GuardrailRaisedException(
                    guardrail_name=self.guardrail_name or GUARDRAIL_NAME,
                    message=message,
                )

    @log_guardrail_information
    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response: LLMResponseTypes,
    ):
        if not isinstance(response, ModelResponse):
            return response

        if not self.should_run_guardrail(
            data=data, event_type=GuardrailEventHooks.post_call
        ):
            return response

        tool_calls: List[ChatCompletionMessageToolCall] = []
        for choice in response.choices:
            if isinstance(choice, Choices):
                for tool in choice.message.tool_calls or []:
                    tool_calls.append(tool)

        if not tool_calls:
            return response

        await self._evaluate_all_tool_calls(tool_calls, data)

        add_guardrail_to_applied_guardrails_header(
            request_data=data, guardrail_name=self.guardrail_name
        )
        return response

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response: Any,
        request_data: dict,
    ) -> AsyncGenerator[ModelResponseStream, None]:
        if not self.should_run_guardrail(
            data=request_data, event_type=GuardrailEventHooks.post_call
        ):
            async for chunk in response:
                yield chunk
            return

        all_chunks: List[Any] = []
        async for chunk in response:
            all_chunks.append(chunk)

        if not all_chunks:
            return

        first = all_chunks[0]
        is_anthropic_sse = isinstance(first, (bytes, str))

        if is_anthropic_sse:
            tool_calls = ToolPermissionGuardrail._extract_tool_calls_from_anthropic_sse_chunks(all_chunks)
        elif isinstance(first, (ModelResponseStream, dict)):
            tool_calls = ToolPermissionGuardrail._extract_tool_calls_from_openai_stream_chunks(all_chunks)
        else:
            for chunk in all_chunks:
                yield chunk
            return

        if not tool_calls:
            for chunk in all_chunks:
                yield chunk
            return

        try:
            await self._evaluate_all_tool_calls(tool_calls, request_data)
        except GuardrailRaisedException as e:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Violated guardrail policy",
                    "guardrail_name": self.guardrail_name or GUARDRAIL_NAME,
                    "detection_message": e.message,
                },
            )

        add_guardrail_to_applied_guardrails_header(
            request_data=request_data, guardrail_name=self.guardrail_name
        )

        for chunk in all_chunks:
            yield chunk
