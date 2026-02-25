import json
import re
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional, Union

from fastapi import HTTPException

from litellm import ChatCompletionToolParam
from litellm._logging import verbose_proxy_logger
from litellm.caching.dual_cache import DualCache
from litellm.exceptions import GuardrailRaisedException
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.callback_utils import (
    add_guardrail_to_applied_guardrails_header,
)
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.proxy.guardrails.guardrail_hooks.tool_permission import (
    PermissionError,
    ToolPermissionRule,
    ToolResult,
)
from litellm.types.utils import (
    CallTypesLiteral,
    ChatCompletionMessageToolCall,
    Choices,
    LLMResponseTypes,
    ModelResponse,
    ModelResponseStream,
)

GUARDRAIL_NAME = "tool_permission"


class ToolPermissionGuardrail(CustomGuardrail):
    def __init__(
        self,
        rules: Optional[List[Dict]] = None,
        default_action: Literal["deny", "allow"] = "deny",
        on_disallowed_action: Literal["block", "rewrite"] = "block",
        **kwargs,
    ):
        """
        Initialize the Tool Permission Guardrail

        Args:
            rules: List of permission rules
            default_action: Default action when no rule matches ("allow" or "deny")
            on_disallowed_action:
            **kwargs: Additional arguments passed to CustomGuardrail
        """
        # Set supported event hooks - this guardrail only works on post_call
        if "supported_event_hooks" not in kwargs:
            kwargs["supported_event_hooks"] = [
                GuardrailEventHooks.pre_call,
                GuardrailEventHooks.post_call,
            ]

        super().__init__(**kwargs)

        self.rules: List[ToolPermissionRule] = []
        self._compiled_rule_patterns: Dict[str, Dict[str, re.Pattern]] = {}
        self._compiled_rule_targets: Dict[str, Dict[str, Optional[re.Pattern]]] = {}
        if rules:
            for rule_item in rules:
                if isinstance(rule_item, ToolPermissionRule):
                    rule = rule_item
                else:
                    rule = ToolPermissionRule(**rule_item)
                self.rules.append(rule)

                compiled_target_patterns: Dict[str, Optional[re.Pattern]] = {
                    "tool_name": None,
                    "tool_type": None,
                }
                if rule.tool_name is not None:
                    try:
                        compiled_target_patterns["tool_name"] = re.compile(
                            rule.tool_name
                        )
                    except re.error as exc:
                        raise ValueError(
                            f"Invalid regex for tool_name in rule '{rule.id}': {exc}"
                        ) from exc
                if rule.tool_type is not None:
                    try:
                        compiled_target_patterns["tool_type"] = re.compile(
                            rule.tool_type
                        )
                    except re.error as exc:
                        raise ValueError(
                            f"Invalid regex for tool_type in rule '{rule.id}': {exc}"
                        ) from exc
                self._compiled_rule_targets[rule.id] = compiled_target_patterns

                if rule.allowed_param_patterns:
                    compiled_patterns: Dict[str, re.Pattern] = {}
                    for path, pattern in rule.allowed_param_patterns.items():
                        try:
                            compiled_patterns[path] = re.compile(pattern)
                        except re.error as exc:
                            raise ValueError(
                                f"Invalid regex in allowed_param_patterns for rule '{rule.id}': {exc}"
                            ) from exc

                    if compiled_patterns:
                        self._compiled_rule_patterns[rule.id] = compiled_patterns

        # Normalize to lowercase for case-insensitive handling
        self.default_action = default_action.lower() if isinstance(default_action, str) else default_action
        self.on_disallowed_action = on_disallowed_action.lower() if isinstance(on_disallowed_action, str) else on_disallowed_action

        verbose_proxy_logger.debug(
            "Tool Permission Guardrail initialized with %d rules, default_action: %s",
            len(self.rules),
            self.default_action,
        )

    @staticmethod
    def get_config_model():
        from litellm.types.proxy.guardrails.guardrail_hooks.tool_permission import (
            ToolPermissionGuardrailConfigModel,
        )

        return ToolPermissionGuardrailConfigModel

    def _matches_regex(
        self, pattern: Optional[re.Pattern], value: Optional[str]
    ) -> bool:
        if pattern is None:
            return True
        if value is None:
            return False
        return bool(pattern.fullmatch(value))

    def _rule_matches_tool(
        self,
        rule: ToolPermissionRule,
        *,
        tool_name: Optional[str],
        tool_type: Optional[str] = None,
    ) -> tuple[bool, bool]:
        target_patterns = self._compiled_rule_targets.get(rule.id, {})
        name_pattern = target_patterns.get("tool_name")
        type_pattern = target_patterns.get("tool_type")

        name_required = rule.tool_name is not None
        type_required = rule.tool_type is not None

        name_matched = (
            self._matches_regex(name_pattern, tool_name) if name_required else True
        )
        type_matched = (
            self._matches_regex(type_pattern, tool_type) if type_required else True
        )

        overall_match = name_matched and type_matched
        should_check_params = name_required and name_matched

        return overall_match, should_check_params

    def _check_tool_permission(
        self,
        tool_name: Optional[str],
        tool_type: Optional[str] = None,
    ) -> tuple[bool, Optional[str], Optional[str]]:
        """
        Check if a tool is allowed based on the configured rules

        Args:
            tool_name: Name of the tool to check
            tool_type: Type of the tool to check

        Returns:
            Tuple of (is_allowed, rule_id, message)
        """
        verbose_proxy_logger.debug(
            f"Checking permission for tool: {tool_name or tool_type}"
        )

        # Check each rule in order
        for rule in self.rules:
            matches, _ = self._rule_matches_tool(
                rule,
                tool_name=tool_name,
                tool_type=tool_type,
            )
            if matches:
                is_allowed = rule.decision == "allow"
                tool_identifier = tool_name or tool_type or "unknown_tool"
                default_message = f"Tool '{tool_identifier}' {'allowed' if is_allowed else 'denied'} by rule '{rule.id}'"
                message = self.render_violation_message(
                    default=default_message,
                    context={
                        "tool_name": tool_name or tool_identifier,
                        "rule_id": rule.id,
                    },
                )
                verbose_proxy_logger.debug(message)
                return is_allowed, rule.id, message

        # No rule matched, use default action
        is_allowed = self.default_action == "allow"
        tool_identifier = tool_name or tool_type or "unknown_tool"
        default_message = f"Tool '{tool_identifier}' {'allowed' if is_allowed else 'denied'} by default action"
        message = self.render_violation_message(
            default=default_message,
            context={
                "tool_name": tool_name or tool_identifier,
                "rule_id": None,
            },
        )
        verbose_proxy_logger.debug(message)
        return is_allowed, None, message

    def _parse_tool_call_arguments(
        self, tool_call: ChatCompletionMessageToolCall
    ) -> Dict[str, Any]:
        arguments = getattr(tool_call.function, "arguments", None)
        if not arguments:
            return {}

        parsed_arguments: Any = {}
        try:
            if isinstance(arguments, str):
                parsed_arguments = json.loads(arguments)
            elif isinstance(arguments, dict):
                parsed_arguments = arguments
        except json.JSONDecodeError as exc:
            verbose_proxy_logger.warning(
                "Tool Permission Guardrail: Failed to decode arguments for tool %s: %s",
                tool_call.function.name,
                exc,
            )
            return {}

        if isinstance(parsed_arguments, dict):
            return parsed_arguments

        verbose_proxy_logger.debug(
            "Tool Permission Guardrail: Ignoring non-dict arguments for tool %s",
            tool_call.function.name,
        )
        return {}

    def _collect_argument_paths(
        self, value: Any, current_path: str, collected: Dict[str, List[Any]], depth: int = 0
    ) -> None:
        from litellm.constants import DEFAULT_MAX_RECURSE_DEPTH

        if depth > DEFAULT_MAX_RECURSE_DEPTH:
            return

        if isinstance(value, dict):
            for key, sub_value in value.items():
                next_path = f"{current_path}.{key}" if current_path else key
                self._collect_argument_paths(sub_value, next_path, collected, depth + 1)
        elif isinstance(value, list):
            list_path = f"{current_path}[]" if current_path else "[]"
            for item in value:
                self._collect_argument_paths(item, list_path, collected, depth + 1)
        else:
            if not current_path:
                return
            collected.setdefault(current_path, []).append(value)

    def _patterns_match_for_rule(
        self,
        *,
        arguments: Dict[str, Any],
        rule: ToolPermissionRule,
        tool_name: Optional[str],
    ) -> tuple[bool, Optional[str]]:
        compiled_patterns = self._compiled_rule_patterns.get(rule.id)
        if not compiled_patterns:
            return True, None

        path_value_map: Dict[str, List[Any]] = {}
        self._collect_argument_paths(arguments, "", path_value_map)

        for path, compiled_pattern in compiled_patterns.items():
            values = path_value_map.get(path)
            if not values:
                return (
                    False,
                    f"Missing value for path '{path}' required by rule '{rule.id}'",
                )
            for raw_value in values:
                if not compiled_pattern.fullmatch(str(raw_value)):
                    return (
                        False,
                        f"Value '{raw_value}' for path '{path}' does not match allowed pattern"
                        f" '{compiled_pattern.pattern}' for tool '{tool_name or 'unknown_tool'}'",
                    )

        return True, None

    def _get_permission_for_tool_call(
        self, tool_call: ChatCompletionMessageToolCall
    ) -> tuple[bool, Optional[str], Optional[str]]:
        tool_name = tool_call.function.name if tool_call.function else None
        tool_type = getattr(tool_call, "type", None)
        if not tool_name and not tool_type:
            return self.default_action == "allow", None, None

        tool_identifier = tool_name or tool_type or "unknown_tool"

        last_pattern_failure_msg: Optional[str] = None

        for rule in self.rules:
            matches, should_check_params = self._rule_matches_tool(
                rule,
                tool_name=tool_name,
                tool_type=tool_type,
            )
            if not matches:
                continue

            if rule.allowed_param_patterns and should_check_params:
                arguments = self._parse_tool_call_arguments(tool_call)
                if not arguments:
                    last_pattern_failure_msg = f"Tool '{tool_identifier}' is missing arguments required by rule '{rule.id}'"
                    continue

                patterns_match, failure_message = self._patterns_match_for_rule(
                    arguments=arguments,
                    rule=rule,
                    tool_name=tool_name,
                )
                if not patterns_match:
                    last_pattern_failure_msg = failure_message
                    continue

            is_allowed = rule.decision == "allow"
            default_message = f"Tool '{tool_identifier}' {'allowed' if is_allowed else 'denied'} by rule '{rule.id}'"
            message = self.render_violation_message(
                default=default_message,
                context={"tool_name": tool_identifier, "rule_id": rule.id},
            )
            return is_allowed, rule.id, message

        is_allowed = self.default_action == "allow"
        default_message = (
            last_pattern_failure_msg
            if (last_pattern_failure_msg and not is_allowed)
            else f"Tool '{tool_identifier}' {'allowed' if is_allowed else 'denied'} by default action"
        )
        message = self.render_violation_message(
            default=default_message,
            context={"tool_name": tool_identifier, "rule_id": None},
        )
        return is_allowed, None, message

    def _extract_tool_calls_from_response(
        self, response: ModelResponse
    ) -> List[ChatCompletionMessageToolCall]:
        """
        Extract tool_calls from all choices in a model response.

        Args:
            response: The model response to analyze

        Returns:
            List of tool_calls blocks found in the response
        """
        tool_calls = []

        for choice in response.choices:
            if isinstance(choice, Choices):
                for tool in choice.message.tool_calls or []:
                    tool_calls.append(tool)

        return tool_calls

    def _modify_request_with_permission_errors(
        self,
        data: dict,
        denied_tool_names: List[str],
    ):
        """
        Modify the request to replace denied tool_calls blocks with error results

        Args:
            data: The model request to modify
            denied_tools: List of (tool_use, error) tuples for denied tools
        """
        if not denied_tool_names:
            return data

        verbose_proxy_logger.info(
            f"Blocking {len(denied_tool_names)} unauthorized tool uses"
        )

        # Create a mapping of tool_use_id to error result
        error_tool_names = set()
        for tool_use in denied_tool_names:
            error_tool_names.add(tool_use)

        # Modify the tools
        tools: Optional[List[ChatCompletionToolParam]] = data.get("tools")
        if tools is None:
            return data

        new_tools = []
        for tool in tools:
            tool_type = tool.get("type")
            if tool_type == "function":
                tool_name = tool["function"]["name"]
            elif "name" in tool:
                # Anthropic-native tool (e.g. bash_20250124, text_editor_20250124)
                tool_name = tool["name"]
            else:
                continue
            if tool_name not in error_tool_names:
                new_tools.append(tool)
        data["tools"] = new_tools
        return data

    def _create_permission_error_result(
        self, tool_call: ChatCompletionMessageToolCall, error: PermissionError
    ) -> ToolResult:
        """
        Create a tool_result block for a permission error

        Args:
            tool_use: The tool use that was denied
            error: The permission error details

        Returns:
            A tool_result block with the error message
        """
        error_message = f"Permission denied: {error.message}"
        if error.rule_id:
            error_message += f" (Rule: {error.rule_id})"

        return ToolResult(
            tool_use_id=tool_call.id, content=error_message, is_error=True
        )

    def _modify_response_with_permission_errors(
        self,
        response: ModelResponse,
        denied_tools: List[tuple[ChatCompletionMessageToolCall, PermissionError]],
    ) -> None:
        """
        Modify the response to replace denied tool_calls blocks with error results

        Args:
            response: The model response to modify
            denied_tools: List of (tool_use, error) tuples for denied tools
        """
        if not denied_tools:
            return

        verbose_proxy_logger.info(
            f"Blocking {len(denied_tools)} unauthorized tool uses"
        )

        # Create a mapping of tool_use_id to error result
        error_results = {}
        for tool_use, error in denied_tools:
            error_result = self._create_permission_error_result(tool_use, error)
            error_results[tool_use.id] = error_result

        # Modify the response content
        for choice in response.choices:
            if isinstance(choice, Choices):
                filtered_tool_calls = []
                error_messages = []

                # Rewrite tool_calls
                for tool_call in choice.message.tool_calls or []:
                    tool_call_id = tool_call.id
                    if tool_call_id in error_results:
                        error_result = error_results[tool_call_id]
                        error_messages.append(error_result.content)
                    else:
                        filtered_tool_calls.append(tool_call)

                choice.message.tool_calls = (
                    filtered_tool_calls if filtered_tool_calls else None
                )

                # Add error messages to content
                if error_messages:
                    existing_content = choice.message.content
                    if existing_content:
                        choice.message.content = (
                            existing_content + "\n\n" + "\n".join(error_messages)
                        )
                    else:
                        choice.message.content = "\n".join(error_messages)

    @log_guardrail_information
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: CallTypesLiteral,
    ) -> Union[Exception, str, dict, None]:
        """ """
        verbose_proxy_logger.debug("Tool Permission Guardrail Pre-Call Hook")

        from litellm.proxy.common_utils.callback_utils import (
            add_guardrail_to_applied_guardrails_header,
        )

        event_type: GuardrailEventHooks = GuardrailEventHooks.pre_call
        if self.should_run_guardrail(data=data, event_type=event_type) is not True:
            return data

        new_tools: Optional[List[ChatCompletionToolParam]] = data.get("tools")
        if new_tools is None:
            verbose_proxy_logger.warning(
                "Tool Permission Guardrail: not running guardrail. No tools in data"
            )
            return data

        # Debug: log tool names and types seen by the guardrail
        for _t in (new_tools or []):
            _ttype = _t.get("type", "?")
            _tname = _t.get("function", {}).get("name") if _ttype == "function" else _t.get("name", "?")
            verbose_proxy_logger.debug(
                "Tool Permission Guardrail: inspecting tool type=%s name=%s", _ttype, _tname
            )

        # Check permissions for each tool
        denied_tool_names = []
        for tool in new_tools:
            tool_type: Optional[str] = tool.get("type")
            # Handle both OpenAI format (type: "function", function.name)
            # and Anthropic-native format (type: "bash_20250124", name: "bash")
            if tool_type == "function":
                tool_name: str = tool["function"]["name"]
            elif "name" in tool:
                # Anthropic-native tool (e.g. bash_20250124, text_editor_20250124)
                tool_name = tool["name"]
            else:
                continue

            is_allowed, _, message = self._check_tool_permission(tool_name, tool_type)

            if not is_allowed and message is not None:
                verbose_proxy_logger.warning(f"Tool Permission Guardrail: {message}")
                if self.on_disallowed_action == "block":
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": "Violated guardrail policy",
                            "detection_message": message,
                        },
                    )
                denied_tool_names.append(tool_name)

        if denied_tool_names:
            data = self._modify_request_with_permission_errors(data, denied_tool_names)

        verbose_proxy_logger.debug(
            "Tool Permission Guardrail Pre-Call Hook: All tools allowed"
        )

        add_guardrail_to_applied_guardrails_header(
            request_data=data, guardrail_name=self.guardrail_name
        )
        return data

    @log_guardrail_information
    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response: LLMResponseTypes,
    ):
        """
        Check tool usage permissions after the LLM call

        Args:
            data: Request data
            user_api_key_dict: User API key information (unused but required by interface)
            response: The model response to check
        """
        if not isinstance(response, ModelResponse):
            return response

        verbose_proxy_logger.debug(
            "Tool Permission Guardrail Post-Call Hook: Checking response"
        )

        if not self.should_run_guardrail(
            data=data, event_type=GuardrailEventHooks.post_call
        ):
            verbose_proxy_logger.debug(
                "Tool Permission Guardrail: Skipping check (not enabled)"
            )
            return response

        # Extract tool_calls from the response
        tool_calls = self._extract_tool_calls_from_response(response)

        if not tool_calls:
            verbose_proxy_logger.debug("Tool Permission Guardrail: No tool uses found")
            return response

        verbose_proxy_logger.debug(
            f"Tool Permission Guardrail: Found {len(tool_calls)} tool calls"
        )

        # Check permissions for each tool use
        denied_tools = []
        for tool_call in tool_calls:
            is_allowed, rule_id, message = self._get_permission_for_tool_call(tool_call)

            if not is_allowed and message is not None:
                verbose_proxy_logger.warning(f"Tool Permission Guardrail: {message}")

                if self.on_disallowed_action == "block":
                    raise GuardrailRaisedException(
                        guardrail_name=self.guardrail_name,
                        message=message,
                    )
                denied_tools.append(
                    (
                        tool_call,
                        PermissionError(
                            tool_name=(
                                tool_call.function.name
                                if tool_call.function and tool_call.function.name
                                else "unknown_tool"
                            ),
                            rule_id=rule_id,
                            message=message,
                        ),
                    )
                )

        if denied_tools:
            self._modify_response_with_permission_errors(response, denied_tools)
        else:
            verbose_proxy_logger.debug(
                "Tool Permission Guardrail Post-Call Hook: All tools allowed"
            )

        add_guardrail_to_applied_guardrails_header(
            request_data=data, guardrail_name=self.guardrail_name
        )
        return response

    @staticmethod
    def _extract_tool_calls_from_anthropic_sse_chunks(
        chunks: List[Any],
    ) -> List[ChatCompletionMessageToolCall]:
        """
        Parse Anthropic SSE byte/str chunks and extract tool_use blocks.

        Looks for ``content_block_start`` events with ``type: "tool_use"``
        (provides tool id + name) and ``content_block_delta`` events with
        ``input_json_delta`` (provides argument fragments).  Accumulates
        fragments per block index and returns assembled
        ``ChatCompletionMessageToolCall`` objects.
        """
        # Per-block accumulators keyed by content_block index
        block_names: Dict[int, str] = {}
        block_ids: Dict[int, str] = {}
        block_args: Dict[int, str] = {}

        for raw_chunk in chunks:
            text = raw_chunk.decode("utf-8", errors="replace") if isinstance(raw_chunk, bytes) else str(raw_chunk)
            for line in text.splitlines():
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                json_str = line[len("data:"):].strip()
                if not json_str or json_str == "[DONE]":
                    continue
                try:
                    event = json.loads(json_str)
                except json.JSONDecodeError:
                    continue

                event_type = event.get("type")
                index = event.get("index", 0)

                if event_type == "content_block_start":
                    cb = event.get("content_block", {})
                    if cb.get("type") == "tool_use":
                        block_names[index] = cb.get("name", "")
                        block_ids[index] = cb.get("id", "")
                        block_args.setdefault(index, "")

                elif event_type == "content_block_delta":
                    delta = event.get("delta", {})
                    if delta.get("type") == "input_json_delta":
                        partial = delta.get("partial_json", "")
                        block_args[index] = block_args.get(index, "") + partial

        tool_calls: List[ChatCompletionMessageToolCall] = []
        for idx in sorted(block_names):
            tool_calls.append(
                ChatCompletionMessageToolCall(
                    id=block_ids.get(idx, f"toolu_{idx}"),
                    function={
                        "name": block_names[idx],
                        "arguments": block_args.get(idx, ""),
                    },
                    type="function",
                )
            )
        return tool_calls

    @staticmethod
    def _extract_tool_calls_from_openai_stream_chunks(
        chunks: List[ModelResponseStream],
    ) -> List[ChatCompletionMessageToolCall]:
        """
        Iterate OpenAI-format ``ModelResponseStream`` chunks and accumulate
        ``delta.tool_calls`` fragments into complete
        ``ChatCompletionMessageToolCall`` objects.
        """
        # Accumulators keyed by tool_call index
        tc_ids: Dict[int, str] = {}
        tc_names: Dict[int, str] = {}
        tc_args: Dict[int, str] = {}

        for chunk in chunks:
            if not hasattr(chunk, "choices") or not chunk.choices:
                continue
            delta = getattr(chunk.choices[0], "delta", None)
            if delta is None:
                continue
            delta_tool_calls = getattr(delta, "tool_calls", None)
            if not delta_tool_calls:
                continue
            for dtc in delta_tool_calls:
                idx = getattr(dtc, "index", 0)
                tc_id = getattr(dtc, "id", None)
                if tc_id:
                    tc_ids[idx] = tc_id
                func = getattr(dtc, "function", None)
                if func:
                    name = getattr(func, "name", None)
                    if name:
                        tc_names[idx] = name
                    args = getattr(func, "arguments", None)
                    if args:
                        tc_args[idx] = tc_args.get(idx, "") + args

        tool_calls: List[ChatCompletionMessageToolCall] = []
        for idx in sorted(tc_names):
            tool_calls.append(
                ChatCompletionMessageToolCall(
                    id=tc_ids.get(idx, f"call_{idx}"),
                    function={
                        "name": tc_names[idx],
                        "arguments": tc_args.get(idx, ""),
                    },
                    type="function",
                )
            )
        return tool_calls

    def _check_and_enforce_tool_permissions(
        self,
        tool_calls: List[ChatCompletionMessageToolCall],
        request_data: Optional[dict] = None,
    ) -> List[tuple]:
        """
        Check permissions for a list of tool_calls and either raise on block
        or return the list of denied (tool_call, PermissionError) tuples for
        rewrite mode.

        When *request_data* is provided and a block-mode exception is raised,
        guardrail information is attached to the request metadata so the
        LiteLLM UI displays the structured guardrail response instead of a
        raw 500 error.
        """
        from datetime import datetime

        start_time = datetime.now()
        denied_tools: List[tuple] = []
        for tool_call in tool_calls:
            is_allowed, rule_id, message = self._get_permission_for_tool_call(
                tool_call
            )
            if not is_allowed and message is not None:
                verbose_proxy_logger.warning(
                    f"Tool Permission Guardrail: {message}"
                )
                if self.on_disallowed_action == "block":
                    exc = GuardrailRaisedException(
                        guardrail_name=self.guardrail_name,
                        message=message,
                    )
                    if request_data is not None:
                        now = datetime.now()
                        self.add_standard_logging_guardrail_information_to_request_data(
                            guardrail_json_response=exc,
                            request_data=request_data,
                            guardrail_status="guardrail_intervened",
                            start_time=start_time.timestamp(),
                            end_time=now.timestamp(),
                            duration=(now - start_time).total_seconds(),
                            event_type=GuardrailEventHooks.post_call,
                        )
                    raise exc
                denied_tools.append(
                    (
                        tool_call,
                        PermissionError(
                            tool_name=(
                                tool_call.function.name
                                if tool_call.function and tool_call.function.name
                                else "unknown_tool"
                            ),
                            rule_id=rule_id,
                            message=message,
                        ),
                    )
                )
        return denied_tools

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response: Any,
        request_data: dict,
    ) -> AsyncGenerator[ModelResponseStream, None]:
        """
        Check tool usage permissions after the LLM stream call.

        Handles both OpenAI-format chunks (``ModelResponseStream`` objects)
        and Anthropic SSE format (raw ``bytes``/``str`` chunks).
        """
        from litellm.llms.base_llm.base_model_iterator import MockResponseIterator

        # Buffer all chunks
        all_chunks: List[Any] = []
        async for chunk in response:
            all_chunks.append(chunk)

        if not all_chunks:
            return

        # Detect format: OpenAI (ModelResponseStream / dict) vs Anthropic SSE (bytes / str)
        first = all_chunks[0]
        is_anthropic_sse = isinstance(first, (bytes, str))

        if is_anthropic_sse:
            tool_calls = self._extract_tool_calls_from_anthropic_sse_chunks(all_chunks)
        elif isinstance(first, (ModelResponseStream, dict)):
            tool_calls = self._extract_tool_calls_from_openai_stream_chunks(all_chunks)
        else:
            # Unknown format — pass through
            for chunk in all_chunks:
                yield chunk
            return

        if not tool_calls:
            verbose_proxy_logger.debug(
                "Tool Permission Guardrail: No tool uses found"
            )
            for chunk in all_chunks:
                yield chunk
            return

        verbose_proxy_logger.debug(
            f"Tool Permission Guardrail: Found {len(tool_calls)} tool calls"
        )

        denied_tools = self._check_and_enforce_tool_permissions(tool_calls, request_data=request_data)

        if denied_tools and not is_anthropic_sse:
            # Rewrite mode for OpenAI format: reassemble via stream_chunk_builder
            from litellm.main import stream_chunk_builder
            from litellm.types.utils import TextCompletionResponse

            assembled: Optional[
                Union[ModelResponse, TextCompletionResponse]
            ] = stream_chunk_builder(chunks=all_chunks)

            if isinstance(assembled, ModelResponse):
                self._modify_response_with_permission_errors(assembled, denied_tools)
                mock_response = MockResponseIterator(model_response=assembled)
                async for chunk in mock_response:
                    yield chunk
                return

        if not denied_tools:
            verbose_proxy_logger.debug(
                "Tool Permission Guardrail Post-Call Hook: All tools allowed"
            )

        # Yield original buffered chunks (allowed, or anthropic rewrite passthrough)
        for chunk in all_chunks:
            yield chunk
