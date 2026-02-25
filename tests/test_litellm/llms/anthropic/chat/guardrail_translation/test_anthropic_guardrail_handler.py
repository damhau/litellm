"""
Unit tests for Anthropic Messages Guardrail Translation Handler

Tests the handler's ability to process streaming output for Anthropic Messages API
with guardrail transformations, specifically testing edge cases with empty choices.
"""

import os
import sys
from typing import Any, List, Literal, Optional
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

sys.path.insert(
    0, os.path.abspath("../../../../../../..")
)  # Adds the parent directory to the system path

from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.llms.anthropic.chat.guardrail_translation.handler import (
    AnthropicMessagesHandler,
)
from litellm.types.utils import GenericGuardrailAPIInputs


class MockPassThroughGuardrail(CustomGuardrail):
    """Mock guardrail that passes through without blocking - for testing streaming fallback behavior"""

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional[Any] = None,
    ) -> GenericGuardrailAPIInputs:
        """Simply return inputs unchanged"""
        return inputs


class MockToolRemovingGuardrail(CustomGuardrail):
    """Mock guardrail that removes tools by name - for testing tool reconciliation."""

    def __init__(self, guardrail_name: str, tools_to_remove: List[str]):
        super().__init__(guardrail_name=guardrail_name)
        self.tools_to_remove = tools_to_remove

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional[Any] = None,
    ) -> GenericGuardrailAPIInputs:
        """Remove specified tools from the input."""
        tools = inputs.get("tools")
        if tools is not None:
            inputs["tools"] = [
                t
                for t in tools
                if isinstance(t, dict)
                and t.get("function", {}).get("name") not in self.tools_to_remove
            ]
        return inputs


class MockBlockingGuardrail(CustomGuardrail):
    """Mock guardrail that blocks requests containing specific keywords."""

    def __init__(self, guardrail_name: str, blocked_keywords: List[str]):
        super().__init__(guardrail_name=guardrail_name)
        self.blocked_keywords = blocked_keywords

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional[Any] = None,
    ) -> GenericGuardrailAPIInputs:
        """Block request if any text contains a blocked keyword."""
        texts = inputs.get("texts", [])
        for text in texts:
            for keyword in self.blocked_keywords:
                if keyword.lower() in text.lower():
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": f"Content blocked by guardrail: contains '{keyword}'",
                            "type": "content_policy_violation",
                        },
                    )
        return inputs


class MockTextModifyingGuardrail(CustomGuardrail):
    """Mock guardrail that redacts PII-like patterns from text."""

    def __init__(self, guardrail_name: str, redact_map: dict):
        super().__init__(guardrail_name=guardrail_name)
        self.redact_map = redact_map

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional[Any] = None,
    ) -> GenericGuardrailAPIInputs:
        """Replace sensitive patterns in texts."""
        texts = inputs.get("texts", [])
        modified_texts = []
        for text in texts:
            for pattern, replacement in self.redact_map.items():
                text = text.replace(pattern, replacement)
            modified_texts.append(text)
        inputs["texts"] = modified_texts
        return inputs


class MockRecordingGuardrail(CustomGuardrail):
    """Mock guardrail that records what it received for inspection."""

    def __init__(self, guardrail_name: str):
        super().__init__(guardrail_name=guardrail_name)
        self.received_inputs: Optional[GenericGuardrailAPIInputs] = None
        self.received_request_data: Optional[dict] = None
        self.call_count: int = 0

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional[Any] = None,
    ) -> GenericGuardrailAPIInputs:
        """Record inputs for later assertion."""
        self.received_inputs = inputs
        self.received_request_data = request_data
        self.call_count += 1
        return inputs


class MockDynamicGuardrail(CustomGuardrail):
    """Mock guardrail that records dynamic params from request metadata."""

    def __init__(self, guardrail_name: str):
        super().__init__(guardrail_name=guardrail_name)
        self.dynamic_params: Optional[dict] = None

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional[Any] = None,
    ) -> GenericGuardrailAPIInputs:
        self.dynamic_params = self.get_guardrail_dynamic_request_body_params(
            request_data
        )
        return inputs


class TestAnthropicMessagesHandlerStreamingOutputProcessing:
    """Test streaming output processing functionality"""

    @pytest.mark.asyncio
    async def test_process_output_streaming_response_empty_model_response(self):
        """Test that streaming response with None model_response doesn't raise error

        This test verifies the fix for the bug where accessing model_response.choices[0]
        would raise an error when _build_complete_streaming_response returns None.
        """
        handler = AnthropicMessagesHandler()
        guardrail = MockPassThroughGuardrail(guardrail_name="test")

        # Mock _check_streaming_has_ended to return True (stream ended)
        # and _build_complete_streaming_response to return None
        with patch.object(
            handler, "_check_streaming_has_ended", return_value=True
        ), patch(
            "litellm.llms.anthropic.chat.guardrail_translation.handler.AnthropicPassthroughLoggingHandler._build_complete_streaming_response",
            return_value=None,
        ):
            responses_so_far = [b"data: some chunk"]

            # This should not raise an error
            result = await handler.process_output_streaming_response(
                responses_so_far=responses_so_far,
                guardrail_to_apply=guardrail,
                litellm_logging_obj=MagicMock(),
            )

            # Should return the responses unchanged
            assert result == responses_so_far


class TestAnthropicMessagesHandlerInputProcessing:
    """Test input processing preserves litellm_metadata for dynamic guardrails."""

    @pytest.mark.asyncio
    async def test_process_input_messages_preserves_litellm_metadata_guardrails(self):
        handler = AnthropicMessagesHandler()
        guardrail = MockDynamicGuardrail(guardrail_name="cygnal-monitor")

        data = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "hello"}],
            "litellm_metadata": {
                "guardrails": [
                    {
                        "cygnal-monitor": {
                            "extra_body": {"policy_id": "policy-123"}
                        }
                    }
                ]
            },
        }

        with patch("litellm.proxy.proxy_server.premium_user", True):
            await handler.process_input_messages(data=data, guardrail_to_apply=guardrail)

        assert data.get("litellm_metadata", {}).get("guardrails")
        assert guardrail.dynamic_params == {"policy_id": "policy-123"}

    @pytest.mark.asyncio
    async def test_process_output_streaming_response_empty_choices(self):
        """Test that streaming response with empty choices doesn't raise IndexError

        This test verifies the fix for the bug where accessing model_response.choices[0]
        would raise IndexError when the response has an empty choices list.
        """
        from litellm.types.utils import ModelResponse

        handler = AnthropicMessagesHandler()
        guardrail = MockPassThroughGuardrail(guardrail_name="test")

        # Create a mock response with empty choices
        mock_response = ModelResponse(
            id="msg_123",
            created=1234567890,
            model="claude-3",
            object="chat.completion",
            choices=[],  # Empty choices
        )

        # Mock _check_streaming_has_ended to return True (stream ended)
        # and _build_complete_streaming_response to return the mock response
        with patch.object(
            handler, "_check_streaming_has_ended", return_value=True
        ), patch(
            "litellm.llms.anthropic.chat.guardrail_translation.handler.AnthropicPassthroughLoggingHandler._build_complete_streaming_response",
            return_value=mock_response,
        ):
            responses_so_far = [b"data: some chunk"]

            # This should not raise IndexError
            result = await handler.process_output_streaming_response(
                responses_so_far=responses_so_far,
                guardrail_to_apply=guardrail,
                litellm_logging_obj=MagicMock(),
            )

            # Should return the responses unchanged
            assert result == responses_so_far

    @pytest.mark.asyncio
    async def test_process_output_streaming_response_with_valid_choices(self):
        """Test that streaming response with valid choices still works correctly"""
        from litellm.types.utils import Choices, Message, ModelResponse

        handler = AnthropicMessagesHandler()
        guardrail = MockPassThroughGuardrail(guardrail_name="test")

        # Create a mock response with valid choices
        mock_response = ModelResponse(
            id="msg_123",
            created=1234567890,
            model="claude-3",
            object="chat.completion",
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=Message(
                        content="Hello world",
                        role="assistant",
                    ),
                )
            ],
        )

        # Mock _check_streaming_has_ended to return True (stream ended)
        # and _build_complete_streaming_response to return the mock response
        with patch.object(
            handler, "_check_streaming_has_ended", return_value=True
        ), patch(
            "litellm.llms.anthropic.chat.guardrail_translation.handler.AnthropicPassthroughLoggingHandler._build_complete_streaming_response",
            return_value=mock_response,
        ):
            responses_so_far = [b"data: some chunk"]

            # This should process successfully
            result = await handler.process_output_streaming_response(
                responses_so_far=responses_so_far,
                guardrail_to_apply=guardrail,
                litellm_logging_obj=MagicMock(),
            )

            # Should return the responses
            assert result == responses_so_far

    @pytest.mark.asyncio
    async def test_process_output_streaming_response_stream_not_ended(self):
        """Test that streaming response falls back to text processing when stream hasn't ended"""
        handler = AnthropicMessagesHandler()
        guardrail = MockPassThroughGuardrail(guardrail_name="test")

        # Mock _check_streaming_has_ended to return False (stream not ended)
        with patch.object(
            handler, "_check_streaming_has_ended", return_value=False
        ), patch.object(
            handler, "get_streaming_string_so_far", return_value="partial text"
        ):
            responses_so_far = [b"data: some chunk"]

            # This should process successfully using text-based guardrail
            result = await handler.process_output_streaming_response(
                responses_so_far=responses_so_far,
                guardrail_to_apply=guardrail,
                litellm_logging_obj=MagicMock(),
            )

            # Should return the responses
            assert result == responses_so_far


class TestReconcileGuardrailedTools:
    """Test _reconcile_guardrailed_tools preserves Anthropic-native tool format."""

    def test_tools_unchanged_returns_original(self):
        """When guardrail returns tools unchanged, original Anthropic tools are preserved."""
        original_anthropic_tools = [
            {"type": "bash_20250124", "name": "bash"},
            {"type": "text_editor_20250124", "name": "text_editor"},
        ]
        openai_tools_before = [
            {"type": "function", "function": {"name": "bash", "parameters": {}}},
            {"type": "function", "function": {"name": "text_editor", "parameters": {}}},
        ]
        # Guardrail returns tools unchanged
        openai_tools_after = list(openai_tools_before)

        result = AnthropicMessagesHandler._reconcile_guardrailed_tools(
            original_anthropic_tools=original_anthropic_tools,
            openai_tools_before=openai_tools_before,
            openai_tools_after=openai_tools_after,
        )

        assert result == original_anthropic_tools
        # Verify the native types are preserved
        assert result[0]["type"] == "bash_20250124"
        assert result[1]["type"] == "text_editor_20250124"

    def test_tool_removed_by_guardrail(self):
        """When guardrail removes a tool, corresponding Anthropic tool is removed."""
        original_anthropic_tools = [
            {"type": "bash_20250124", "name": "bash"},
            {"type": "text_editor_20250124", "name": "text_editor"},
            {"type": "web_search_20260209", "name": "web_search"},
        ]
        openai_tools_before = [
            {"type": "function", "function": {"name": "bash", "parameters": {}}},
            {"type": "function", "function": {"name": "text_editor", "parameters": {}}},
            {"type": "function", "function": {"name": "web_search", "parameters": {}}},
        ]
        # Guardrail removed "bash" tool
        openai_tools_after = [
            {"type": "function", "function": {"name": "text_editor", "parameters": {}}},
            {"type": "function", "function": {"name": "web_search", "parameters": {}}},
        ]

        result = AnthropicMessagesHandler._reconcile_guardrailed_tools(
            original_anthropic_tools=original_anthropic_tools,
            openai_tools_before=openai_tools_before,
            openai_tools_after=openai_tools_after,
        )

        assert len(result) == 2
        assert result[0]["name"] == "text_editor"
        assert result[0]["type"] == "text_editor_20250124"
        assert result[1]["name"] == "web_search"
        assert result[1]["type"] == "web_search_20260209"

    def test_no_original_tools_returns_guardrail_output(self):
        """When there are no original Anthropic tools, return guardrail output as-is."""
        openai_tools_after = [
            {"type": "function", "function": {"name": "bash", "parameters": {}}},
        ]

        result = AnthropicMessagesHandler._reconcile_guardrailed_tools(
            original_anthropic_tools=None,
            openai_tools_before=[],
            openai_tools_after=openai_tools_after,
        )

        assert result == openai_tools_after

    def test_all_tools_removed_by_guardrail(self):
        """When guardrail removes all tools, return empty list."""
        original_anthropic_tools = [
            {"type": "bash_20250124", "name": "bash"},
        ]
        openai_tools_before = [
            {"type": "function", "function": {"name": "bash", "parameters": {}}},
        ]
        openai_tools_after: list = []

        result = AnthropicMessagesHandler._reconcile_guardrailed_tools(
            original_anthropic_tools=original_anthropic_tools,
            openai_tools_before=openai_tools_before,
            openai_tools_after=openai_tools_after,
        )

        assert result == []

    def test_standard_anthropic_tools_preserved(self):
        """Standard Anthropic tools (without special types) are also preserved."""
        original_anthropic_tools = [
            {
                "name": "get_weather",
                "description": "Get weather information",
                "input_schema": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                },
            },
        ]
        openai_tools_before = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather information",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                    },
                },
            },
        ]
        openai_tools_after = list(openai_tools_before)

        result = AnthropicMessagesHandler._reconcile_guardrailed_tools(
            original_anthropic_tools=original_anthropic_tools,
            openai_tools_before=openai_tools_before,
            openai_tools_after=openai_tools_after,
        )

        assert result == original_anthropic_tools
        # Verify original structure is preserved (input_schema, not parameters)
        assert "input_schema" in result[0]


def _mock_proxy_server():
    """Create a mock for litellm.proxy.proxy_server when proxy deps aren't installed."""
    import types

    mock_module = types.ModuleType("litellm.proxy.proxy_server")
    mock_module.premium_user = True  # type: ignore[attr-defined]
    return mock_module


class TestAnthropicToolFormatPreservation:
    """Integration tests for Anthropic-native tool format preservation through guardrails."""

    @pytest.mark.asyncio
    async def test_native_anthropic_tools_preserved_through_passthrough_guardrail(self):
        """Anthropic-native tools (bash_20250124, etc.) survive guardrail processing unchanged.

        This is the core regression test for the bug where guardrail processing
        converted Anthropic tools to OpenAI format (type: "function"), causing
        the Anthropic API to reject the request.
        """
        handler = AnthropicMessagesHandler()
        guardrail = MockPassThroughGuardrail(guardrail_name="test")

        data = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "List files in current directory"}],
            "tools": [
                {"type": "bash_20250124", "name": "bash"},
                {"type": "text_editor_20250124", "name": "text_editor"},
            ],
        }

        with patch.dict(sys.modules, {"litellm.proxy.proxy_server": _mock_proxy_server()}):
            result = await handler.process_input_messages(
                data=data, guardrail_to_apply=guardrail
            )

        # Tools must remain in Anthropic-native format
        tools = result["tools"]
        assert len(tools) == 2
        assert tools[0]["type"] == "bash_20250124"
        assert tools[0]["name"] == "bash"
        assert tools[1]["type"] == "text_editor_20250124"
        assert tools[1]["name"] == "text_editor"

    @pytest.mark.asyncio
    async def test_native_anthropic_tools_with_removal_guardrail(self):
        """When a guardrail removes a tool, the remaining tools stay in Anthropic format."""
        handler = AnthropicMessagesHandler()
        guardrail = MockToolRemovingGuardrail(
            guardrail_name="test", tools_to_remove=["bash"]
        )

        data = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [
                {"type": "bash_20250124", "name": "bash"},
                {"type": "text_editor_20250124", "name": "text_editor"},
            ],
        }

        with patch.dict(sys.modules, {"litellm.proxy.proxy_server": _mock_proxy_server()}):
            result = await handler.process_input_messages(
                data=data, guardrail_to_apply=guardrail
            )

        # Only text_editor should remain, in Anthropic format
        tools = result["tools"]
        assert len(tools) == 1
        assert tools[0]["type"] == "text_editor_20250124"
        assert tools[0]["name"] == "text_editor"

    @pytest.mark.asyncio
    async def test_mixed_tools_preserved_through_guardrail(self):
        """Mixed Anthropic-native and standard tools are all preserved."""
        handler = AnthropicMessagesHandler()
        guardrail = MockPassThroughGuardrail(guardrail_name="test")

        data = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [
                {"type": "bash_20250124", "name": "bash"},
                {
                    "name": "get_weather",
                    "description": "Get weather",
                    "input_schema": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                    },
                },
            ],
        }

        with patch.dict(sys.modules, {"litellm.proxy.proxy_server": _mock_proxy_server()}):
            result = await handler.process_input_messages(
                data=data, guardrail_to_apply=guardrail
            )

        tools = result["tools"]
        assert len(tools) == 2
        assert tools[0]["type"] == "bash_20250124"
        assert tools[0]["name"] == "bash"
        # Standard tool preserves its original structure
        assert tools[1]["name"] == "get_weather"
        assert "input_schema" in tools[1]

    @pytest.mark.asyncio
    async def test_web_search_tool_preserved_through_guardrail(self):
        """Web search tool type is preserved through guardrail processing."""
        handler = AnthropicMessagesHandler()
        guardrail = MockPassThroughGuardrail(guardrail_name="test")

        data = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "search the web"}],
            "tools": [
                {"type": "web_search_20250305", "name": "web_search"},
            ],
        }

        with patch.dict(sys.modules, {"litellm.proxy.proxy_server": _mock_proxy_server()}):
            result = await handler.process_input_messages(
                data=data, guardrail_to_apply=guardrail
            )

        tools = result["tools"]
        assert len(tools) == 1
        assert tools[0]["type"] == "web_search_20250305"
        assert tools[0]["name"] == "web_search"


class TestGuardrailsAreActuallyApplied:
    """Regression tests: verify guardrails are truly invoked and their effects are visible.

    These tests cover the concern that after the tool-reconciliation fix,
    guardrails might appear to be "not applied anymore."
    """

    @pytest.mark.asyncio
    async def test_blocking_guardrail_raises_exception(self):
        """A guardrail that raises HTTPException must still block the request."""
        handler = AnthropicMessagesHandler()
        guardrail = MockBlockingGuardrail(
            guardrail_name="blocker", blocked_keywords=["secret"]
        )

        data = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Tell me the secret password"}],
            "tools": [
                {"type": "bash_20250124", "name": "bash"},
            ],
        }

        with patch.dict(sys.modules, {"litellm.proxy.proxy_server": _mock_proxy_server()}):
            with pytest.raises(HTTPException) as exc_info:
                await handler.process_input_messages(
                    data=data, guardrail_to_apply=guardrail
                )
            assert exc_info.value.status_code == 400
            assert "secret" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_blocking_guardrail_allows_clean_request(self):
        """A blocking guardrail should let clean requests through."""
        handler = AnthropicMessagesHandler()
        guardrail = MockBlockingGuardrail(
            guardrail_name="blocker", blocked_keywords=["secret"]
        )

        data = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "What is the weather today?"}],
            "tools": [
                {"type": "bash_20250124", "name": "bash"},
            ],
        }

        with patch.dict(sys.modules, {"litellm.proxy.proxy_server": _mock_proxy_server()}):
            result = await handler.process_input_messages(
                data=data, guardrail_to_apply=guardrail
            )

        # Request should pass through
        assert result["messages"][0]["content"] == "What is the weather today?"
        # Tools should remain in Anthropic format
        assert result["tools"][0]["type"] == "bash_20250124"

    @pytest.mark.asyncio
    async def test_text_modifying_guardrail_changes_message_content(self):
        """A guardrail that modifies text must have its changes reflected in messages."""
        handler = AnthropicMessagesHandler()
        guardrail = MockTextModifyingGuardrail(
            guardrail_name="pii-redactor",
            redact_map={
                "John Smith": "[REDACTED_NAME]",
                "555-1234": "[REDACTED_PHONE]",
            },
        )

        data = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": "Call John Smith at 555-1234"},
            ],
        }

        with patch.dict(sys.modules, {"litellm.proxy.proxy_server": _mock_proxy_server()}):
            result = await handler.process_input_messages(
                data=data, guardrail_to_apply=guardrail
            )

        # Text modifications must be applied
        assert result["messages"][0]["content"] == "Call [REDACTED_NAME] at [REDACTED_PHONE]"

    @pytest.mark.asyncio
    async def test_text_modifying_guardrail_with_native_tools(self):
        """Text modifications work correctly alongside Anthropic-native tools."""
        handler = AnthropicMessagesHandler()
        guardrail = MockTextModifyingGuardrail(
            guardrail_name="pii-redactor",
            redact_map={"password123": "[REDACTED]"},
        )

        data = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": "My password is password123"},
            ],
            "tools": [
                {"type": "bash_20250124", "name": "bash"},
                {"type": "text_editor_20250124", "name": "text_editor"},
            ],
        }

        with patch.dict(sys.modules, {"litellm.proxy.proxy_server": _mock_proxy_server()}):
            result = await handler.process_input_messages(
                data=data, guardrail_to_apply=guardrail
            )

        # Text must be modified
        assert result["messages"][0]["content"] == "My password is [REDACTED]"
        # Tools must remain in Anthropic format
        assert len(result["tools"]) == 2
        assert result["tools"][0]["type"] == "bash_20250124"
        assert result["tools"][1]["type"] == "text_editor_20250124"

    @pytest.mark.asyncio
    async def test_guardrail_is_actually_called(self):
        """Verify the guardrail's apply_guardrail method is invoked."""
        handler = AnthropicMessagesHandler()
        guardrail = MockRecordingGuardrail(guardrail_name="recorder")

        data = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "hello world"}],
            "tools": [
                {"type": "bash_20250124", "name": "bash"},
            ],
        }

        with patch.dict(sys.modules, {"litellm.proxy.proxy_server": _mock_proxy_server()}):
            await handler.process_input_messages(
                data=data, guardrail_to_apply=guardrail
            )

        # Guardrail must have been called exactly once
        assert guardrail.call_count == 1
        # Guardrail must have received the text content
        assert guardrail.received_inputs is not None
        assert "hello world" in guardrail.received_inputs.get("texts", [])
        # Guardrail must have received tools in OpenAI format for inspection
        tools = guardrail.received_inputs.get("tools", [])
        assert len(tools) == 1
        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["name"] == "bash"

    @pytest.mark.asyncio
    async def test_guardrail_receives_all_message_texts(self):
        """Guardrail receives text from all messages in the conversation."""
        handler = AnthropicMessagesHandler()
        guardrail = MockRecordingGuardrail(guardrail_name="recorder")

        data = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": "first message"},
                {"role": "assistant", "content": "response"},
                {"role": "user", "content": "second message"},
            ],
        }

        with patch.dict(sys.modules, {"litellm.proxy.proxy_server": _mock_proxy_server()}):
            await handler.process_input_messages(
                data=data, guardrail_to_apply=guardrail
            )

        texts = guardrail.received_inputs.get("texts", [])
        assert "first message" in texts
        assert "response" in texts
        assert "second message" in texts

    @pytest.mark.asyncio
    async def test_text_modifying_guardrail_with_multimodal_content(self):
        """Text modifications work with list-format content (multimodal messages)."""
        handler = AnthropicMessagesHandler()
        guardrail = MockTextModifyingGuardrail(
            guardrail_name="redactor",
            redact_map={"sensitive_data": "[REDACTED]"},
        )

        data = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Here is sensitive_data for you"},
                        {"type": "text", "text": "More sensitive_data here"},
                    ],
                },
            ],
        }

        with patch.dict(sys.modules, {"litellm.proxy.proxy_server": _mock_proxy_server()}):
            result = await handler.process_input_messages(
                data=data, guardrail_to_apply=guardrail
            )

        content = result["messages"][0]["content"]
        assert content[0]["text"] == "Here is [REDACTED] for you"
        assert content[1]["text"] == "More [REDACTED] here"

    @pytest.mark.asyncio
    async def test_blocking_guardrail_with_multimodal_content(self):
        """Blocking guardrail works with list-format content."""
        handler = AnthropicMessagesHandler()
        guardrail = MockBlockingGuardrail(
            guardrail_name="blocker", blocked_keywords=["forbidden"]
        )

        data = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "This contains forbidden content"},
                    ],
                },
            ],
        }

        with patch.dict(sys.modules, {"litellm.proxy.proxy_server": _mock_proxy_server()}):
            with pytest.raises(HTTPException) as exc_info:
                await handler.process_input_messages(
                    data=data, guardrail_to_apply=guardrail
                )
            assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_no_messages_skips_guardrail(self):
        """When there are no messages, guardrail is not called."""
        handler = AnthropicMessagesHandler()
        guardrail = MockRecordingGuardrail(guardrail_name="recorder")

        data = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
        }

        with patch.dict(sys.modules, {"litellm.proxy.proxy_server": _mock_proxy_server()}):
            result = await handler.process_input_messages(
                data=data, guardrail_to_apply=guardrail
            )

        assert guardrail.call_count == 0

    @pytest.mark.asyncio
    async def test_tool_removal_and_text_modification_combined(self):
        """Tool removal by guardrail works alongside text being preserved."""
        handler = AnthropicMessagesHandler()

        # Create a guardrail that both removes tools and modifies texts
        class CombinedGuardrail(CustomGuardrail):
            async def apply_guardrail(self, inputs, request_data, input_type, logging_obj=None):
                # Remove bash tool
                tools = inputs.get("tools")
                if tools is not None:
                    inputs["tools"] = [
                        t for t in tools
                        if isinstance(t, dict)
                        and t.get("function", {}).get("name") != "bash"
                    ]
                # Modify text
                texts = inputs.get("texts", [])
                inputs["texts"] = [t.upper() for t in texts]
                return inputs

        guardrail = CombinedGuardrail(guardrail_name="combined")

        data = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "hello world"}],
            "tools": [
                {"type": "bash_20250124", "name": "bash"},
                {"type": "text_editor_20250124", "name": "text_editor"},
            ],
        }

        with patch.dict(sys.modules, {"litellm.proxy.proxy_server": _mock_proxy_server()}):
            result = await handler.process_input_messages(
                data=data, guardrail_to_apply=guardrail
            )

        # Text must be uppercased
        assert result["messages"][0]["content"] == "HELLO WORLD"
        # Only text_editor should remain, in Anthropic format
        assert len(result["tools"]) == 1
        assert result["tools"][0]["type"] == "text_editor_20250124"
        assert result["tools"][0]["name"] == "text_editor"


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
