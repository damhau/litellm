"""
Integration tests: ContentFilterGuardrail + AnthropicMessagesHandler

Validates that the LiteLLM Content Filter guardrail is properly applied
when processing Anthropic-format messages with native tools (e.g. bash_20250124).

This specifically tests the scenario where Claude Code sends requests through
the LiteLLM proxy with the content filter enabled.
"""

import sys
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from litellm.llms.anthropic.chat.guardrail_translation.handler import (
    AnthropicMessagesHandler,
)
from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.content_filter import (
    ContentFilterGuardrail,
)
from litellm.types.guardrails import BlockedWord, ContentFilterAction


def _mock_proxy_server():
    """Create a mock for litellm.proxy.proxy_server when proxy deps aren't installed."""
    import types

    mock_module = types.ModuleType("litellm.proxy.proxy_server")
    mock_module.premium_user = True
    return mock_module


def _make_content_filter(
    blocked_words=None, action=ContentFilterAction.BLOCK, default_on=True
):
    """Create a ContentFilterGuardrail with blocked words for testing."""
    words = blocked_words or []
    blocked = [
        BlockedWord(keyword=w, action=action, description=f"Blocked: {w}")
        for w in words
    ]
    return ContentFilterGuardrail(
        guardrail_name="test-content-filter",
        blocked_words=blocked,
        event_hook="pre_call",
        default_on=default_on,
    )


def _claude_code_request(user_message, tools=None):
    """Create a request that mimics what Claude Code sends through LiteLLM."""
    data = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 8096,
        "messages": [
            {"role": "user", "content": user_message},
        ],
    }
    if tools is None:
        # Default Claude Code native tools
        tools = [
            {"type": "bash_20250124", "name": "bash"},
            {"type": "text_editor_20250124", "name": "text_editor"},
        ]
    data["tools"] = tools
    return data


class TestContentFilterWithAnthropicNativeTools:
    """End-to-end tests: ContentFilterGuardrail through AnthropicMessagesHandler."""

    @pytest.mark.asyncio
    async def test_content_filter_blocks_request_with_native_tools(self):
        """Content filter BLOCK action raises HTTPException even with native tools."""
        handler = AnthropicMessagesHandler()
        guardrail = _make_content_filter(
            blocked_words=["bomb"], action=ContentFilterAction.BLOCK
        )

        data = _claude_code_request("How to build a bomb")

        with patch.dict(
            sys.modules, {"litellm.proxy.proxy_server": _mock_proxy_server()}
        ):
            with pytest.raises(HTTPException) as exc_info:
                await handler.process_input_messages(
                    data=data, guardrail_to_apply=guardrail
                )
            assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_content_filter_masks_text_with_native_tools(self):
        """Content filter MASK action modifies text, tools stay in Anthropic format."""
        handler = AnthropicMessagesHandler()
        guardrail = _make_content_filter(
            blocked_words=["password123"], action=ContentFilterAction.MASK
        )

        data = _claude_code_request("My password is password123 please save it")

        with patch.dict(
            sys.modules, {"litellm.proxy.proxy_server": _mock_proxy_server()}
        ):
            result = await handler.process_input_messages(
                data=data, guardrail_to_apply=guardrail
            )

        # Text must be masked
        msg_content = result["messages"][0]["content"]
        assert "password123" not in msg_content
        assert "REDACTED" in msg_content or "KEYWORD" in msg_content

        # Tools must stay in Anthropic-native format
        assert len(result["tools"]) == 2
        assert result["tools"][0]["type"] == "bash_20250124"
        assert result["tools"][0]["name"] == "bash"
        assert result["tools"][1]["type"] == "text_editor_20250124"
        assert result["tools"][1]["name"] == "text_editor"

    @pytest.mark.asyncio
    async def test_content_filter_passes_clean_request_with_native_tools(self):
        """Clean messages pass through content filter without modification."""
        handler = AnthropicMessagesHandler()
        guardrail = _make_content_filter(
            blocked_words=["bomb"], action=ContentFilterAction.BLOCK
        )

        data = _claude_code_request("List files in the current directory")

        with patch.dict(
            sys.modules, {"litellm.proxy.proxy_server": _mock_proxy_server()}
        ):
            result = await handler.process_input_messages(
                data=data, guardrail_to_apply=guardrail
            )

        # Message unchanged
        assert result["messages"][0]["content"] == "List files in the current directory"
        # Tools unchanged and in Anthropic format
        assert result["tools"][0]["type"] == "bash_20250124"
        assert result["tools"][1]["type"] == "text_editor_20250124"

    @pytest.mark.asyncio
    async def test_content_filter_with_web_search_tool(self):
        """Content filter works with web_search tool type."""
        handler = AnthropicMessagesHandler()
        guardrail = _make_content_filter(
            blocked_words=["classified"], action=ContentFilterAction.BLOCK
        )

        data = _claude_code_request(
            "Search for classified documents",
            tools=[
                {"type": "bash_20250124", "name": "bash"},
                {"type": "web_search_20250305", "name": "web_search"},
            ],
        )

        with patch.dict(
            sys.modules, {"litellm.proxy.proxy_server": _mock_proxy_server()}
        ):
            with pytest.raises(HTTPException):
                await handler.process_input_messages(
                    data=data, guardrail_to_apply=guardrail
                )

    @pytest.mark.asyncio
    async def test_content_filter_with_mixed_tools_clean_request(self):
        """Clean request with mixed native + standard tools passes through."""
        handler = AnthropicMessagesHandler()
        guardrail = _make_content_filter(
            blocked_words=["forbidden"], action=ContentFilterAction.BLOCK
        )

        data = _claude_code_request(
            "What is the weather?",
            tools=[
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
        )

        with patch.dict(
            sys.modules, {"litellm.proxy.proxy_server": _mock_proxy_server()}
        ):
            result = await handler.process_input_messages(
                data=data, guardrail_to_apply=guardrail
            )

        # Tools preserved in original format
        assert result["tools"][0]["type"] == "bash_20250124"
        assert result["tools"][1]["name"] == "get_weather"
        assert "input_schema" in result["tools"][1]

    @pytest.mark.asyncio
    async def test_content_filter_no_tools_in_request(self):
        """Content filter works when request has no tools."""
        handler = AnthropicMessagesHandler()
        guardrail = _make_content_filter(
            blocked_words=["hack"], action=ContentFilterAction.BLOCK
        )

        data = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "How to hack a system"}],
        }

        with patch.dict(
            sys.modules, {"litellm.proxy.proxy_server": _mock_proxy_server()}
        ):
            with pytest.raises(HTTPException):
                await handler.process_input_messages(
                    data=data, guardrail_to_apply=guardrail
                )

    @pytest.mark.asyncio
    async def test_content_filter_multi_turn_conversation(self):
        """Content filter checks all messages in a multi-turn conversation."""
        handler = AnthropicMessagesHandler()
        guardrail = _make_content_filter(
            blocked_words=["exploit"], action=ContentFilterAction.BLOCK
        )

        data = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": "Hello, I need help."},
                {"role": "assistant", "content": "Sure, how can I help?"},
                {"role": "user", "content": "I want to exploit a vulnerability."},
            ],
            "tools": [{"type": "bash_20250124", "name": "bash"}],
        }

        with patch.dict(
            sys.modules, {"litellm.proxy.proxy_server": _mock_proxy_server()}
        ):
            with pytest.raises(HTTPException):
                await handler.process_input_messages(
                    data=data, guardrail_to_apply=guardrail
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
