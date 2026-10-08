"""Anthropic 파라미터 제약과 SDK 응답 경계의 회귀 검증."""

import json
import unittest
from unittest.mock import patch

import anthropic
import httpx2 as httpx

from srt_trans.providers import (
    AnthropicProvider,
    AuthError,
    ContentBlockedError,
    GenerationParams,
    ProviderError,
    QuotaExceededError,
    Turn,
)
from srt_trans.translator import EngineOptions, TranslationEngine, parse_subtitles


MODEL = "claude-sonnet-4-6"
RESULT = '{"translations":[{"index":"0","content":"안녕하세요."}]}'


def message(text=RESULT, reason="end_turn"):
    return {
        "id": "msg_test", "type": "message", "role": "assistant", "model": MODEL,
        "content": [
            {"type": "thinking", "thinking": "사고 요약", "signature": "test-signature"},
            {"type": "text", "text": text},
        ],
        "stop_reason": reason, "stop_sequence": None,
        "usage": {"input_tokens": 20, "output_tokens": 30},
    }


def sse(text=RESULT, reason="end_turn"):
    initial = message("", None)
    initial["content"] = []
    events = [
        {"type": "message_start", "message": initial},
        {"type": "content_block_start", "index": 0,
         "content_block": {"type": "thinking", "thinking": "", "signature": ""}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "thinking_delta", "thinking": "사고 요약"}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "signature_delta", "signature": "test-signature"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1,
         "content_block": {"type": "text", "text": ""}},
    ]
    for fragment in (text[:10], text[10:]):
        events.append({"type": "content_block_delta", "index": 1,
                       "delta": {"type": "text_delta", "text": fragment}})
    events.append({"type": "content_block_stop", "index": 1})
    if reason is not None:
        events.extend([
            {"type": "message_delta", "delta": {"stop_reason": reason, "stop_sequence": None},
             "usage": {"output_tokens": 30}},
            {"type": "message_stop"},
        ])
    return "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events)


class AnthropicProviderTests(unittest.TestCase):
    def provider(self, *, model=MODEL, handler=None, **params):
        provider = AnthropicProvider("test-key", model, GenerationParams(**params))
        if handler is not None:
            provider._client = anthropic.Anthropic(
                api_key="test-key", max_retries=0, timeout=600.0,
                http_client=httpx.Client(transport=httpx.MockTransport(handler)),
            )
            self.addCleanup(provider.client.close)
        return provider

    def test_manual_budget_leaves_room_for_translation_and_omits_sampling(self):
        for model in ("claude-3-7-sonnet-20250219", "claude-opus-4-20250514", "claude-haiku-4-5-20251001"):
            for budget in (0, 1000, 2048, 100000):
                with self.subTest(model=model, budget=budget):
                    provider = self.provider(model=model, thinking_budget=budget, temperature=0.4, top_p=0.8, top_k=10)
                    request = provider._request_kwargs()
                    self.assertGreaterEqual(request["thinking"]["budget_tokens"], 1024)
                    self.assertGreaterEqual(provider.output_token_limit(), 4096)
                    self.assertLess(request["thinking"]["budget_tokens"], request["max_tokens"])
                    self.assertTrue({"temperature", "top_p", "top_k"}.isdisjoint(request))

    def test_sampling_is_bounded_and_mutually_exclusive(self):
        provider = self.provider(model="claude-sonnet-4-5", thinking=False, temperature=1.5, top_p=0.8)
        request = provider._request_kwargs()
        self.assertEqual(request["temperature"], 1.0)
        self.assertNotIn("top_p", request)
        self.assertNotIn("thinking", request)
        provider.params.temperature = None
        self.assertEqual(provider._request_kwargs()["top_p"], 0.8)

    def test_adaptive_models_never_receive_manual_budget_or_sampling(self):
        for model in (MODEL, "claude-opus-4-7", "claude-sonnet-5-5", "claude-opus-5-5", "claude-mythos-preview"):
            with self.subTest(model=model):
                request = self.provider(model=model, reasoning_effort="high", temperature=0.4, top_k=5)._request_kwargs()
                self.assertEqual(request["thinking"], {"type": "adaptive"})
                self.assertEqual(request["output_config"]["effort"], "high")
                self.assertTrue({"temperature", "top_p", "top_k"}.isdisjoint(request))

    def test_disabling_thinking_respects_model_constraints(self):
        for model, mode in ((MODEL, "disabled"), ("claude-sonnet-5-5", "between_tools"), ("claude-opus-5-5", "adaptive")):
            with self.subTest(model=model):
                provider = self.provider(model=model, reasoning_effort="none")
                self.assertEqual(provider._request_kwargs()["thinking"]["type"], mode)
                self.assertEqual(provider._request_kwargs()["output_config"]["effort"], "low")
        self.assertNotIn("none", AnthropicProvider.model_capabilities("claude-opus-5-5").effort_choices)
        self.assertNotIn("xhigh", AnthropicProvider.model_capabilities(MODEL).effort_choices)

    def test_legacy_models_do_not_receive_unsupported_features(self):
        request = self.provider(model="claude-3-haiku-20240307")._request_kwargs()
        self.assertNotIn("thinking", request)
        self.assertNotIn("output_config", request)
        self.assertLessEqual(request["max_tokens"], 4096)

    def test_translation_preserves_timing_and_excludes_thoughts_in_both_modes(self):
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                def handler(request):
                    if streaming:
                        return httpx.Response(200, text=sse(), headers={"content-type": "text/event-stream"})
                    return httpx.Response(200, json=message())
                provider = self.provider(streaming=streaming, handler=handler)
                source = parse_subtitles("1\n00:00:01,000 --> 00:00:02,500\nHello.\n")
                result = TranslationEngine(provider, "Translate", EngineOptions(batch_size=1)).translate(source)
                self.assertEqual(result.subtitles[0].content, "안녕하세요")
                self.assertEqual(result.subtitles[0].start, source[0].start)
                self.assertEqual(result.subtitles[0].end, source[0].end)
                self.assertEqual(source[0].content, "Hello.")

    def test_refused_truncated_empty_and_disconnected_responses_fail(self):
        cases = [("refusal", RESULT, ContentBlockedError), ("max_tokens", RESULT, ProviderError),
                 ("model_context_window_exceeded", RESULT, ProviderError),
                 ("end_turn", "  ", ProviderError), (None, RESULT, ProviderError)]
        for streaming in (False, True):
            for reason, text, error in cases:
                with self.subTest(streaming=streaming, reason=reason, text=text):
                    def handler(request):
                        if streaming:
                            return httpx.Response(200, text=sse(text, reason), headers={"content-type": "text/event-stream"})
                        return httpx.Response(200, json=message(text, reason))
                    provider = self.provider(streaming=streaming, handler=handler)
                    with self.assertRaises(error):
                        list(provider.generate(system_instruction="Translate", history=[], user_text="Hello"))

    def test_history_does_not_replay_unsigned_thinking(self):
        def handler(request):
            body = json.loads(request.content)
            self.assertNotIn("system", [turn["role"] for turn in body["messages"]])
            self.assertNotIn("unsigned secret", request.content.decode())
            self.assertEqual(body["messages"][-2]["role"], "assistant")
            return httpx.Response(200, json=message())
        provider = self.provider(streaming=False, handler=handler)
        chunks = list(provider.generate(system_instruction="Translate", user_text="Next",
                     history=[Turn("user", "Hello"), Turn("model", RESULT, "unsigned secret")]))
        self.assertEqual("".join(c.text for c in chunks if c.is_thought), "사고 요약")
        self.assertEqual("".join(c.text for c in chunks if not c.is_thought), RESULT)

    def test_http_errors_keep_engine_error_categories(self):
        for status, error in ((401, AuthError), (403, AuthError), (429, QuotaExceededError), (500, ProviderError)):
            with self.subTest(status=status):
                provider = self.provider(handler=lambda request: httpx.Response(
                    status, json={"type": "error", "error": {"type": "api_error", "message": "test error"}}))
                with self.assertRaises(error):
                    list(provider.generate(system_instruction="Translate", history=[], user_text="Hello"))
                with self.assertRaises(error):
                    provider.count_tokens("Hello")

    def test_model_listing_consumes_pages_and_removes_duplicates(self):
        def handler(request):
            ids = [MODEL, "not-chat"] if "after_id" not in request.url.params else [MODEL, "claude-haiku-4-5"]
            return httpx.Response(200, json={
                "data": [{"id": model, "type": "model", "display_name": model,
                          "created_at": "2026-01-01T00:00:00Z"} for model in ids],
                "has_more": "after_id" not in request.url.params,
                "first_id": ids[0], "last_id": ids[-1],
            })
        client = anthropic.Anthropic(api_key="test-key", http_client=httpx.Client(transport=httpx.MockTransport(handler)))
        with patch("srt_trans.providers.anthropic_provider.anthropic.Anthropic", return_value=client):
            self.assertEqual(AnthropicProvider.list_models("test-key"), ["claude-haiku-4-5", MODEL])


if __name__ == "__main__":
    unittest.main()
