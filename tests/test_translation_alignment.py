"""불완전한 배치가 자막과 다음 요청 문맥에 섞이지 않는지 검증함."""

import json
import unittest
from datetime import timedelta
from unittest.mock import Mock

from srt import Subtitle

from srt_trans.providers import Chunk, ContentBlockedError, LLMProvider, ModelCapabilities
from srt_trans.translator import EngineOptions, TranslationEngine, TranslationFailed


def source(count):
    return [Subtitle(index=i + 1, start=timedelta(seconds=i), end=timedelta(seconds=i + 1),
                     content=f"Original {i}") for i in range(count)]


def response(items):
    return [Chunk(text=json.dumps({"translations": items}, ensure_ascii=False))]


def make_engine(replies, batch_size):
    provider = Mock(spec=LLMProvider)
    provider.output_format_instruction.return_value = ""
    provider.capabilities.return_value = ModelCapabilities()
    provider.generate.side_effect = replies
    engine = TranslationEngine(provider, "Translate", EngineOptions(batch_size=batch_size))
    return engine, provider


class TranslationAlignmentTests(unittest.TestCase):
    def test_short_renumbered_response_is_not_committed_before_failure(self):
        merged = [{"index": str(i), "content": f"Merged translation {i}"} for i in range(223)]
        engine, _ = make_engine([response(merged), ContentBlockedError("blocked")], 250)
        originals = source(250)
        with self.assertRaises(TranslationFailed):
            engine.translate(originals)
        self.assertEqual(engine.completed_count, 0)
        self.assertIsNone(engine.partial_result())
        self.assertEqual([s.content for s in engine._translated], [s.content for s in originals])

    def test_invalid_batch_restarts_at_same_index_and_never_enters_history(self):
        good = [{"index": str(i), "content": f"번역 {i}"} for i in range(4)]
        merged = [{"index": "0", "content": "앞뒤 자막을 합친 잘못된 번역"}]
        engine, provider = make_engine([response(merged), response(good[:2]), response(good[2:])], 4)
        originals = source(4)
        result = engine.translate(originals)
        requests = provider.generate.call_args_list
        self.assertEqual([row["index"] for row in json.loads(requests[1].kwargs["user_text"])], ["0", "1"])
        self.assertEqual(requests[1].kwargs["history"], [])
        self.assertEqual(json.loads(requests[2].kwargs["history"][1].text), good[:2])
        self.assertEqual([s.content for s in result.subtitles], [f"번역 {i}" for i in range(4)])
        self.assertEqual([(s.index, s.start, s.end) for s in result.subtitles],
                         [(s.index, s.start, s.end) for s in originals])

    def test_invalid_last_entry_cannot_modify_valid_prefix(self):
        malformed = [
            [{"index": "0", "content": "잘못 반영될 앞부분"}, {"index": "0", "content": "중복 번호"}],
            [{"index": "0", "content": "잘못 반영될 앞부분"}, {"index": "2", "content": "건너뛴 번호"}],
            [{"index": "0", "content": "잘못 반영될 앞부분"}, {"index": "1", "content": ""}],
            [{"index": "0", "content": "잘못 반영될 앞부분"}, {"index": "1", "content": 42}],
            [{"index": "0", "content": "잘못 반영될 앞부분"}, {"index": "1"}],
            [{"index": str(i), "content": "초과 응답"} for i in range(3)],
        ]
        for items in malformed:
            with self.subTest(items=items):
                engine, _ = make_engine([response(items), ContentBlockedError("blocked")], 2)
                originals = source(2)
                with self.assertRaises(TranslationFailed):
                    engine.translate(originals)
                self.assertEqual([s.content for s in engine._translated], [s.content for s in originals])
                self.assertEqual(engine.completed_count, 0)

    def test_truncated_json_is_not_repaired_into_a_completed_translation(self):
        truncated = [Chunk(text='{"translations":[{"index":"0","content":"미완성')]
        engine, _ = make_engine([truncated, ContentBlockedError("blocked")], 1)
        with self.assertRaises(TranslationFailed):
            engine.translate(source(1))
        self.assertIsNone(engine.partial_result())

    def test_permanently_bad_single_entry_stops_after_bounded_retries(self):
        engine, provider = make_engine(lambda **kwargs: response([]), 1)
        with self.assertRaises(TranslationFailed):
            engine.translate(source(1))
        self.assertEqual(provider.generate.call_count, 5)
        self.assertIsNone(engine.partial_result())

    def test_later_bad_batch_preserves_only_previously_validated_results(self):
        good = [{"index": str(i), "content": f"정상 {i}"} for i in range(2)]
        bad = [{"index": "2", "content": "병합된 오역"}]
        engine, _ = make_engine([response(good), response(bad), ContentBlockedError("blocked")], 2)
        with self.assertRaises(TranslationFailed):
            engine.translate(source(4))
        partial = engine.partial_result()
        self.assertEqual(partial.translated_count, 2)
        self.assertEqual([s.content for s in partial.subtitles], ["정상 0", "정상 1", "Original 2", "Original 3"])


if __name__ == "__main__":
    unittest.main()
