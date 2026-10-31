"""데스크톱 서비스의 파일 보존과 번역 실행 경계를 검증함."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from srt_trans.application import (
    TranslateRequest, build_tmdb_context, default_output_path, load_source,
    run_translation, save_result, validate_output_path,
)
from srt_trans.jobs import Job
from srt_trans.providers import Chunk, ContentBlockedError, LLMProvider, ModelCapabilities
from srt_trans.tmdb import TMDBItem
from srt_trans.translator import TranslationFailed, parse_subtitles

SOURCE = (
    "1\n00:00:01,000 --> 00:00:02,000\nHello.\n\n"
    "2\n00:00:03,000 --> 00:00:04,000\nGoodbye.\n\n"
)


class FakeProvider(LLMProvider):
    def __init__(self):
        super().__init__("test-key", "test-model")
        self.calls = []
        self.before_generate = None

    @classmethod
    def list_models(cls, api_key):
        return ["test-model"]

    def generate(self, *, system_instruction, history, user_text):
        self.calls.append((system_instruction, json.loads(user_text)))
        if self.before_generate:
            self.before_generate(len(self.calls))
        rows = [dict(row, content=f"번역 {row['index']}입니다.") for row in json.loads(user_text)]
        yield Chunk(text=json.dumps(rows, ensure_ascii=False))


class ApplicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.path = self.root / "Movie.2024.en.srt"
        self.path.write_text(SOURCE, encoding="utf-8-sig")
        self.source = load_source(self.path)
        self.target = default_output_path(self.source)
        self.job = Job(self.source.name, self.target.name)
        self.provider = FakeProvider()
        self.factory = self.enterContext(patch("srt_trans.application.create_provider", return_value=self.provider))
        self.enterContext(patch("srt_trans.application.model_capabilities", return_value=ModelCapabilities()))

    def run_job(self, **values):
        request = TranslateRequest(model="test-model", **values)
        run_translation(self.job, source=self.source, payload=request, api_key="test-key", output_path=self.target)
        return request

    def test_local_source_bom_and_legacy_encoding(self):
        self.assertEqual(self.source.subtitle_count, 2)
        self.assertEqual(self.source.text, SOURCE)
        self.path.write_bytes(SOURCE.replace("Hello.", "안녕.").encode("cp949"))
        self.assertIn("안녕.", load_source(self.path).text)

    def test_empty_wrong_extension_and_oversized_files_are_rejected(self):
        for name, content in (("empty.srt", ""), ("movie.txt", SOURCE), ("invalid.srt", "not subtitles")):
            with self.subTest(name=name):
                path = self.root / name
                path.write_text(content)
                with self.assertRaises((ValueError, TranslationFailed)):
                    load_source(path)
        with patch("srt_trans.application.MAX_SOURCE_BYTES", 1):
            with self.assertRaisesRegex(ValueError, "20MB"):
                load_source(self.path)

    def test_translation_keeps_timing_and_uses_manual_context(self):
        self.run_job(story_context="수동 줄거리", tmdb_context="사용하면 안 되는 줄거리", temperature=1.0)
        self.assertEqual(self.job.status, "completed")
        self.assertEqual(self.job.done, 2)
        output = parse_subtitles(self.target.read_text(encoding="utf-8"))
        self.assertEqual([row.content for row in output[:2]], ["번역 0입니다", "번역 1입니다"])
        self.assertEqual([(row.start, row.end) for row in output[:2]],
                         [(row.start, row.end) for row in parse_subtitles(SOURCE)])
        self.assertIn("수동 줄거리", self.provider.calls[0][0])
        self.assertNotIn("사용하면 안 되는 줄거리", self.provider.calls[0][0])
        self.assertIsNone(self.factory.call_args.kwargs["params"].temperature)
        self.assertEqual(self.path.read_text(encoding="utf-8-sig"), SOURCE)

    def test_context_fallback_and_persisted_settings_exclude_work_metadata(self):
        request = TranslateRequest(model="test-model", story_context=" ", tmdb_context="TMDB", title="Movie", start_index=1)
        self.assertEqual(request.resolve_context(), ("TMDB", "tmdb"))
        self.assertNotIn("story_context", request.settings())
        self.assertNotIn("tmdb_context", request.settings())
        self.assertNotIn("start_index", request.settings())
        self.assertNotIn("title", request.settings())
        self.assertEqual(request.settings()["models"], {"gemini": "test-model"})
        context = build_tmdb_context(TMDBItem(1, "작품", genres=["드라마"], cast=[{"character": "주인공", "name": "배우"}]))
        self.assertIn("[장르] 드라마", context)
        self.assertIn("- 주인공 (배우: 배우)", context)

    def test_failure_saves_only_completed_batches_and_can_resume(self):
        def fail_second(call):
            if call == 2:
                raise ContentBlockedError("blocked")
        self.provider.before_generate = fail_second
        self.run_job(batch_size=1)
        self.assertEqual(self.job.status, "failed")
        self.assertEqual(self.job.done, 1)
        rows = parse_subtitles(self.target.read_text(encoding="utf-8"))
        self.assertEqual([row.content for row in rows], ["번역 0입니다", "Goodbye."])
        resumed_source = load_source(self.target)
        resumed_target = default_output_path(resumed_source)
        self.assertNotEqual(resumed_source.path, resumed_target)
        self.provider.before_generate = None
        request = TranslateRequest(model="test-model", start_index=1)
        run_translation(self.job, source=resumed_source, payload=request, api_key="test-key", output_path=resumed_target)
        self.assertEqual(self.job.status, "completed")
        self.assertEqual(parse_subtitles(resumed_target.read_text(encoding="utf-8"))[0].content, "번역 0입니다")

    def test_cancellation_saves_partial_output_without_credit(self):
        self.provider.before_generate = lambda call: self.job.cancel() if call == 2 else None
        self.run_job(batch_size=1)
        self.assertEqual(self.job.status, "cancelled")
        self.assertEqual(self.job.done, 1)
        text = self.target.read_text(encoding="utf-8")
        self.assertIn("번역 0", text)
        self.assertIn("Goodbye.", text)
        self.assertNotIn("Translated by AI", text)

    def test_cancel_before_start_makes_no_request(self):
        self.job.cancel()
        self.run_job()
        self.factory.assert_not_called()
        self.assertEqual(self.job.status, "cancelled")
        self.assertFalse(self.target.exists())

    def test_save_failure_preserves_previous_file_and_keeps_result_in_memory(self):
        self.target.write_text("previous result", encoding="utf-8")
        with patch("srt_trans.application.os.replace", side_effect=PermissionError("locked")):
            self.run_job()
        self.assertEqual(self.target.read_text(encoding="utf-8"), "previous result")
        self.assertEqual(self.job.status, "completed")
        self.assertIsNone(self.job.output_path)
        self.assertIn("번역 0", self.job.result_text)
        self.assertFalse(list(self.root.glob("*.tmp")))

    def test_no_automatic_save_and_save_as_share_source_protection(self):
        request = TranslateRequest(model="test-model")
        run_translation(self.job, source=self.source, payload=request, api_key="test-key")
        self.assertFalse(self.target.exists())
        self.assertIsNotNone(self.job.result_text)
        with self.assertRaisesRegex(ValueError, "원본"):
            save_result(self.job.result_text, self.path, self.source)
        alias = self.root / "alias.srt"
        alias.hardlink_to(self.path)
        with self.assertRaisesRegex(ValueError, "원본"):
            validate_output_path(alias, self.source)
        save_result(self.job.result_text, self.target, self.source)
        self.assertTrue(self.target.exists())

    def test_bad_start_or_output_path_never_calls_provider(self):
        self.run_job(start_index=99)
        self.assertEqual(self.job.status, "failed")
        self.factory.assert_not_called()
        with self.assertRaises(ValueError):
            validate_output_path(self.root / "missing" / "out.srt", self.source)
        with self.assertRaises(ValueError):
            TranslateRequest(model="test-model", language_code="../out")


if __name__ == "__main__":
    unittest.main()
