"""실제 Qt 이벤트 루프에서 데스크톱 작업 수명과 사용자 동작을 검증함."""

import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import tempfile
import threading
import time
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

from PySide6.QtCore import QMimeData, QPoint, QPointF, Qt, QTimer, QUrl
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

from srt_trans.config import ConfigManager
from srt_trans.gui import MainWindow
from srt_trans.tmdb import TMDBItem

from test_application import FakeProvider, SOURCE


class DesktopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.app.setQuitOnLastWindowClosed(False)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.config = ConfigManager(self.root / "config")
        self.config.update({
            "provider": "openai", "api_keys": {"openai": "saved-key"},
            "models": {"openai": "gpt-4o"}, "batch_size": 1, "top_p": 0.8,
            "extra_instruction": "추가 지시", "tmdb_api_key": "tmdb-test-key",
        })
        self.config.save()
        self.path = self.root / "Movie.2024.en.srt"
        self.path.write_text(SOURCE, encoding="utf-8")
        self.provider = FakeProvider()
        self.factory = self.enterContext(patch("srt_trans.application.create_provider", return_value=self.provider))
        self.window = MainWindow(ConfigManager(self.root / "config"))
        self.errors = self.enterContext(patch.object(self.window, "_error"))
        self.release_events = []
        self.window.show()
        self.wait_idle()

    def tearDown(self):
        for event in self.release_events:
            event.set()
        if self.window.runner:
            self.window.job.cancel()
        self.wait_until(lambda: not self.window._tasks and self.window.runner is None)
        with patch("srt_trans.gui.QMessageBox.question", return_value=QMessageBox.StandardButton.Yes):
            self.window.close()
            self.app.processEvents()
        self.window.deleteLater()
        self.app.processEvents()

    def wait_until(self, predicate, timeout=5.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.app.processEvents()
            if predicate():
                return
            QTest.qWait(10)
        self.fail("Qt 작업이 제한 시간 안에 완료되지 않았습니다.")

    def wait_idle(self):
        self.wait_until(lambda: not self.window._tasks and not self.window._model_timer.isActive()
                        and self.window.runner is None)

    def open_source(self):
        self.window.load_file(str(self.path))
        self.wait_idle()
        self.assertIsNotNone(self.window.source)

    def blocking_event(self):
        event = threading.Event()
        self.release_events.append(event)
        return event

    def test_existing_config_is_restored_and_capabilities_follow_model(self):
        self.assertEqual(self.window.selection, ("openai", "gpt-4o"))
        self.assertEqual(self.window.api_key_edit.text(), "saved-key")
        self.assertEqual(self.window.tmdb_key_edit.text(), "tmdb-test-key")
        self.assertEqual(self.window.extra_edit.toPlainText(), "추가 지시")
        self.assertEqual(self.window.batch_spin.value(), 1)
        self.assertTrue(self.window.temperature_edit.isEnabled())
        self.assertFalse(self.window.top_k_edit.isEnabled())
        self.window.model_combo.setEditText("gpt-5")
        QTest.keyClick(self.window.model_combo.lineEdit(), Qt.Key.Key_Return)
        self.wait_idle()
        self.assertFalse(self.window.temperature_edit.isEnabled())
        self.assertTrue(self.window.effort_combo.isEnabled())
        self.assertEqual(self.window.effort_combo.currentText(), "minimal")
        self.assertEqual(ConfigManager(self.root / "config").get_model("openai"), "gpt-5")

    def test_file_dialog_and_drop_populate_local_output_and_reset_context(self):
        with patch("srt_trans.gui.QFileDialog.getOpenFileName", return_value=(str(self.path), "")):
            self.window.open_file_dialog()
        self.wait_idle()
        self.assertEqual(self.window.title_edit.text(), "Movie")
        self.assertEqual(self.window.year_edit.text(), "2024")
        self.assertEqual(self.window.output_edit.text(), str(self.root / "Movie.2024.kor.srt"))
        self.window.story_edit.setPlainText("다른 작품의 컨텍스트")
        second = self.root / "Series.S01E02.srt"
        second.write_text(SOURCE, encoding="utf-8")
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(str(second))])
        # 최소 지원 버전인 Qt 6.11에서도 같은 드롭 이벤트를 생성함
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*QDragEnterEvent.*")
            enter = QDragEnterEvent(QPoint(10, 10), Qt.DropAction.CopyAction, mime,
                                    Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        self.app.sendEvent(self.window.drop_area, enter)
        self.assertTrue(enter.isAccepted())
        drop = QDropEvent(QPointF(10, 10), Qt.DropAction.CopyAction, mime,
                          Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        self.app.sendEvent(self.window.drop_area, drop)
        self.wait_idle()
        self.assertTrue(drop.isAccepted())
        self.assertEqual(self.window.source.path, second)
        self.assertTrue(self.window.series_check.isChecked())
        self.assertEqual(self.window.story_edit.toPlainText(), "")
        self.assertEqual(self.window.start_spin.value(), 1)
        self.assertTrue(self.window.start_button.isEnabled())

    def test_invalid_file_keeps_previous_source(self):
        self.open_source()
        invalid = self.root / "empty.srt"
        invalid.write_text("")
        self.window.load_file(str(invalid))
        self.wait_idle()
        self.errors.assert_called_once()
        self.assertEqual(self.window.source.path, self.path)

    def test_provider_drafts_and_explicit_key_save_keep_other_keys(self):
        self.window.api_key_edit.setText("new-openai-key")
        self.window.provider_combo.setCurrentIndex(self.window.provider_combo.findData("anthropic"))
        self.wait_idle()
        self.window.api_key_edit.setText("anthropic-key")
        self.window.save_keys()
        self.window.provider_combo.setCurrentIndex(self.window.provider_combo.findData("openai"))
        self.wait_idle()
        self.assertEqual(self.window.api_key_edit.text(), "new-openai-key")
        self.assertEqual(self.window.config.get_api_key("openai"), "saved-key")
        self.window.save_keys()
        saved = ConfigManager(self.root / "config")
        self.assertEqual(saved.get_api_key("openai"), "new-openai-key")
        self.assertEqual(saved.get_api_key("anthropic"), "anthropic-key")

    def test_model_lookup_runs_off_ui_thread(self):
        thread_ids = []
        timer_fired = []
        release = self.blocking_event()

        def models(key):
            thread_ids.append(threading.get_ident())
            release.wait(5)
            return ["gpt-4o", "gpt-5", "other-model"]

        with patch("srt_trans.providers.openai_provider.OpenAIProvider.list_models", side_effect=models):
            self.window.load_models()
            QTimer.singleShot(0, lambda: timer_fired.append(True))
            self.wait_until(lambda: timer_fired and thread_ids)
            self.assertNotEqual(thread_ids[0], threading.get_ident())
            self.assertFalse(self.window.load_models_button.isEnabled())
            release.set()
            self.wait_idle()
        self.assertEqual(self.window.model_combo.count(), 3)
        self.assertEqual(self.window.selection[1], "gpt-4o")

    def test_combo_search_filters_each_keystroke_and_only_saves_selected_model(self):
        self.open_source()
        with patch("srt_trans.providers.openai_provider.OpenAIProvider.list_models", return_value=[
            "vendor/gpt-4o", "vendor/gpt-5", "vendor/claude-sonnet",
        ]):
            self.window.load_models()
            self.wait_idle()
        edit = self.window.model_combo.lineEdit()
        edit.setFocus()
        edit.selectAll()
        completer = self.window.model_combo.completer()
        with patch("srt_trans.gui.model_capabilities") as lookup:
            for character, count in (("G", 2), ("P", 2), ("T", 2), ("-", 2), ("5", 1)):
                QTest.keyClicks(edit, character)
                self.assertEqual(completer.completionModel().rowCount(), count)
            QTest.qWait(350)
            lookup.assert_not_called()
        self.assertEqual(edit.text(), "GPT-5")
        self.assertEqual(ConfigManager(self.root / "config").get_model("openai"), "gpt-4o")
        self.assertFalse(self.window.start_button.isEnabled())
        popup = completer.popup()
        self.assertTrue(popup.isVisible())
        index = completer.completionModel().index(0, 0)
        QTest.mouseClick(popup.viewport(), Qt.MouseButton.LeftButton, pos=popup.visualRect(index).center())
        self.wait_idle()
        self.assertEqual(self.window.selection[1], "vendor/gpt-5")
        self.assertEqual(ConfigManager(self.root / "config").get_model("openai"), "vendor/gpt-5")
        self.assertTrue(self.window.start_button.isEnabled())

    def test_model_id_can_be_entered_without_matching_suggestions(self):
        with patch("srt_trans.providers.openai_provider.OpenAIProvider.list_models", return_value=["gpt-4o"]):
            self.window.load_models()
            self.wait_idle()
        edit = self.window.model_combo.lineEdit()
        edit.setFocus()
        edit.selectAll()
        QTest.keyClicks(edit, "custom-model")
        self.assertEqual(self.window.model_combo.completer().completionModel().rowCount(), 0)
        QTest.keyClick(edit, Qt.Key.Key_Return)
        self.wait_idle()
        self.assertEqual(ConfigManager(self.root / "config").get_model("openai"), "custom-model")

    def test_stale_tmdb_response_cannot_leak_into_new_file(self):
        self.open_source()
        release = self.blocking_event()
        started = threading.Event()

        def search(*args, **kwargs):
            started.set()
            release.wait(5)
            return [TMDBItem(123, "이전 작품")]

        with patch("srt_trans.gui.TMDBClient.search", side_effect=search):
            self.window.search_tmdb()
            self.wait_until(started.is_set)
            other = self.root / "Other.srt"
            other.write_text(SOURCE)
            self.window.load_file(str(other))
            self.wait_until(lambda: self.window.source.path == other)
            release.set()
            self.wait_idle()
        self.assertEqual(self.window.tmdb_results.count(), 0)
        self.assertEqual(self.window._tmdb_context, "")

    def test_tmdb_selection_supplies_context(self):
        self.open_source()
        movie = TMDBItem(123, "한국어 제목", original_title="Original", year="2024", overview="줄거리")
        with patch("srt_trans.gui.TMDBClient.search", return_value=[movie]), \
                patch("srt_trans.gui.TMDBClient.details", return_value=movie):
            self.window.search_tmdb()
            self.wait_idle()
            self.window.select_tmdb()
            self.wait_idle()
        self.assertEqual(self.window.title_edit.text(), "한국어 제목")
        self.assertIn("줄거리", self.window._collect_request().resolve_context()[0])
        self.window.story_edit.setPlainText("직접 작성한 줄거리")
        self.assertEqual(self.window._collect_request().resolve_context(), ("직접 작성한 줄거리", "manual"))

    def test_translation_saves_and_enables_save_as(self):
        self.open_source()
        self.window.story_edit.setPlainText("수동 컨텍스트")
        self.window.start_translation()
        self.assertFalse(self.window.tabs.isEnabled())
        self.wait_idle()
        self.assertEqual(self.window.job.status, "completed")
        self.assertEqual(self.window.progress_bar.value(), 1000)
        self.assertTrue(Path(self.window.job.output_path).exists())
        self.assertTrue(self.window.save_result_button.isEnabled())
        self.assertTrue(self.window.open_folder_button.isEnabled())
        copy = self.root / "copy.srt"
        with patch("srt_trans.gui.QFileDialog.getSaveFileName", return_value=(str(copy), "")):
            self.window.save_result_dialog()
        self.assertIn("번역 0", copy.read_text(encoding="utf-8"))
        saved = ConfigManager(self.root / "config").all()
        self.assertNotIn("story_context", saved)
        self.assertEqual(saved["extra_instruction"], "추가 지시")

    def test_overwrite_and_unsaved_result_confirmations_are_respected(self):
        self.open_source()
        target = Path(self.window.output_edit.text())
        target.write_text("old")
        with patch("srt_trans.gui.QMessageBox.question", return_value=QMessageBox.StandardButton.No):
            self.window.start_translation()
        self.assertIsNone(self.window.runner)
        self.assertEqual(target.read_text(), "old")
        self.window.auto_save.setChecked(False)
        self.window.start_translation()
        self.wait_idle()
        self.assertIsNone(self.window.job.output_path)
        with patch("srt_trans.gui.QMessageBox.question", return_value=QMessageBox.StandardButton.No):
            self.assertFalse(self.window.close())
        self.assertTrue(self.window.isVisible())

    def test_cancel_and_close_wait_for_worker_and_preserve_partial_result(self):
        self.open_source()
        release = self.blocking_event()
        second_started = threading.Event()

        def before_generate(call):
            if call == 2:
                second_started.set()
                release.wait(5)

        self.provider.before_generate = before_generate
        self.window.start_translation()
        self.wait_until(second_started.is_set)
        worker = self.window.runner
        with patch("srt_trans.gui.QMessageBox.question", return_value=QMessageBox.StandardButton.Yes):
            self.assertFalse(self.window.close())
        self.assertTrue(self.window.isVisible())
        self.assertTrue(worker.isRunning())
        self.assertTrue(self.window.job.cancel_event.is_set())
        self.assertFalse(self.window.cancel_button.isEnabled())
        release.set()
        self.wait_until(lambda: not self.window.isVisible())
        self.assertIsNone(self.window.runner)
        self.assertEqual(self.window.job.status, "cancelled")
        self.assertEqual(self.window.job.done, 1)
        text = Path(self.window.job.output_path).read_text(encoding="utf-8")
        self.assertIn("번역 0", text)
        self.assertIn("Goodbye.", text)
        self.assertNotIn("Translated by AI", text)

    def test_declining_close_after_lookup_restores_model_controls(self):
        self.open_source()
        self.window.auto_save.setChecked(False)
        self.window.start_translation()
        self.wait_idle()
        release = self.blocking_event()
        started = threading.Event()
        from srt_trans.providers import ModelCapabilities

        def capabilities(*args):
            started.set()
            release.wait(5)
            return ModelCapabilities(temperature=True)

        with patch("srt_trans.gui.model_capabilities", side_effect=capabilities):
            self.window.model_combo.setEditText("another-model")
            QTest.keyClick(self.window.model_combo.lineEdit(), Qt.Key.Key_Return)
            self.wait_until(started.is_set)
            self.assertFalse(self.window.close())
            self.assertTrue(self.window._closing)
            with patch("srt_trans.gui.QMessageBox.question", return_value=QMessageBox.StandardButton.No):
                release.set()
                self.wait_until(lambda: not self.window._closing)
                self.wait_idle()
        self.assertTrue(self.window.isVisible())
        self.assertTrue(self.window.start_button.isEnabled())
        self.assertEqual(self.window._capability_key, self.window.selection)


if __name__ == "__main__":
    unittest.main()
