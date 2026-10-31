"""명령행 진입점의 네이티브 앱 연결을 검증함."""

import subprocess
import sys
import unittest
from unittest.mock import patch

from srt_trans.cli import build_parser, main


class CLITests(unittest.TestCase):
    def test_optional_file_is_forwarded_to_desktop(self):
        with patch("srt_trans.gui.main", return_value=0) as gui_main:
            self.assertEqual(main(["한글 자막.srt"]), 0)
            gui_main.assert_called_once_with("한글 자막.srt")
        self.assertIsNone(build_parser().parse_args([]).file)

    def test_version_does_not_import_qt_or_web_server(self):
        code = """
import sys
from srt_trans.cli import main
try:
    main(['--version'])
except SystemExit as exc:
    assert exc.code == 0
assert 'PySide6' not in sys.modules
assert 'uvicorn' not in sys.modules
assert 'srt_trans.server' not in sys.modules
"""
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("srt-trans", result.stdout)


if __name__ == "__main__":
    unittest.main()
