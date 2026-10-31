"""Qt 메인 스레드를 점유하지 않는 조회 및 번역 작업."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QThread


class BackgroundTask(QThread):
    """finished 이후에만 결과를 읽고 객체를 해제하는 단발성 작업."""

    def __init__(self, kind: str, token: Any, operation: Callable[[], Any], parent=None) -> None:
        super().__init__(parent)
        self.kind = kind
        self.token = token
        self.operation = operation
        self.result: Any = None
        self.error = ""

    def run(self) -> None:
        try:
            self.result = self.operation()
        except Exception as exc:
            self.error = str(exc)
        finally:
            # API 키 등을 캡처한 호출 객체는 작업 종료 즉시 해제함
            self.operation = lambda: None
