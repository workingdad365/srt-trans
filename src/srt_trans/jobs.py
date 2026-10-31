"""백그라운드 번역 작업의 스레드 안전 상태 저장소."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Literal

JobStatus = Literal["pending", "running", "completed", "failed", "cancelled"]
MAX_LOG_LINES = 500


@dataclass
class Job:
    """창에서 조회하는 번역 상태와 취소 신호."""

    source_name: str
    output_name: str
    status: JobStatus = "pending"
    total: int = 0
    done: int = 0
    output_path: str | None = None
    result_text: str | None = None
    error: str | None = None
    logs: list[dict[str, str]] = field(default_factory=list)
    cancel_event: threading.Event = field(default_factory=threading.Event)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "status": self.status,
                "total": self.total,
                "done": self.done,
                "output_path": self.output_path,
                "error": self.error,
                "logs": list(self.logs),
                "has_result": self.result_text is not None,
            }

    def log(self, level: str, message: str) -> None:
        with self._lock:
            self.logs.append({"level": level, "message": message, "time": time.strftime("%H:%M:%S")})
            if len(self.logs) > MAX_LOG_LINES:
                del self.logs[: len(self.logs) - MAX_LOG_LINES]

    def set_progress(self, done: int, total: int | None = None) -> None:
        with self._lock:
            if total is not None and total >= 0:
                self.total = total
            self.done = max(0, min(done, self.total if self.total else done))

    def set_status(self, status: JobStatus, error: str | None = None) -> None:
        with self._lock:
            self.status = status
            self.error = error

    def set_result(self, text: str, output_path: str | None = None) -> None:
        with self._lock:
            self.result_text = text
            self.output_path = output_path

    def cancel(self) -> None:
        if not self.cancel_event.is_set():
            self.cancel_event.set()
            self.log("warning", "취소 요청됨. 진행 중인 API 응답을 정리한 뒤 부분 결과를 저장합니다.")
