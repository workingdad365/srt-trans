"""번역 작업(Job) 상태 관리 및 이벤트 브로드캐스트.

작업은 백그라운드 스레드에서 실행되고, 진행 상황은 SSE 구독자에게 전달됨.
"""

from __future__ import annotations

import queue
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

JobStatus = Literal["pending", "running", "completed", "failed", "cancelled"]

# 작업당 보관하는 최대 로그 줄 수
MAX_LOG_LINES = 500
# 완료된 작업을 메모리에 유지하는 시간(초)
JOB_TTL_SECONDS = 3600


@dataclass
class Job:
    """번역 작업 하나의 상태."""

    id: str
    source_name: str
    output_name: str
    status: JobStatus = "pending"
    total: int = 0
    done: int = 0
    output_path: str | None = None
    result_text: str | None = None
    error: str | None = None
    logs: list[dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    _subscribers: list[queue.Queue] = field(default_factory=list, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def snapshot(self) -> dict[str, Any]:
        """UI 초기 렌더링용 상태 스냅샷."""
        with self._lock:
            return {
                "id": self.id,
                "status": self.status,
                "source_name": self.source_name,
                "output_name": self.output_name,
                "total": self.total,
                "done": self.done,
                "output_path": self.output_path,
                "error": self.error,
                "logs": list(self.logs),
                "has_result": self.result_text is not None,
            }

    # --- 이벤트 ----------------------------------------------------------

    def subscribe(self) -> queue.Queue:
        subscriber: queue.Queue = queue.Queue(maxsize=1000)
        with self._lock:
            self._subscribers.append(subscriber)
        return subscriber

    def unsubscribe(self, subscriber: queue.Queue) -> None:
        with self._lock:
            if subscriber in self._subscribers:
                self._subscribers.remove(subscriber)

    def publish(self, event_type: str, payload: dict[str, Any]) -> None:
        event = {"type": event_type, **payload}
        with self._lock:
            subscribers = list(self._subscribers)
        for subscriber in subscribers:
            try:
                subscriber.put_nowait(event)
            except queue.Full:
                # 느린 구독자는 이벤트를 건너뜀
                pass

    # --- 상태 변경 -------------------------------------------------------

    def log(self, level: str, message: str) -> None:
        entry = {"level": level, "message": message, "time": time.strftime("%H:%M:%S")}
        with self._lock:
            self.logs.append(entry)
            if len(self.logs) > MAX_LOG_LINES:
                del self.logs[: len(self.logs) - MAX_LOG_LINES]
        self.publish("log", entry)

    def set_progress(self, done: int, total: int | None = None) -> None:
        with self._lock:
            if total is not None and total >= 0:
                self.total = total
            self.done = max(0, min(done, self.total if self.total else done))
            current = {"done": self.done, "total": self.total}
        self.publish("progress", current)

    def set_status(self, status: JobStatus, error: str | None = None) -> None:
        with self._lock:
            self.status = status
            if error:
                self.error = error
            if status in ("completed", "failed", "cancelled"):
                self.finished_at = time.time()
            payload = {
                "status": self.status,
                "error": self.error,
                "output_path": self.output_path,
                "output_name": self.output_name,
                "has_result": self.result_text is not None,
            }
        self.publish("status", payload)

    def set_result(self, text: str, output_path: str | None = None) -> None:
        with self._lock:
            self.result_text = text
            if output_path:
                self.output_path = output_path


class JobManager:
    """작업 생성/조회/실행을 담당함."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.RLock()

    def create(self, source_name: str, output_name: str) -> Job:
        self._cleanup()
        job = Job(id=uuid.uuid4().hex[:12], source_name=source_name, output_name=output_name)
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def start(self, job: Job, runner: Callable[[Job], None]) -> None:
        """작업을 백그라운드 스레드에서 실행함."""

        def _target() -> None:
            try:
                runner(job)
            except Exception as exc:  # noqa: BLE001 - 스레드에서 예외 유실 방지
                job.log("error", f"예상치 못한 오류: {exc}")
                job.set_status("failed", str(exc))
            finally:
                job.publish("end", {})

        thread = threading.Thread(target=_target, name=f"srt-trans-job-{job.id}", daemon=True)
        thread.start()

    def cancel(self, job_id: str) -> bool:
        job = self.get(job_id)
        if not job or job.status not in ("pending", "running"):
            return False
        job.cancel_event.set()
        job.log("warning", "취소 요청을 받았습니다. 진행 중인 배치를 정리합니다.")
        return True

    def _cleanup(self) -> None:
        """오래된 완료 작업을 제거함."""
        now = time.time()
        with self._lock:
            stale = [
                job_id
                for job_id, job in self._jobs.items()
                if job.finished_at and now - job.finished_at > JOB_TTL_SECONDS
            ]
            for job_id in stale:
                del self._jobs[job_id]


job_manager = JobManager()
