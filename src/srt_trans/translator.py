"""프로바이더 독립적인 자막 번역 엔진.

참조 구현(gemini-srt-translator)의 배치 번역 흐름을 계승하되,
LLM 호출은 providers.LLMProvider 인터페이스로 추상화함.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import json_repair
import srt
from srt import Subtitle

from .providers import (
    ContentBlockedError,
    LLMProvider,
    ProviderError,
    QuotaExceededError,
    Turn,
)
from .srt_utils import dominant_direction

# 배치 하나당 허용하는 최대 재시도 횟수
MAX_BATCH_RETRIES = 5
# 분당 요청 한도(429)에 걸렸을 때 대기 시간(초)
RATE_LIMIT_WAIT_SECONDS = 20
# 배치 크기 자동 축소 시 최소값
MIN_BATCH_SIZE = 10


class TranslationCancelled(Exception):
    """사용자가 번역을 취소함."""


class TranslationFailed(Exception):
    """복구 불가능한 번역 실패."""


@dataclass
class EngineOptions:
    """번역 엔진 동작 옵션."""

    batch_size: int = 300
    start_index: int = 0


@dataclass
class TranslationResult:
    """번역 결과."""

    subtitles: list[Subtitle]
    translated_count: int = 0
    total: int = 0
    thoughts: list[str] = field(default_factory=list)

    def compose(self) -> str:
        return srt.compose(self.subtitles, reindex=False, strict=False)


def parse_subtitles(text: str) -> list[Subtitle]:
    """SRT 텍스트를 파싱함."""
    try:
        subtitles = list(srt.parse(text))
    except Exception as exc:  # noqa: BLE001 - srt 라이브러리 예외 종류가 다양함
        raise TranslationFailed(f"SRT 파일을 해석할 수 없습니다: {exc}") from exc
    if not subtitles:
        raise TranslationFailed("자막 항목이 비어 있습니다.")
    return subtitles


class TranslationEngine:
    """자막을 배치 단위로 번역함."""

    def __init__(
        self,
        provider: LLMProvider,
        system_instruction: str,
        options: EngineOptions | None = None,
        *,
        on_progress: Callable[[int, int], None] | None = None,
        on_log: Callable[[str, str], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> None:
        self.provider = provider
        self.system_instruction = system_instruction
        self.options = options or EngineOptions()
        self._on_progress = on_progress or (lambda done, total: None)
        self._on_log = on_log or (lambda level, message: None)
        self._cancel = cancel_event or threading.Event()

        self.batch_size = max(1, self.options.batch_size)
        self._output_token_limit: int | None = None
        self._token_check_enabled = True

    # --- 공개 API --------------------------------------------------------

    def cancel(self) -> None:
        self._cancel.set()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def translate(self, subtitles: list[Subtitle]) -> TranslationResult:
        """자막 전체를 번역해 결과를 반환함."""
        total = len(subtitles)
        translated = [
            Subtitle(
                index=item.index,
                start=item.start,
                end=item.end,
                content=item.content,
                proprietary=item.proprietary,
            )
            for item in subtitles
        ]

        capabilities = self.provider.capabilities()
        self._output_token_limit = capabilities.output_token_limit

        start = max(0, min(self.options.start_index, total))
        if start > 0:
            self._log("info", f"{start + 1}번째 자막부터 번역을 시작합니다.")

        cursor = start
        history: list[Turn] = []
        self._progress(cursor, total)

        while cursor < total:
            self._raise_if_cancelled()

            batch = self._build_batch(subtitles, cursor, total)
            batch = self._fit_batch_to_token_limit(batch)
            batch_end = cursor + len(batch)

            self._log(
                "info",
                f"{cursor + 1}~{batch_end}번 자막 번역 중... (배치 {len(batch)}줄)",
            )

            history = self._process_batch(batch, history, translated, cursor)

            cursor = batch_end
            self._progress(cursor, total)

        self._log("success", "번역이 완료되었습니다.")
        return TranslationResult(subtitles=translated, translated_count=total - start, total=total)

    # --- 내부 구현 -------------------------------------------------------

    def _build_batch(
        self, subtitles: list[Subtitle], cursor: int, total: int
    ) -> list[dict[str, str]]:
        end = min(cursor + self.batch_size, total)
        return [
            {"index": str(i), "content": subtitles[i].content} for i in range(cursor, end)
        ]

    def _fit_batch_to_token_limit(self, batch: list[dict[str, str]]) -> list[dict[str, str]]:
        """출력 토큰 한도를 넘지 않도록 배치 크기를 자동 축소함."""
        if not self._token_check_enabled or not self._output_token_limit:
            return batch

        while len(batch) > MIN_BATCH_SIZE:
            self._raise_if_cancelled()
            payload = json.dumps(batch, ensure_ascii=False)
            token_count = self.provider.count_tokens(payload)
            if token_count is None:
                # 토큰 계산을 지원하지 않으면 이후 검증을 생략함
                self._token_check_enabled = False
                return batch
            if token_count <= self._output_token_limit * 0.9:
                return batch

            new_size = max(MIN_BATCH_SIZE, len(batch) // 2)
            self._log(
                "warning",
                f"토큰 한도 초과({token_count} > {self._output_token_limit}). "
                f"배치 크기를 {len(batch)}에서 {new_size}로 줄입니다.",
            )
            batch = batch[:new_size]
            self.batch_size = new_size
        return batch

    def _process_batch(
        self,
        batch: list[dict[str, str]],
        history: list[Turn],
        translated: list[Subtitle],
        cursor: int,
    ) -> list[Turn]:
        """배치 하나를 번역하고 다음 배치에 넘길 대화 이력을 반환함."""
        user_text = json.dumps(batch, ensure_ascii=False)
        expected = len(batch)
        attempt = 0

        while True:
            self._raise_if_cancelled()
            attempt += 1
            if attempt > MAX_BATCH_RETRIES:
                raise TranslationFailed(
                    f"{cursor + 1}번 자막부터의 배치를 {MAX_BATCH_RETRIES}회 시도했으나 실패했습니다."
                )

            response_text = ""
            thoughts_text = ""
            parsed: list[dict[str, Any]] = []
            applied = 0

            try:
                for chunk in self.provider.generate(
                    system_instruction=self.system_instruction,
                    history=history,
                    user_text=user_text,
                ):
                    self._raise_if_cancelled()
                    if chunk.is_thought:
                        thoughts_text += chunk.text
                        continue

                    response_text += chunk.text
                    candidate = self._safe_parse(response_text)
                    if candidate is None:
                        continue
                    parsed = candidate
                    # 스트리밍 중간 결과로 진행률만 갱신함
                    if len(parsed) > applied:
                        applied = len(parsed)
                        self._progress(cursor + min(applied, expected), None)

            except ContentBlockedError as exc:
                raise TranslationFailed(
                    "모델이 응답을 차단했습니다. 줄거리/추가 지시문 또는 배치 크기를 조정한 뒤 "
                    f"다시 시도하세요. ({exc})"
                ) from exc
            except QuotaExceededError as exc:
                self._log(
                    "warning",
                    f"요청 한도에 걸렸습니다. {RATE_LIMIT_WAIT_SECONDS}초 대기 후 재시도합니다. ({exc})",
                )
                self._sleep_interruptible(RATE_LIMIT_WAIT_SECONDS)
                attempt -= 1  # 한도 대기는 재시도 횟수에서 제외함
                continue
            except ProviderError as exc:
                self._log("error", f"요청 실패: {exc}")
                self._log("info", "같은 배치를 다시 시도합니다.")
                continue

            final = self._safe_parse(response_text)
            if final is None:
                self._log("warning", "응답을 JSON으로 해석하지 못했습니다. 다시 시도합니다.")
                continue

            if len(final) != expected:
                self._log(
                    "warning",
                    f"응답 줄 수가 맞지 않습니다. 기대 {expected}줄, 수신 {len(final)}줄. 다시 시도합니다.",
                )
                continue

            if not self._apply_lines(final, batch, translated):
                self._log("info", "같은 배치를 다시 시도합니다.")
                continue

            model_turn = Turn(role="model", text=response_text)
            return [
                Turn(role="user", text=user_text),
                model_turn,
            ]

    @staticmethod
    def _safe_parse(text: str) -> list[dict[str, Any]] | None:
        """부분적으로 수신된 JSON도 최대한 복구해 파싱함."""
        if not text.strip():
            return None
        try:
            data = json_repair.loads(text)
        except Exception:  # noqa: BLE001 - 복구 실패는 정상 흐름
            return None
        if not isinstance(data, list):
            return None
        return [item for item in data if isinstance(item, dict)]

    def _apply_lines(
        self,
        lines: list[dict[str, Any]],
        batch: list[dict[str, str]],
        translated: list[Subtitle],
    ) -> bool:
        """번역 결과를 자막 목록에 반영함. 검증 실패 시 False를 반환함."""
        valid_indexes = {item["index"] for item in batch}
        source_by_index = {item["index"]: item["content"] for item in batch}

        for line in lines:
            index = line.get("index")
            content = line.get("content")
            if index is None or content is None:
                self._log("warning", "응답에 index/content가 없는 항목이 있습니다.")
                return False

            index = str(index)
            if index not in valid_indexes:
                self._log("warning", f"요청하지 않은 자막 번호가 반환되었습니다: {index}")
                return False

            content = str(content)
            if not content.strip() and source_by_index[index].strip():
                self._log("warning", f"{int(index) + 1}번 자막이 빈 값으로 반환되었습니다.")
                return False

            position = int(index)
            if dominant_direction(content) == "rtl":
                translated[position].content = f"‫{content}‬"
            else:
                translated[position].content = content

        return True

    # --- 보조 ------------------------------------------------------------

    def _progress(self, done: int, total: int | None) -> None:
        self._on_progress(done, total if total is not None else -1)

    def _log(self, level: str, message: str) -> None:
        self._on_log(level, message)

    def _raise_if_cancelled(self) -> None:
        if self._cancel.is_set():
            raise TranslationCancelled("번역이 취소되었습니다.")

    def _sleep_interruptible(self, seconds: float) -> None:
        """취소 이벤트를 감지하며 대기함."""
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self._raise_if_cancelled()
            time.sleep(min(0.5, max(0.0, deadline - time.monotonic())))
