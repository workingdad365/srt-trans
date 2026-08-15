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
from .srt_utils import dominant_direction, strip_trailing_period

# 배치 하나당 허용하는 최대 재시도 횟수
MAX_BATCH_RETRIES = 5
# 분당 요청 한도(429)에 걸렸을 때 대기 시간(초)
RATE_LIMIT_WAIT_SECONDS = 20
# 토큰 한도 때문에 배치를 줄일 때의 최소값(물리적 제약이라 낮게 잡음)
MIN_BATCH_SIZE = 10
# 응답 부족으로 배치를 줄일 때의 최소값.
# 배치가 작아지면 한 번에 주는 문맥도 줄어 번역 품질이 떨어지므로 너무 낮추지 않음
MIN_ADAPTIVE_BATCH_SIZE = 50
# 요청 대비 이 비율 미만으로 반환되면 '모자란 응답'으로 셈함.
# 조금 모자란 정도는 남은 줄을 다음 배치에서 이어 처리하므로 손실이 없음
SHRINK_THRESHOLD = 0.7
# 모자란 응답이 이만큼 연속되어야 배치를 줄임(1회는 일시적 오류로 보고 넘어감)
SHRINK_AFTER = 2
# 완전한 응답이 이만큼 연속되면 배치를 다시 늘림
GROW_AFTER = 2


class TranslationCancelled(Exception):
    """사용자가 번역을 취소함."""


class TranslationFailed(Exception):
    """복구 불가능한 번역 실패."""


@dataclass
class EngineOptions:
    """번역 엔진 동작 옵션."""

    batch_size: int = 300
    start_index: int = 0
    # 모델이 놓친 종결 마침표를 결과에서 제거함(프롬프트 지시의 보완 장치)
    strip_trailing_period: bool = True


@dataclass
class TranslationResult:
    """번역 결과."""

    subtitles: list[Subtitle]
    translated_count: int = 0
    total: int = 0
    thoughts: list[str] = field(default_factory=list)

    def compose(self) -> str:
        return srt.compose(self.subtitles, reindex=False, strict=False)


def _describe_index(value: Any) -> str:
    """모델이 돌려준 index 값을 로그에 안전하게 표시함.

    모델이 index 자리에 번역문 같은 엉뚱한 값을 넣는 경우가 있어
    숫자로 단정하고 변환하면 안 됨.
    """
    text = str(value)
    if text.lstrip("-").isdigit():
        return f"{int(text) + 1}번"
    snippet = text if len(text) <= 30 else f"{text[:30]}…"
    return f"숫자가 아닌 값 {snippet!r}"


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
        # 프로바이더마다 요구하는 응답 형식이 다르므로 해당 안내를 덧붙임
        self.system_instruction = system_instruction + provider.output_format_instruction()
        self.options = options or EngineOptions()
        self._on_progress = on_progress or (lambda done, total: None)
        self._on_log = on_log or (lambda level, message: None)
        self._cancel = cancel_event or threading.Event()

        self.batch_size = max(1, self.options.batch_size)
        self._output_token_limit: int | None = None
        self._token_check_enabled = True
        # 후처리로 마침표를 제거한 횟수
        self._period_fixes = 0
        # 실패/취소 시에도 부분 결과를 돌려주기 위한 상태
        self._translated: list[Subtitle] = []
        self._completed = 0
        # 배치 크기 조절용 연속 카운터
        self._short_streak = 0
        self._full_streak = 0

    # --- 공개 API --------------------------------------------------------

    def cancel(self) -> None:
        self._cancel.set()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def partial_result(self) -> TranslationResult | None:
        """실패/취소 시점까지 번역된 결과를 반환함. 진행분이 없으면 None."""
        if not self._translated or self._completed <= self.options.start_index:
            return None
        return TranslationResult(
            subtitles=self._translated,
            translated_count=self._completed - self.options.start_index,
            total=len(self._translated),
        )

    @property
    def completed_count(self) -> int:
        """정상적으로 번역을 마친 자막 개수(1-based 이어하기 위치의 직전)."""
        return self._completed

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
        # 토큰 계산 API가 없는 프로바이더는 배치 크기 자동 축소를 건너뜀
        self._token_check_enabled = capabilities.token_counting
        self._output_token_limit = self.provider.output_token_limit() if capabilities.token_counting else None

        start = max(0, min(self.options.start_index, total))
        if start > 0:
            self._log("info", f"{start + 1}번째 자막부터 번역을 시작합니다.")

        cursor = start
        history: list[Turn] = []
        self._progress(cursor, total)
        # 실패 시에도 여기까지 번역된 결과를 꺼낼 수 있도록 보관함
        self._translated = translated
        self._completed = cursor

        while cursor < total:
            self._raise_if_cancelled()

            batch = self._build_batch(subtitles, cursor, total)
            batch = self._fit_batch_to_token_limit(batch)
            batch_end = cursor + len(batch)

            self._log(
                "info",
                f"{cursor + 1}~{batch_end}번 자막 번역 중... (배치 {len(batch)}줄)",
            )

            history, applied = self._process_batch(batch, history, translated, cursor)

            cursor += applied
            self._completed = cursor
            self._progress(cursor, total)

        if self._period_fixes:
            self._log(
                "info",
                f"모델이 남긴 종결 마침표 {self._period_fixes}건을 제거했습니다.",
            )
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
    ) -> tuple[list[Turn], int]:
        """배치 하나를 번역함.

        Returns:
            (다음 배치에 넘길 대화 이력, 실제로 반영한 줄 수)

        모델이 요청한 줄 수보다 적게 돌려주는 일이 흔하므로, 앞에서부터 이어지는
        정상 구간까지는 그대로 반영하고 나머지는 다음 배치로 넘김.
        """
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
                self._progress(cursor, None)
                continue

            try:
                accepted = self._apply_lines(final, batch, translated)
            except TranslationCancelled:
                raise
            except Exception as exc:  # noqa: BLE001 - 이상한 응답 하나로 전체가 죽지 않게 함
                self._log("warning", f"응답을 처리하는 중 문제가 발생했습니다: {exc}")
                self._log("info", "같은 배치를 다시 시도합니다.")
                self._progress(cursor, None)
                continue

            if accepted == 0:
                self._log("warning", "쓸 수 있는 번역이 없어 같은 배치를 다시 시도합니다.")
                self._progress(cursor, None)
                continue

            if accepted < expected:
                self._log(
                    "warning",
                    f"{expected}줄을 요청했으나 {accepted}줄만 정상 반환되었습니다. "
                    f"{cursor + accepted + 1}번부터 이어서 번역합니다.",
                )
            self._adjust_batch_size(accepted, expected)

            # 다음 배치 문맥에는 실제로 반영된 구간만 넘김
            return (
                [
                    Turn(role="user", text=json.dumps(batch[:accepted], ensure_ascii=False)),
                    Turn(role="model", text=json.dumps(final[:accepted], ensure_ascii=False)),
                ],
                accepted,
            )

    def _adjust_batch_size(self, accepted: int, expected: int) -> None:
        """응답 상태에 따라 배치 크기를 조절함.

        - 한 번 모자란 정도는 일시적인 오류로 보고 크기를 유지함
        - 연속으로 모자라면 절반으로 줄임(한 번에 급락시키지 않음)
        - 다시 안정되면 원래 크기까지 서서히 되돌림
        """
        # 마지막 배치는 남은 자막 수만큼이라 '모자란 응답'으로 볼 수 없음
        if expected < self.batch_size:
            return

        if accepted >= expected:
            self._short_streak = 0
            self._full_streak += 1
            if self._full_streak >= GROW_AFTER and self.batch_size < self.options.batch_size:
                new_size = min(self.options.batch_size, max(self.batch_size + 1, int(self.batch_size * 1.5)))
                self._log("info", f"응답이 안정되어 배치 크기를 {self.batch_size}에서 {new_size}로 되돌립니다.")
                self.batch_size = new_size
                self._full_streak = 0
            return

        self._full_streak = 0
        if accepted >= expected * SHRINK_THRESHOLD:
            # 조금 모자란 정도는 이어 처리로 충분하므로 크기를 건드리지 않음
            return

        if self.batch_size <= MIN_ADAPTIVE_BATCH_SIZE:
            # 이미 최소 크기라 더 줄일 수 없음(같은 안내를 반복하지 않음)
            return

        self._short_streak += 1
        if self._short_streak < SHRINK_AFTER:
            self._log("info", "일시적인 오류일 수 있어 배치 크기를 유지한 채 계속합니다.")
            return

        new_size = max(MIN_ADAPTIVE_BATCH_SIZE, self.batch_size // 2)
        if new_size < self.batch_size:
            self._log("info", f"응답이 계속 모자라 배치 크기를 {self.batch_size}에서 {new_size}로 줄입니다.")
            self.batch_size = new_size
        self._short_streak = 0

    @staticmethod
    def _safe_parse(text: str) -> list[dict[str, Any]] | None:
        """부분적으로 수신된 JSON도 최대한 복구해 파싱함.

        Gemini는 최상위 배열을, OpenAI 구조화 출력은 최상위 객체를 반환하므로
        객체로 감싸인 경우 그 안의 배열을 꺼내 동일하게 취급함.
        """
        if not text.strip():
            return None
        try:
            data = json_repair.loads(text)
        except Exception:  # noqa: BLE001 - 복구 실패는 정상 흐름
            return None

        if isinstance(data, dict):
            for value in data.values():
                if isinstance(value, list):
                    data = value
                    break

        if not isinstance(data, list):
            return None
        return [item for item in data if isinstance(item, dict)]

    def _apply_lines(
        self,
        lines: list[dict[str, Any]],
        batch: list[dict[str, str]],
        translated: list[Subtitle],
    ) -> int:
        """번역 결과를 자막 목록에 반영함.

        배치 순서와 어긋나거나 잘못된 항목이 나오면 거기서 멈추고, 그 앞까지
        정상적으로 반영한 줄 수를 반환함. 나머지는 다음 배치에서 다시 요청함.
        """
        accepted = 0

        for position, line in enumerate(lines):
            if position >= len(batch):
                self._log("warning", "요청한 줄 수보다 많은 응답이 와서 초과분은 버립니다.")
                break

            expected_index = batch[position]["index"]
            line_number = int(expected_index) + 1
            index = line.get("index")
            content = line.get("content")

            if index is None or content is None:
                self._log("warning", f"{line_number}번 자막 응답에 index/content가 없습니다.")
                break

            if str(index) != expected_index:
                self._log(
                    "warning",
                    f"자막 번호 순서가 어긋났습니다. 기대 {line_number}번, "
                    f"수신 {_describe_index(index)}.",
                )
                break

            content = str(content)
            if not content.strip() and batch[position]["content"].strip():
                self._log("warning", f"{line_number}번 자막이 빈 값으로 반환되었습니다.")
                break

            # 모델이 종결 마침표를 남긴 경우 여기서 정리함
            if self.options.strip_trailing_period:
                stripped = strip_trailing_period(content)
                if stripped != content:
                    self._period_fixes += 1
                    content = stripped

            target = int(expected_index)
            if dominant_direction(content) == "rtl":
                translated[target].content = f"‫{content}‬"
            else:
                translated[target].content = content
            accepted += 1

        return accepted

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
