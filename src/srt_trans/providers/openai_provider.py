"""OpenAI 프로바이더 구현.

GPT-5 이상(및 o 시리즈) 추론 모델과 그 이전 모델은 지원 파라미터가 다름.
- 추론 모델: temperature / top_p / max_tokens 사용 불가(1로 고정), max_completion_tokens와
  reasoning_effort 사용
- 이전 모델: temperature / top_p / max_tokens 사용 가능, reasoning_effort 없음

또한 OpenAI 구조화 출력(strict)은 최상위가 객체여야 하므로, Gemini처럼 최상위 배열
스키마를 쓸 수 없음. {"translations": [...]} 형태로 감싸고 형식 지시문을 덧붙임.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any, ClassVar

import openai

from .base import (
    AuthError,
    Chunk,
    ContentBlockedError,
    GenerationParams,
    LLMProvider,
    ModelCapabilities,
    ProviderError,
    ProviderInfo,
    QuotaExceededError,
    Turn,
)

# 번역 결과를 담는 최상위 키 (엔진은 dict 응답에서 배열을 자동으로 꺼냄)
RESULT_KEY = "translations"

_GPT_VERSION_RE = re.compile(r"^gpt-(\d+)(?:\.(\d+))?", re.IGNORECASE)
_O_SERIES_RE = re.compile(r"^o(\d+)(?:-|$)", re.IGNORECASE)

# 채팅 생성과 무관한 모델을 목록에서 제외하기 위한 키워드
_NON_CHAT_KEYWORDS = (
    "embedding",
    "tts",
    "whisper",
    "dall-e",
    "moderation",
    "audio",
    "realtime",
    "transcribe",
    "image",
    "sora",
    "codex",
    "computer-use",
    "guard",
)


def _gpt_version(model: str) -> tuple[int, int] | None:
    """gpt-5.1-mini → (5, 1). GPT 계열이 아니면 None."""
    match = _GPT_VERSION_RE.match(model or "")
    if not match:
        return None
    return int(match.group(1)), int(match.group(2) or 0)


def _is_o_series(model: str) -> bool:
    return bool(_O_SERIES_RE.match(model or ""))


def is_reasoning_model(model: str) -> bool:
    """GPT-5 이상 또는 o 시리즈인지 판정함.

    이 계열은 temperature/top_p/max_tokens를 받지 않고 reasoning_effort를 사용함.
    """
    version = _gpt_version(model)
    if version is not None:
        return version >= (5, 0)
    return _is_o_series(model)


def effort_choices(model: str) -> list[str]:
    """모델별로 허용되는 reasoning_effort 값 목록을 반환함."""
    if not is_reasoning_model(model):
        return []
    version = _gpt_version(model)
    if version is None:
        # o 시리즈는 minimal/none을 지원하지 않음
        return ["low", "medium", "high"]
    if version >= (5, 1):
        # GPT-5.1부터 최소 단계가 none으로 바뀜
        return ["none", "low", "medium", "high"]
    return ["minimal", "low", "medium", "high"]


def normalize_effort(model: str, effort: str | None) -> str | None:
    """요청된 reasoning_effort를 해당 모델이 받는 값으로 맞춤."""
    choices = effort_choices(model)
    if not choices:
        return None
    if not effort:
        return None
    effort = effort.strip().lower()
    if effort in choices:
        return effort
    # 최소 단계 명칭이 모델마다 다르므로(minimal/none) 서로 대체함
    if effort in ("none", "minimal"):
        for candidate in ("none", "minimal"):
            if candidate in choices:
                return candidate
        return "low"
    return None


def _response_format() -> dict[str, Any]:
    """strict 구조화 출력 스키마.

    strict 모드 제약: 최상위는 객체, 모든 속성은 required, additionalProperties는 false.
    """
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "subtitle_translation",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    RESULT_KEY: {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "index": {"type": "string"},
                                "content": {"type": "string"},
                            },
                            "required": ["index", "content"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": [RESULT_KEY],
                "additionalProperties": False,
            },
        },
    }


def _translate_error(exc: Exception) -> ProviderError:
    """SDK 예외를 프로바이더 공통 예외로 변환함."""
    if isinstance(exc, openai.AuthenticationError | openai.PermissionDeniedError):
        return AuthError(str(exc))
    if isinstance(exc, openai.RateLimitError):
        return QuotaExceededError(str(exc))
    if isinstance(exc, openai.BadRequestError):
        message = str(exc)
        if "content_filter" in message or "content management policy" in message:
            return ContentBlockedError(message)
        return ProviderError(message)
    return ProviderError(str(exc))


class OpenAIProvider(LLMProvider):
    """openai SDK(Chat Completions) 기반 프로바이더."""

    info: ClassVar[ProviderInfo] = ProviderInfo(
        id="openai",
        label="OpenAI",
        api_key_url="https://platform.openai.com/api-keys",
        supports_thinking=True,
        supports_streaming=True,
        default_model="gpt-5.1",
    )

    def __init__(self, api_key: str, model: str, params: GenerationParams | None = None) -> None:
        super().__init__(api_key, model, params)
        self._client: openai.OpenAI | None = None

    @property
    def client(self) -> openai.OpenAI:
        if self._client is None:
            self._client = openai.OpenAI(api_key=self.api_key, timeout=600.0, max_retries=2)
        return self._client

    # --- 조회 계열 -------------------------------------------------------

    @classmethod
    def list_models(cls, api_key: str) -> list[str]:
        if not (api_key or "").strip():
            raise AuthError("OpenAI API 키가 필요합니다.")
        try:
            client = openai.OpenAI(api_key=api_key.strip())
            models: list[str] = []
            for model in client.models.list():
                model_id = model.id or ""
                lowered = model_id.lower()
                if not (lowered.startswith(("gpt-", "chatgpt-")) or _is_o_series(model_id)):
                    continue
                if any(keyword in lowered for keyword in _NON_CHAT_KEYWORDS):
                    continue
                models.append(model_id)
            return sorted(set(models))
        except Exception as exc:  # noqa: BLE001 - SDK 예외를 공통 예외로 변환
            raise _translate_error(exc) from exc

    @classmethod
    def model_capabilities(cls, model: str) -> ModelCapabilities:
        reasoning = is_reasoning_model(model)
        notes: list[str] = []
        if reasoning:
            notes.append(
                "GPT-5 이상/o 시리즈는 temperature·top_p를 지원하지 않습니다(1로 고정). "
                "대신 추론 강도를 사용합니다."
            )
        return ModelCapabilities(
            thinking=reasoning,
            thinking_control="effort" if reasoning else None,
            effort_choices=effort_choices(model),
            temperature=not reasoning,
            top_p=not reasoning,
            # OpenAI Chat Completions에는 top_k가 없음
            top_k=False,
            streaming=True,
            # 토큰 계산 API가 없어 배치 크기 자동 축소는 동작하지 않음
            token_counting=False,
            notes=notes,
        )

    def output_format_instruction(self) -> str:
        return (
            f"\nReturn a single JSON object with exactly one key \"{RESULT_KEY}\", whose value is "
            "the array of translated objects described above. Do not wrap it in anything else.\n"
        )

    # --- 생성 계열 -------------------------------------------------------

    def _messages(
        self, system_instruction: str, history: list[Turn], user_text: str
    ) -> list[dict[str, str]]:
        # 추론 모델은 system 대신 developer 역할을 사용함
        system_role = "developer" if is_reasoning_model(self.model) else "system"
        messages: list[dict[str, str]] = [{"role": system_role, "content": system_instruction}]
        for turn in history:
            # 사고 내용(turn.thought)은 재전송하지 않음. OpenAI는 이를 입력으로 받지 않음
            if not turn.text:
                continue
            role = "assistant" if turn.role == "model" else "user"
            messages.append({"role": role, "content": turn.text})
        messages.append({"role": "user", "content": user_text})
        return messages

    def _request_kwargs(self) -> dict[str, Any]:
        """모델 계열에 맞는 파라미터만 골라 담음."""
        params = self.params
        kwargs: dict[str, Any] = {"response_format": _response_format()}

        if is_reasoning_model(self.model):
            # temperature / top_p / max_tokens 는 전송하지 않음 (400 오류 원인)
            effort = normalize_effort(self.model, params.reasoning_effort)
            if effort is None and not params.thinking:
                # 사고를 끄고 싶은 경우 최소 단계로 낮춤
                effort = normalize_effort(self.model, "none")
            if effort:
                kwargs["reasoning_effort"] = effort
        else:
            if params.temperature is not None:
                kwargs["temperature"] = params.temperature
            if params.top_p is not None:
                kwargs["top_p"] = params.top_p

        return kwargs

    def generate(
        self,
        *,
        system_instruction: str,
        history: list[Turn],
        user_text: str,
    ) -> Iterator[Chunk]:
        messages = self._messages(system_instruction, history, user_text)
        kwargs = self._request_kwargs()

        try:
            if self.params.streaming:
                yield from self._generate_stream(messages, kwargs)
            else:
                yield from self._generate_once(messages, kwargs)
        except ProviderError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise _translate_error(exc) from exc

    def _generate_stream(
        self, messages: list[dict[str, str]], kwargs: dict[str, Any]
    ) -> Iterator[Chunk]:
        stream = self.client.chat.completions.create(
            model=self.model, messages=messages, stream=True, **kwargs
        )
        finish_reason: str | None = None
        for chunk in stream:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            if choice.finish_reason:
                finish_reason = choice.finish_reason
            delta = getattr(choice, "delta", None)
            text = getattr(delta, "content", None) if delta else None
            if text:
                yield Chunk(text=text)
        _check_finish_reason(finish_reason)

    def _generate_once(
        self, messages: list[dict[str, str]], kwargs: dict[str, Any]
    ) -> Iterator[Chunk]:
        response = self.client.chat.completions.create(
            model=self.model, messages=messages, **kwargs
        )
        if not response.choices:
            raise ProviderError("모델이 빈 응답을 반환했습니다.")
        choice = response.choices[0]
        _check_finish_reason(choice.finish_reason)
        content = choice.message.content or ""
        if not content.strip():
            raise ProviderError("모델이 빈 응답을 반환했습니다.")
        yield Chunk(text=content)


def _check_finish_reason(reason: str | None) -> None:
    """비정상 종료 사유를 공통 예외로 바꿈."""
    if reason == "content_filter":
        raise ContentBlockedError("OpenAI 콘텐츠 필터가 응답을 차단했습니다.")
    if reason == "length":
        raise ProviderError(
            "응답이 출력 길이 한도에 걸려 잘렸습니다. 배치 크기를 줄이고 다시 시도하세요."
        )
