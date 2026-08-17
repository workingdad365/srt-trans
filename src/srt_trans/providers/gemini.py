"""Google Gemini 프로바이더 구현."""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import ClassVar

from google import genai
from google.genai import types

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

_VERSION_RE = re.compile(r"gemini-(\d+)(?:\.(\d+))?")

# 토큰 계산 전용으로 사용하는 경량 모델 (참조 구현과 동일)
_TOKEN_COUNT_MODEL = "gemini-2.0-flash"


def _model_version(model: str) -> tuple[int, int] | None:
    match = _VERSION_RE.search(model or "")
    if not match:
        return None
    major = int(match.group(1))
    minor = int(match.group(2) or 0)
    return major, minor


def _supports_thinking(model: str) -> bool:
    """Gemini 2.5 이상 계열은 thinking을 지원함."""
    version = _model_version(model)
    if version is None:
        return False
    return version >= (2, 5)


def _supports_thinking_budget(model: str) -> bool:
    """thinking_budget 지정이 가능한 계열(flash)인지 판단함."""
    return _supports_thinking(model) and "flash" in (model or "")


def _translate_error(exc: Exception) -> ProviderError:
    """SDK 예외를 프로바이더 공통 예외로 변환함."""
    message = str(exc)
    lowered = message.lower()
    if "quota" in lowered or "resource_exhausted" in lowered or "429" in lowered:
        return QuotaExceededError(message)
    if (
        "api key" in lowered
        or "unauthenticated" in lowered
        or "permission_denied" in lowered
        or "401" in lowered
        or "403" in lowered
    ):
        return AuthError(message)
    return ProviderError(message)


def _safety_settings() -> list[types.SafetySetting]:
    """자막 번역이 안전 필터에 막히지 않도록 차단을 해제함(참조 구현과 동일)."""
    categories = [
        types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
        types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
    ]
    return [
        types.SafetySetting(category=category, threshold=types.HarmBlockThreshold.BLOCK_NONE)
        for category in categories
    ]


def _response_schema() -> types.Schema:
    return types.Schema(
        type="ARRAY",
        items=types.Schema(
            type="OBJECT",
            properties={
                "index": types.Schema(type="STRING"),
                "content": types.Schema(type="STRING"),
            },
            required=["index", "content"],
        ),
    )


class GeminiProvider(LLMProvider):
    """google-genai SDK 기반 Gemini 프로바이더."""

    info: ClassVar[ProviderInfo] = ProviderInfo(
        id="gemini",
        label="Google Gemini",
        api_key_url="https://aistudio.google.com/app/apikey",
        supports_thinking=True,
        supports_streaming=True,
        default_model="gemini-2.5-flash",
    )

    def __init__(self, api_key: str, model: str, params: GenerationParams | None = None) -> None:
        super().__init__(api_key, model, params)
        self._client: genai.Client | None = None

    # --- 내부 유틸 -------------------------------------------------------

    @property
    def client(self) -> genai.Client:
        if self._client is None:
            http_options = None
            if self.params.timeout:
                # google-genai의 timeout 단위는 밀리초임
                http_options = types.HttpOptions(timeout=int(self.params.timeout * 1000))
            self._client = genai.Client(api_key=self.api_key, http_options=http_options)
        return self._client

    def _config(self, system_instruction: str) -> types.GenerateContentConfig:
        thinking_ok = _supports_thinking(self.model)
        budget_ok = _supports_thinking_budget(self.model)
        params = self.params

        thinking_config = None
        if thinking_ok:
            thinking_config = types.ThinkingConfig(
                include_thoughts=params.thinking,
                thinking_budget=(
                    (params.thinking_budget if params.thinking else 0) if budget_ok else None
                ),
            )

        return types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=_response_schema(),
            safety_settings=_safety_settings(),
            temperature=params.temperature,
            top_p=params.top_p,
            top_k=params.top_k,
            system_instruction=system_instruction,
            thinking_config=thinking_config,
        )

    @staticmethod
    def _to_contents(history: list[Turn], user_text: str) -> list[types.Content]:
        contents: list[types.Content] = []
        for turn in history:
            parts: list[types.Part] = []
            if turn.thought:
                parts.append(types.Part(thought=True, text=turn.thought))
            if turn.text:
                parts.append(types.Part(text=turn.text))
            if parts:
                contents.append(types.Content(role=turn.role, parts=parts))
        contents.append(types.Content(role="user", parts=[types.Part(text=user_text)]))
        return contents

    # --- 조회 계열 -------------------------------------------------------

    @classmethod
    def list_models(cls, api_key: str) -> list[str]:
        if not (api_key or "").strip():
            raise AuthError("Gemini API 키가 필요합니다.")
        try:
            client = genai.Client(api_key=api_key.strip())
            models: list[str] = []
            for model in client.models.list():
                actions = model.supported_actions or []
                if "generateContent" in actions:
                    models.append((model.name or "").replace("models/", ""))
            return sorted(m for m in models if m)
        except Exception as exc:  # noqa: BLE001 - SDK 예외를 공통 예외로 변환
            raise _translate_error(exc) from exc

    @classmethod
    def model_capabilities(cls, model: str) -> ModelCapabilities:
        thinking = _supports_thinking(model)
        notes: list[str] = []
        if thinking and not _supports_thinking_budget(model):
            notes.append("이 모델은 thinking 사용 여부만 지정할 수 있고 예산은 조절되지 않습니다.")
        return ModelCapabilities(
            thinking=thinking,
            thinking_control="budget" if _supports_thinking_budget(model) else ("on_off" if thinking else None),
            temperature=True,
            top_p=True,
            top_k=True,
            streaming=True,
            token_counting=True,
            notes=notes,
        )

    def output_token_limit(self) -> int | None:
        try:
            model_info = self.client.models.get(model=self.model)
            return getattr(model_info, "output_token_limit", None)
        except Exception:  # noqa: BLE001 - 한도 조회 실패는 치명적이지 않음
            return None

    def count_tokens(self, text: str) -> int | None:
        try:
            result = self.client.models.count_tokens(model=_TOKEN_COUNT_MODEL, contents=text)
            return result.total_tokens
        except Exception:  # noqa: BLE001 - 토큰 계산 실패 시 검증을 건너뜀
            return None

    # --- 생성 계열 -------------------------------------------------------

    def generate(
        self,
        *,
        system_instruction: str,
        history: list[Turn],
        user_text: str,
    ) -> Iterator[Chunk]:
        contents = self._to_contents(history, user_text)
        config = self._config(system_instruction)

        try:
            if self.params.streaming:
                yield from self._generate_stream(contents, config)
            else:
                yield from self._generate_once(contents, config)
        except ProviderError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise _translate_error(exc) from exc

    def _generate_stream(
        self, contents: list[types.Content], config: types.GenerateContentConfig
    ) -> Iterator[Chunk]:
        stream = self.client.models.generate_content_stream(
            model=self.model, contents=contents, config=config
        )
        for chunk in stream:
            if getattr(chunk, "prompt_feedback", None):
                raise ContentBlockedError("모델이 응답을 차단했습니다.")
            candidates = getattr(chunk, "candidates", None) or []
            if not candidates:
                continue
            parts = getattr(candidates[0].content, "parts", None) or []
            for part in parts:
                if not part.text:
                    continue
                yield Chunk(text=part.text, is_thought=bool(getattr(part, "thought", False)))

    def _generate_once(
        self, contents: list[types.Content], config: types.GenerateContentConfig
    ) -> Iterator[Chunk]:
        response = self.client.models.generate_content(
            model=self.model, contents=contents, config=config
        )
        if getattr(response, "prompt_feedback", None):
            raise ContentBlockedError("모델이 응답을 차단했습니다.")
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            raise ProviderError("모델이 빈 응답을 반환했습니다.")
        parts = getattr(candidates[0].content, "parts", None) or []
        for part in parts:
            if not part.text:
                continue
            yield Chunk(text=part.text, is_thought=bool(getattr(part, "thought", False)))
