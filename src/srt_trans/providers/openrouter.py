"""OpenRouter 프로바이더 구현.

OpenRouter는 여러 벤더의 모델을 하나의 OpenAI 호환 API로 중계함.
모델마다 지원 파라미터가 크게 다르므로(어떤 모델은 temperature조차 받지 않음),
모델 목록 API가 알려주는 supported_parameters를 그대로 사용해 판정함.
- https://openrouter.ai/api/v1/models 는 인증 없이 조회 가능하며 모델별로
  supported_parameters, reasoning.supported_efforts, context_length 등을 제공함
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from typing import Any, ClassVar

import httpx
import openai

from .. import __version__
from .base import (
    AuthError,
    Chunk,
    GenerationParams,
    LLMProvider,
    ModelCapabilities,
    ProviderError,
    ProviderInfo,
    Turn,
)
from .openai_provider import (
    RESULT_KEY,
    _check_finish_reason,
    _response_format,
    _translate_error,
)

BASE_URL = "https://openrouter.ai/api/v1"
MODELS_URL = f"{BASE_URL}/models"
# 모델 메타데이터 캐시 유지 시간(초)
CACHE_TTL = 3600.0

# OpenRouter 랭킹 페이지에 표시되는 선택 헤더
_APP_HEADERS = {
    "HTTP-Referer": "https://github.com/srt-trans",
    "X-Title": f"srt-trans {__version__}",
}

_cache: dict[str, dict[str, Any]] = {}
_cache_time = 0.0
_cache_lock = threading.RLock()


def _fetch_model_metadata(force: bool = False) -> dict[str, dict[str, Any]]:
    """모델 메타데이터를 조회해 캐시함. 실패하면 기존 캐시(또는 빈 값)를 반환함."""
    global _cache_time

    with _cache_lock:
        fresh = _cache and (time.monotonic() - _cache_time) < CACHE_TTL
        if fresh and not force:
            return _cache

    try:
        response = httpx.get(MODELS_URL, timeout=20.0)
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError):
        with _cache_lock:
            return dict(_cache)

    models: dict[str, dict[str, Any]] = {}
    for item in payload.get("data") or []:
        model_id = item.get("id")
        if model_id:
            models[model_id] = item

    with _cache_lock:
        if models:
            _cache.clear()
            _cache.update(models)
            _cache_time = time.monotonic()
        return dict(_cache)


def _model_meta(model: str) -> dict[str, Any]:
    """모델 하나의 메타데이터를 반환함. 캐시에 없으면 한 번 조회함."""
    with _cache_lock:
        if model in _cache:
            return _cache[model]
    return _fetch_model_metadata().get(model, {})


def _is_text_model(item: dict[str, Any]) -> bool:
    """텍스트를 출력하는 모델인지 확인함(이미지 생성 모델 등 제외)."""
    architecture = item.get("architecture") or {}
    outputs = architecture.get("output_modalities") or ["text"]
    return "text" in outputs


class OpenRouterProvider(LLMProvider):
    """OpenRouter(OpenAI 호환 API) 프로바이더."""

    info: ClassVar[ProviderInfo] = ProviderInfo(
        id="openrouter",
        label="OpenRouter",
        api_key_url="https://openrouter.ai/settings/keys",
        supports_thinking=True,
        supports_streaming=True,
        default_model="google/gemini-2.5-flash",
    )

    def __init__(self, api_key: str, model: str, params: GenerationParams | None = None) -> None:
        super().__init__(api_key, model, params)
        self._client: openai.OpenAI | None = None

    @property
    def client(self) -> openai.OpenAI:
        if self._client is None:
            self._client = openai.OpenAI(
                api_key=self.api_key,
                base_url=BASE_URL,
                default_headers=_APP_HEADERS,
                timeout=600.0,
                max_retries=2,
            )
        return self._client

    # --- 조회 계열 -------------------------------------------------------

    @classmethod
    def list_models(cls, api_key: str) -> list[str]:
        if not (api_key or "").strip():
            raise AuthError("OpenRouter API 키가 필요합니다.")
        metadata = _fetch_model_metadata(force=True)
        if not metadata:
            raise ProviderError("OpenRouter 모델 목록을 가져오지 못했습니다.")
        return sorted(
            model_id for model_id, item in metadata.items() if _is_text_model(item)
        )

    @classmethod
    def validate_api_key(cls, api_key: str) -> bool:
        """모델 목록은 인증 없이도 조회되므로 키 자체를 확인함."""
        key = (api_key or "").strip()
        if not key:
            return False
        try:
            response = httpx.get(
                f"{BASE_URL}/key",
                headers={"Authorization": f"Bearer {key}"},
                timeout=15.0,
            )
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    @classmethod
    def model_capabilities(cls, model: str) -> ModelCapabilities:
        meta = _model_meta(model)
        params = set(meta.get("supported_parameters") or [])
        reasoning = meta.get("reasoning") or {}
        notes: list[str] = []

        if not meta:
            notes.append(
                "이 모델의 정보를 가져오지 못했습니다. '모델 목록 불러오기'를 누르면 "
                "지원 파라미터를 정확히 판정합니다."
            )
            # 정보가 없으면 일반적인 설정을 허용해 두고 서버가 오류를 알리게 함
            return ModelCapabilities(
                temperature=True, top_p=True, top_k=False, streaming=True, notes=notes
            )

        efforts = list(reasoning.get("supported_efforts") or [])
        # 허용 단계를 알 수 있을 때만 '제어 가능'으로 봄.
        # reasoning은 지원하지만 단계 목록이 없는 모델은 모델 기본 동작에 맡김
        has_effort = bool(efforts)
        thinking = has_effort or "reasoning" in params
        if thinking and not has_effort:
            notes.append("이 모델은 추론 강도를 지정할 수 없어 모델 기본 설정으로 동작합니다.")

        if not cls._supports_structured_output(params):
            notes.append(
                "이 모델은 구조화 출력(JSON 스키마)을 지원하지 않아 형식이 어긋날 수 있습니다. "
                "배치 크기를 작게 잡는 것이 안전합니다."
            )
        if reasoning.get("mandatory"):
            notes.append("이 모델은 추론을 끌 수 없습니다.")

        return ModelCapabilities(
            thinking=thinking,
            thinking_control="effort" if has_effort else None,
            effort_choices=efforts,
            temperature="temperature" in params,
            top_p="top_p" in params,
            top_k="top_k" in params,
            streaming=True,
            # OpenRouter에는 토큰 계산 API가 없음
            token_counting=False,
            notes=notes,
        )

    @staticmethod
    def _supports_structured_output(params: set[str]) -> bool:
        return "structured_outputs" in params

    def output_token_limit(self) -> int | None:
        meta = _model_meta(self.model)
        top_provider = meta.get("top_provider") or {}
        return top_provider.get("max_completion_tokens")

    def output_format_instruction(self) -> str:
        return (
            f"\nReturn a single JSON object with exactly one key \"{RESULT_KEY}\", whose value is "
            "the array of translated objects described above. Return raw JSON only: no markdown "
            "code fences, no commentary before or after it.\n"
        )

    # --- 생성 계열 -------------------------------------------------------

    def _messages(
        self, system_instruction: str, history: list[Turn], user_text: str
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [{"role": "system", "content": system_instruction}]
        for turn in history:
            if not turn.text:
                continue
            messages.append(
                {"role": "assistant" if turn.role == "model" else "user", "content": turn.text}
            )
        messages.append({"role": "user", "content": user_text})
        return messages

    def _request_kwargs(self) -> dict[str, Any]:
        """모델이 실제로 지원하는 파라미터만 담음."""
        meta = _model_meta(self.model)
        params = set(meta.get("supported_parameters") or [])
        # 메타데이터가 없으면 최소한의 파라미터만 보냄
        known = bool(params)
        generation = self.params
        kwargs: dict[str, Any] = {}
        extra_body: dict[str, Any] = {}

        if self._supports_structured_output(params):
            kwargs["response_format"] = _response_format()
            # JSON 스키마를 지원하는 제공자로만 라우팅하도록 요청함
            extra_body["provider"] = {"require_parameters": True}
        elif "response_format" in params:
            kwargs["response_format"] = {"type": "json_object"}

        if generation.temperature is not None and (not known or "temperature" in params):
            kwargs["temperature"] = generation.temperature
        if generation.top_p is not None and (not known or "top_p" in params):
            kwargs["top_p"] = generation.top_p
        if generation.top_k is not None and "top_k" in params:
            extra_body["top_k"] = generation.top_k

        effort = self._resolve_effort(meta)
        if effort:
            if "reasoning_effort" in params:
                kwargs["reasoning_effort"] = effort
            elif "reasoning" in params:
                extra_body["reasoning"] = {"effort": effort}

        if extra_body:
            kwargs["extra_body"] = extra_body
        return kwargs

    def _resolve_effort(self, meta: dict[str, Any]) -> str | None:
        """모델이 허용하는 값으로 추론 강도를 맞춤.

        허용 단계를 모르는 모델에는 아무것도 보내지 않고 기본 동작에 맡김.
        """
        reasoning = meta.get("reasoning") or {}
        choices = [str(c) for c in (reasoning.get("supported_efforts") or [])]
        requested = (self.params.reasoning_effort or "").strip().lower()

        if not choices:
            return None
        if requested in choices:
            return requested
        if not self.params.thinking:
            # 추론을 끄고 싶으면 가장 낮은 단계를 고름
            for candidate in ("none", "minimal", "low"):
                if candidate in choices:
                    return candidate
        if requested:
            # 요청한 단계가 없으면 중간값으로 대체함
            for candidate in ("medium", "low", "high"):
                if candidate in choices:
                    return candidate
        return None

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
            choices = getattr(chunk, "choices", None)
            if not choices:
                continue
            choice = choices[0]
            if choice.finish_reason:
                finish_reason = choice.finish_reason
            delta = getattr(choice, "delta", None)
            if not delta:
                continue
            # 일부 모델은 사고 과정을 reasoning 필드로 따로 내려줌
            reasoning = getattr(delta, "reasoning", None)
            if reasoning:
                yield Chunk(text=reasoning, is_thought=True)
            text = getattr(delta, "content", None)
            if text:
                yield Chunk(text=text)
        _check_finish_reason(finish_reason)

    def _generate_once(
        self, messages: list[dict[str, str]], kwargs: dict[str, Any]
    ) -> Iterator[Chunk]:
        response = self.client.chat.completions.create(
            model=self.model, messages=messages, **kwargs
        )
        choices = getattr(response, "choices", None)
        if not choices:
            raise ProviderError("모델이 빈 응답을 반환했습니다.")
        choice = choices[0]
        _check_finish_reason(choice.finish_reason)
        content = choice.message.content or ""
        if not content.strip():
            raise ProviderError("모델이 빈 응답을 반환했습니다.")
        yield Chunk(text=content)
