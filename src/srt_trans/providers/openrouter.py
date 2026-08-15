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

# 모델명 뒤에 붙여 라우팅 방식을 바꾸는 변형(variant) 접미사.
# nitro/floor는 provider.sort와 동등하므로, 이미 변형이 붙은 모델(:free 등)에는
# 접미사를 겹치는 대신 sort로 대체함
ROUTE_VARIANTS: dict[str, dict[str, str]] = {
    "nitro": {"label": "속도 우선 (:nitro)", "sort": "throughput"},
    "floor": {"label": "최저가 (:floor)", "sort": "price"},
    "exacto": {"label": "정확도 우선 (:exacto)", "sort": ""},
}
# provider.sort 로 직접 지정할 수 있는 값
PROVIDER_SORTS = ("price", "throughput", "latency")

# OpenRouter가 인정하는 추론 강도. 낮은 것부터 나열함.
# "none"은 추론을 완전히 끔(단, reasoning.mandatory 모델은 거부함)
EFFORT_LEVELS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
# 모델이 지원 단계를 알려주지 않을 때 제공하는 기본 선택지.
# 지원하지 않는 단계는 OpenRouter가 가장 가까운 단계로 자동 매핑함
FALLBACK_EFFORTS = ("none", "low", "medium", "high")

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


def base_model_id(model: str) -> str:
    """라우팅 변형 접미사를 뗀 기본 모델 ID를 반환함.

    `:free` 처럼 모델 자체의 변형은 그대로 두고, 라우팅용 접미사만 제거함.
    """
    for variant in ROUTE_VARIANTS:
        suffix = f":{variant}"
        if model.endswith(suffix):
            return model[: -len(suffix)]
    return model


def _model_meta(model: str) -> dict[str, Any]:
    """모델 하나의 메타데이터를 반환함. 캐시에 없으면 한 번 조회함."""
    model = base_model_id(model)
    with _cache_lock:
        if model in _cache:
            return _cache[model]
    return _fetch_model_metadata().get(model, {})


def fetch_endpoints(model: str) -> list[dict[str, Any]]:
    """모델을 제공하는 엔드포인트(제공자) 목록을 조회함.

    같은 모델이라도 제공자마다 가격, 컨텍스트 길이, 지원 파라미터, 가동률이 다름.
    """
    model_id = base_model_id(model).strip()
    if not model_id:
        return []
    try:
        response = httpx.get(f"{MODELS_URL}/{model_id}/endpoints", timeout=20.0)
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise ProviderError(f"제공자 목록을 가져오지 못했습니다: {exc}") from exc

    endpoints = ((payload.get("data") or {}).get("endpoints")) or []
    result: list[dict[str, Any]] = []
    for item in endpoints:
        tag = item.get("tag")
        if not tag:
            continue
        pricing = item.get("pricing") or {}
        result.append(
            {
                "tag": tag,
                "provider_name": item.get("provider_name") or tag,
                "context_length": item.get("context_length"),
                "max_completion_tokens": item.get("max_completion_tokens"),
                "quantization": item.get("quantization"),
                "uptime_last_30m": item.get("uptime_last_30m"),
                "supported_parameters": item.get("supported_parameters") or [],
                "prompt_price": pricing.get("prompt"),
                "completion_price": pricing.get("completion"),
                "supports_structured_outputs": "structured_outputs"
                in (item.get("supported_parameters") or []),
            }
        )
    return result


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

        thinking = "reasoning" in params or "reasoning_effort" in params
        efforts = cls._effort_choices(meta) if thinking else []
        has_effort = bool(efforts)

        if thinking:
            declared = list(reasoning.get("supported_efforts") or [])
            mandatory = bool(reasoning.get("mandatory"))
            if mandatory:
                notes.append("이 모델은 추론을 끌 수 없습니다(제공자가 필수로 지정함).")
            if declared:
                lowest = min(declared, key=lambda level: cls._effort_rank(level))
                if cls._effort_rank(lowest) >= EFFORT_LEVELS.index("high"):
                    notes.append(
                        f"이 모델이 지원하는 최소 추론 강도가 '{lowest}'입니다. "
                        "추론에 시간이 오래 걸릴 수 있습니다."
                    )
                if default_effort := reasoning.get("default_effort"):
                    notes.append(f"모델 기본 추론 강도는 '{default_effort}'입니다.")
            else:
                notes.append(
                    "이 모델은 지원 단계를 알려주지 않습니다. 지정한 값은 OpenRouter가 "
                    "가장 가까운 단계로 변환합니다."
                )

        if not cls._supports_structured_output(params):
            notes.append(
                "이 모델은 구조화 출력(JSON 스키마)을 지원하지 않아 형식이 어긋날 수 있습니다. "
                "배치 크기를 작게 잡는 것이 안전합니다."
            )
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

    @staticmethod
    def _effort_rank(level: str) -> int:
        """추론 강도의 세기 순위. 모르는 값은 가장 뒤로 보냄."""
        try:
            return EFFORT_LEVELS.index(str(level).strip().lower())
        except ValueError:
            return len(EFFORT_LEVELS)

    @classmethod
    def _effort_choices(cls, meta: dict[str, Any]) -> list[str]:
        """UI에 보여 줄 추론 강도 선택지를 만듦.

        모델이 알려준 단계를 그대로 쓰되, 추론을 끌 수 있는 모델에는 'none'을 더함.
        단계를 알려주지 않는 모델에는 표준 선택지를 제공함
        (지원하지 않는 값은 OpenRouter가 가장 가까운 단계로 변환함).
        """
        reasoning = meta.get("reasoning") or {}
        declared = [str(level) for level in (reasoning.get("supported_efforts") or [])]
        mandatory = bool(reasoning.get("mandatory"))

        choices = list(declared) if declared else [c for c in FALLBACK_EFFORTS]
        if mandatory:
            choices = [c for c in choices if c != "none"]
        elif "none" not in choices:
            # 추론을 끌 수 있는 모델이면 끄기 선택지를 제공함
            choices.append("none")

        return sorted(set(choices), key=cls._effort_rank)

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

    def effective_model(self) -> str:
        """실제로 요청에 보낼 모델 ID(라우팅 변형 접미사 포함)."""
        variant = str(self.params.extra.get("route_variant") or "").strip().lower()
        if variant not in ROUTE_VARIANTS:
            return self.model
        if ":" in self.model:
            # 이미 :free 같은 변형이 붙어 있으면 접미사를 겹치지 않음(sort로 대체됨)
            return self.model
        return f"{self.model}:{variant}"

    def _routing_options(self) -> dict[str, Any]:
        """provider 객체에 담을 라우팅 설정을 구성함."""
        extra = self.params.extra
        options: dict[str, Any] = {}

        providers = [str(p).strip() for p in (extra.get("providers") or []) if str(p).strip()]
        if providers:
            options["order"] = providers
            # 전환 허용 여부는 제공자를 지정했을 때만 의미가 있음
            allow_fallbacks = extra.get("allow_fallbacks")
            if allow_fallbacks is not None:
                options["allow_fallbacks"] = bool(allow_fallbacks)

        variant = str(extra.get("route_variant") or "").strip().lower()
        sort = str(extra.get("provider_sort") or "").strip().lower()
        if not sort and variant in ROUTE_VARIANTS and self.effective_model() == self.model:
            # 접미사를 붙이지 못한 경우 같은 의미의 sort로 대체함
            sort = ROUTE_VARIANTS[variant]["sort"]
        if sort in PROVIDER_SORTS:
            options["sort"] = sort

        if extra.get("deny_data_collection"):
            options["data_collection"] = "deny"

        return options

    def _request_kwargs(self) -> dict[str, Any]:
        """모델이 실제로 지원하는 파라미터만 담음."""
        meta = _model_meta(self.model)
        params = set(meta.get("supported_parameters") or [])
        # 메타데이터가 없으면 최소한의 파라미터만 보냄
        known = bool(params)
        generation = self.params
        kwargs: dict[str, Any] = {}
        provider_options = self._routing_options()

        if self._supports_structured_output(params):
            kwargs["response_format"] = _response_format()
            # JSON 스키마를 지원하는 제공자로만 라우팅하도록 요청함
            provider_options["require_parameters"] = True
        elif "response_format" in params:
            kwargs["response_format"] = {"type": "json_object"}

        extra_body: dict[str, Any] = {}
        if provider_options:
            extra_body["provider"] = provider_options

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
        """실제로 보낼 추론 강도를 정함.

        지원하지 않는 단계는 OpenRouter가 가장 가까운 단계로 변환해 주므로
        요청값을 임의로 바꾸지 않음(낮추려는 의도가 뒤집히지 않도록).
        다만 추론이 필수인 모델에 'none'을 보내면 오류가 나므로 그때만 걸러냄.
        """
        reasoning = meta.get("reasoning") or {}
        requested = (self.params.reasoning_effort or "").strip().lower()

        # 체크박스로 추론을 끈 경우도 'none'으로 취급함
        if not self.params.thinking:
            requested = "none"

        if requested not in EFFORT_LEVELS:
            return None
        if requested == "none" and reasoning.get("mandatory"):
            # 이 모델은 추론을 끌 수 없음
            return None
        return requested

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
            model=self.effective_model(), messages=messages, stream=True, **kwargs
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
            model=self.effective_model(), messages=messages, **kwargs
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
