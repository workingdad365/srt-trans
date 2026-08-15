"""프로바이더 레지스트리.

새 프로바이더 추가 절차:
1. base.LLMProvider를 상속한 클래스를 이 패키지에 작성함
2. _PROVIDERS에 등록함
"""

from __future__ import annotations

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
from .gemini import GeminiProvider
from .openai_provider import OpenAIProvider
from .openrouter import OpenRouterProvider

_PROVIDERS: dict[str, type[LLMProvider]] = {
    GeminiProvider.info.id: GeminiProvider,
    OpenAIProvider.info.id: OpenAIProvider,
    OpenRouterProvider.info.id: OpenRouterProvider,
}

DEFAULT_PROVIDER = GeminiProvider.info.id


def list_providers() -> list[ProviderInfo]:
    """등록된 모든 프로바이더의 메타데이터를 반환함."""
    return [cls.info for cls in _PROVIDERS.values()]


def get_provider_class(provider_id: str) -> type[LLMProvider]:
    """프로바이더 ID로 클래스를 조회함."""
    try:
        return _PROVIDERS[provider_id]
    except KeyError as exc:
        raise ProviderError(f"지원하지 않는 프로바이더입니다: {provider_id}") from exc


def create_provider(
    provider_id: str, api_key: str, model: str, params: GenerationParams | None = None
) -> LLMProvider:
    """프로바이더 인스턴스를 생성함."""
    return get_provider_class(provider_id)(api_key=api_key, model=model, params=params)


def model_capabilities(provider_id: str, model: str) -> ModelCapabilities:
    """프로바이더/모델 조합의 기능 정보를 조회함(네트워크 호출 없음)."""
    return get_provider_class(provider_id).model_capabilities(model)


__all__ = [
    "AuthError",
    "Chunk",
    "ContentBlockedError",
    "DEFAULT_PROVIDER",
    "GeminiProvider",
    "GenerationParams",
    "LLMProvider",
    "ModelCapabilities",
    "OpenAIProvider",
    "OpenRouterProvider",
    "ProviderError",
    "ProviderInfo",
    "QuotaExceededError",
    "Turn",
    "create_provider",
    "get_provider_class",
    "list_providers",
    "model_capabilities",
]
