"""LLM 프로바이더 추상화 계층.

번역 엔진은 이 인터페이스에만 의존하며, 특정 벤더 SDK를 직접 호출하지 않음.
새 프로바이더 추가 시 LLMProvider를 상속하고 registry에 등록하면 됨.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import ClassVar, Literal


class ProviderError(Exception):
    """프로바이더 호출 중 발생한 일반 오류."""


class QuotaExceededError(ProviderError):
    """할당량/요청 한도 초과."""


class ContentBlockedError(ProviderError):
    """안전 필터 등에 의해 응답이 차단됨."""


class AuthError(ProviderError):
    """API 키가 없거나 유효하지 않음."""


@dataclass(frozen=True)
class ProviderInfo:
    """UI에 노출할 프로바이더 메타데이터."""

    id: str
    label: str
    api_key_url: str = ""
    supports_thinking: bool = False
    supports_streaming: bool = True
    default_model: str = ""


@dataclass
class GenerationParams:
    """생성 파라미터. 지원하지 않는 값은 프로바이더가 무시함."""

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    thinking: bool = True
    thinking_budget: int = 2048
    streaming: bool = True


@dataclass
class Turn:
    """대화 이력 한 턴."""

    role: Literal["user", "model"]
    text: str
    thought: str = ""


@dataclass
class Chunk:
    """스트리밍 응답 조각."""

    text: str = ""
    is_thought: bool = False


@dataclass
class ModelCapabilities:
    """선택된 모델이 지원하는 기능."""

    thinking: bool = False
    thinking_budget: bool = False
    output_token_limit: int | None = None
    extra: dict = field(default_factory=dict)


class LLMProvider(ABC):
    """모든 프로바이더가 구현해야 하는 인터페이스."""

    info: ClassVar[ProviderInfo]

    def __init__(self, api_key: str, model: str, params: GenerationParams | None = None) -> None:
        if not (api_key or "").strip():
            raise AuthError("API 키가 설정되지 않았습니다.")
        self.api_key = api_key.strip()
        self.model = model
        self.params = params or GenerationParams()

    # --- 조회 계열 -------------------------------------------------------

    @classmethod
    @abstractmethod
    def list_models(cls, api_key: str) -> list[str]:
        """텍스트 생성이 가능한 모델 ID 목록을 반환함."""

    @classmethod
    def validate_api_key(cls, api_key: str) -> bool:
        """API 키 유효성 확인. 기본 구현은 모델 목록 조회로 판단함."""
        try:
            cls.list_models(api_key)
            return True
        except ProviderError:
            return False

    def capabilities(self) -> ModelCapabilities:
        """현재 모델의 기능을 반환함. 기본값은 미지원."""
        return ModelCapabilities()

    def count_tokens(self, text: str) -> int | None:
        """입력 텍스트의 토큰 수. 지원하지 않으면 None을 반환함."""
        return None

    # --- 생성 계열 -------------------------------------------------------

    @abstractmethod
    def generate(
        self,
        *,
        system_instruction: str,
        history: list[Turn],
        user_text: str,
    ) -> Iterator[Chunk]:
        """번역 요청을 보내고 응답 조각을 순차적으로 반환함.

        스트리밍이 비활성/미지원이면 완성된 응답을 한 번에 반환해도 됨.
        차단 시 ContentBlockedError, 할당량 초과 시 QuotaExceededError를 발생시켜야 함.
        """
