"""LLM 프로바이더 추상화 계층.

번역 엔진은 이 인터페이스에만 의존하며, 특정 벤더 SDK를 직접 호출하지 않음.
새 프로바이더 추가 시 LLMProvider를 상속하고 registry에 등록하면 됨.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal


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
    """생성 파라미터. 지원하지 않는 값은 프로바이더가 걸러내고 전송하지 않음.

    프로바이더마다 사고(reasoning) 제어 방식이 다름.
    - Gemini: thinking + thinking_budget(토큰 수)
    - OpenAI: reasoning_effort(단계 값)
    각 프로바이더는 자신이 쓰는 필드만 사용하고 나머지는 무시함.
    """

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    thinking: bool = True
    thinking_budget: int = 2048
    reasoning_effort: str | None = None
    streaming: bool = True
    # 요청 하나가 끝나기를 기다리는 최대 시간(초). None이면 SDK 기본값
    timeout: float | None = None
    # 특정 프로바이더에만 해당하는 옵션(예: OpenRouter의 라우팅 설정).
    # 해당 프로바이더만 읽고 나머지는 무시함
    extra: dict[str, Any] = field(default_factory=dict)


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
    """선택된 모델이 지원하는 기능.

    모델명만으로 판정 가능한 정적 정보이며 네트워크 호출을 하지 않음.
    UI가 지원하지 않는 입력란을 비활성화하는 데 사용함.
    """

    # 사고(reasoning) 기능 유무
    thinking: bool = False
    # 사고 제어 방식: "budget"(토큰 수) | "effort"(단계) | None(제어 불가)
    thinking_control: str | None = None
    # thinking_control이 "effort"일 때 선택 가능한 값
    effort_choices: list[str] = field(default_factory=list)
    # 샘플링 파라미터 지원 여부
    temperature: bool = False
    top_p: bool = False
    top_k: bool = False
    streaming: bool = True
    # 토큰 수 계산 API 제공 여부 (배치 크기 자동 축소에 사용)
    token_counting: bool = False
    # UI에 표시할 참고 사항
    notes: list[str] = field(default_factory=list)

    def unsupported(self, params: GenerationParams) -> list[str]:
        """요청 파라미터 중 이 모델이 지원하지 않는 항목명을 반환함."""
        ignored: list[str] = []
        if params.temperature is not None and not self.temperature:
            ignored.append("temperature")
        if params.top_p is not None and not self.top_p:
            ignored.append("top_p")
        if params.top_k is not None and not self.top_k:
            ignored.append("top_k")
        if params.reasoning_effort and self.thinking_control != "effort":
            ignored.append("reasoning_effort")
        return ignored


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

    @classmethod
    def model_capabilities(cls, model: str) -> ModelCapabilities:
        """모델명만으로 판정 가능한 기능 정보를 반환함(네트워크 호출 없음)."""
        return ModelCapabilities()

    def capabilities(self) -> ModelCapabilities:
        """현재 인스턴스 모델의 기능."""
        return self.model_capabilities(self.model)

    def output_token_limit(self) -> int | None:
        """모델의 출력 토큰 한도. 조회할 수 없으면 None을 반환함."""
        return None

    def count_tokens(self, text: str) -> int | None:
        """입력 텍스트의 토큰 수. 지원하지 않으면 None을 반환함."""
        return None

    def output_format_instruction(self) -> str:
        """프로바이더별 출력 형식 요구사항을 시스템 지시문에 덧붙일 문자열.

        Gemini는 최상위 배열 스키마를 쓰지만 OpenAI 구조화 출력은 최상위가
        객체여야 하므로, 형식 차이를 각 프로바이더가 스스로 설명함.
        """
        return ""

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
