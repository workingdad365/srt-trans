"""Anthropic Messages API 기반 Claude 프로바이더 구현."""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any, ClassVar

import anthropic

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

_MODEL_RE = re.compile(r"^claude-(?:sonnet|opus|haiku|fable|mythos)-(\d+)(?:-(\d{1,2})(?:-|$))?")
_MAX_TOKENS = 32768
_DEFAULT_TIMEOUT = 600.0
_RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "translations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"index": {"type": "string"}, "content": {"type": "string"}},
                "required": ["index", "content"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["translations"],
    "additionalProperties": False,
}


def _version(model: str) -> tuple[int, int]:
    match = _MODEL_RE.match(model)
    if match:
        return int(match[1]), int(match[2] or 0)
    if model.startswith("claude-3-7-"):
        return 3, 7
    return 0, 0


def _adaptive(model: str) -> bool:
    return _version(model) >= (4, 6) or model == "claude-mythos-preview"


def _always_thinking(model: str) -> bool:
    return model.startswith(("claude-fable-", "claude-mythos-")) or (
        model.startswith("claude-opus-") and _version(model) >= (5, 5)
    )


def _translate_error(exc: Exception) -> ProviderError:
    """SDK 예외를 번역 엔진의 공통 예외로 변환함."""
    if isinstance(exc, anthropic.AuthenticationError | anthropic.PermissionDeniedError):
        return AuthError(str(exc))
    if isinstance(exc, anthropic.RateLimitError):
        return QuotaExceededError(str(exc))
    return ProviderError(str(exc))


def _check_response(reason: str | None, has_text: bool) -> None:
    if reason == "refusal":
        raise ContentBlockedError("Anthropic 안전 정책에 의해 응답이 차단되었습니다.")
    if reason in ("max_tokens", "model_context_window_exceeded"):
        raise ProviderError("응답이 토큰 한도에 걸려 잘렸습니다. 배치 크기를 줄이고 다시 시도하세요.")
    if reason != "end_turn":
        raise ProviderError(f"Anthropic 응답이 정상적으로 완료되지 않았습니다: {reason}")
    if not has_text:
        raise ProviderError("모델이 빈 응답을 반환했습니다.")


class AnthropicProvider(LLMProvider):
    """Claude 모델 목록, 사고 제어, 구조화 출력 및 스트리밍 지원."""

    info: ClassVar[ProviderInfo] = ProviderInfo(
        id="anthropic",
        label="Anthropic",
        api_key_url="https://platform.claude.com/settings/keys",
        supports_thinking=True,
        supports_streaming=True,
        default_model="claude-sonnet-4-6",
    )

    def __init__(self, api_key: str, model: str, params: GenerationParams | None = None) -> None:
        super().__init__(api_key, model, params)
        self._client: anthropic.Anthropic | None = None

    @property
    def client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic(
                api_key=self.api_key,
                timeout=self.params.timeout or _DEFAULT_TIMEOUT,
                max_retries=2,
            )
        return self._client

    @classmethod
    def list_models(cls, api_key: str) -> list[str]:
        if not (api_key or "").strip():
            raise AuthError("Anthropic API 키가 필요합니다.")
        try:
            with anthropic.Anthropic(api_key=api_key.strip()) as client:
                return sorted({model.id for model in client.models.list() if model.id.startswith("claude-")})
        except Exception as exc:  # noqa: BLE001
            raise _translate_error(exc) from exc

    @classmethod
    def model_capabilities(cls, model: str) -> ModelCapabilities:
        adaptive = _adaptive(model)
        thinking = adaptive or _version(model) >= (3, 7)
        choices: list[str] = []
        notes: list[str] = []
        if adaptive:
            choices = [] if _always_thinking(model) else ["none"]
            choices.extend(["low", "medium", "high"])
            if _version(model) >= (4, 7):
                choices.append("xhigh")
            choices.append("max")
            notes.append("적응형 사고는 추론 강도로 제어하며 샘플링 설정은 전송하지 않습니다.")
            if _always_thinking(model):
                notes.append("이 모델은 사고를 끌 수 없습니다.")
        elif thinking:
            notes.append("사고 예산은 최소 1,024토큰입니다. 사고 사용 중에는 샘플링 설정을 전송하지 않습니다.")
        if not adaptive:
            notes.append("temperature 범위는 0~1이며 top_p와 함께 지정하면 temperature를 우선합니다.")
        return ModelCapabilities(
            thinking=thinking,
            thinking_control="effort" if adaptive else "budget" if thinking else None,
            effort_choices=choices,
            temperature=not adaptive,
            top_p=not adaptive,
            top_k=not adaptive,
            streaming=True,
            token_counting=True,
            notes=notes,
        )

    def output_format_instruction(self) -> str:
        return (
            '\nReturn a single JSON object with exactly one key "translations", whose value is '
            "the array of translated objects described above. Do not wrap it in anything else.\n"
        )

    def _request_kwargs(self) -> dict[str, Any]:
        params = self.params
        caps = self.capabilities()
        kwargs: dict[str, Any] = {"max_tokens": _MAX_TOKENS if caps.thinking else 4096}
        output_config: dict[str, Any] = {}
        if _version(self.model) >= (4, 5) or self.model == "claude-mythos-preview":
            output_config["format"] = {"type": "json_schema", "schema": _RESULT_SCHEMA}

        if caps.thinking_control == "effort":
            effort = (params.reasoning_effort or "low").strip().lower()
            if not params.thinking or effort in ("none", "minimal"):
                effort = "none" if "none" in caps.effort_choices else "low"
            if effort not in caps.effort_choices:
                effort = "low"
            if effort == "none":
                # Sonnet 5.5는 disabled 대신 between_tools를 요구함
                mode = "between_tools" if self.model.startswith("claude-sonnet-5-5") else "disabled"
                kwargs["thinking"] = {"type": mode}
                output_config["effort"] = "low"
            else:
                kwargs["thinking"] = {"type": "adaptive"}
                output_config["effort"] = effort
        elif caps.thinking and params.thinking:
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": max(1024, min(params.thinking_budget, _MAX_TOKENS - 4096)),
            }
        else:
            if params.temperature is not None:
                kwargs["temperature"] = max(0.0, min(params.temperature, 1.0))
            elif params.top_p is not None:
                kwargs["top_p"] = params.top_p
            if params.top_k is not None:
                kwargs["top_k"] = params.top_k
        if output_config:
            kwargs["output_config"] = output_config
        return kwargs

    def output_token_limit(self) -> int:
        kwargs = self._request_kwargs()
        return kwargs["max_tokens"] - kwargs.get("thinking", {}).get("budget_tokens", 0)

    def count_tokens(self, text: str) -> int:
        try:
            result = self.client.messages.count_tokens(
                model=self.model, messages=[{"role": "user", "content": text}]
            )
            return result.input_tokens
        except Exception as exc:  # noqa: BLE001
            raise _translate_error(exc) from exc

    def generate(
        self,
        *,
        system_instruction: str,
        history: list[Turn],
        user_text: str,
    ) -> Iterator[Chunk]:
        # 서명이 없는 사고 요약은 재전송하지 않음. 완료된 턴의 본문만 문맥으로 사용함
        messages = [
            {"role": "assistant" if turn.role == "model" else "user", "content": turn.text}
            for turn in history
            if turn.text
        ]
        messages.append({"role": "user", "content": user_text})
        kwargs = self._request_kwargs()
        try:
            if self.params.streaming:
                yield from self._generate_stream(system_instruction, messages, kwargs)
            else:
                response = self.client.messages.create(
                    model=self.model, system=system_instruction, messages=messages, **kwargs
                )
                has_text = any(block.type == "text" and block.text.strip() for block in response.content)
                _check_response(response.stop_reason, has_text)
                for block in response.content:
                    if block.type == "text":
                        yield Chunk(text=block.text)
                    elif block.type == "thinking":
                        yield Chunk(text=block.thinking, is_thought=True)
        except ProviderError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise _translate_error(exc) from exc

    def _generate_stream(
        self, system: str, messages: list[dict[str, str]], kwargs: dict[str, Any]
    ) -> Iterator[Chunk]:
        reason: str | None = None
        has_text = False
        with self.client.messages.create(
            model=self.model, system=system, messages=messages, stream=True, **kwargs
        ) as stream:
            for event in stream:
                if event.type == "message_delta":
                    reason = event.delta.stop_reason or reason
                elif event.type == "content_block_start":
                    block = event.content_block
                    if block.type == "text" and block.text:
                        has_text = has_text or bool(block.text.strip())
                        yield Chunk(text=block.text)
                    elif block.type == "thinking" and block.thinking:
                        yield Chunk(text=block.thinking, is_thought=True)
                elif event.type == "content_block_delta":
                    delta = event.delta
                    if delta.type == "text_delta":
                        has_text = has_text or bool(delta.text.strip())
                        yield Chunk(text=delta.text)
                    elif delta.type == "thinking_delta":
                        yield Chunk(text=delta.thinking, is_thought=True)
        _check_response(reason, has_text)
