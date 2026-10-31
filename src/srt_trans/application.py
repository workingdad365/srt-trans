"""데스크톱 UI와 독립적인 파일 입력, 번역 실행, 결과 저장."""

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from .jobs import Job
from .prompts import build_system_instruction
from .providers import DEFAULT_PROVIDER, GenerationParams, create_provider, model_capabilities
from .srt_utils import SUBTITLE_EXTENSIONS, build_output_name, read_text_with_fallback
from .tmdb import TMDBItem
from .translator import (
    EngineOptions, TranslationCancelled, TranslationEngine,
    TranslationResult, parse_subtitles,
)

MAX_SOURCE_BYTES = 20 * 1024 * 1024


@dataclass(frozen=True)
class SourceFile:
    path: Path
    text: str
    subtitle_count: int

    @property
    def name(self) -> str:
        return self.path.name


def load_source(raw_path: str | Path) -> SourceFile:
    path = Path(str(raw_path).strip().strip('"')).expanduser().resolve()
    if path.suffix.lower() not in SUBTITLE_EXTENSIONS:
        raise ValueError("SRT 파일만 열 수 있습니다.")
    if not path.is_file():
        raise ValueError(f"파일을 찾을 수 없습니다: {path}")
    if path.stat().st_size > MAX_SOURCE_BYTES:
        raise ValueError("파일이 너무 큽니다(최대 20MB).")
    text = read_text_with_fallback(path)
    return SourceFile(path, text, len(parse_subtitles(text)))


class RoutingOptions(BaseModel):
    """OpenRouter 라우팅 옵션."""

    route_variant: str = ""
    providers: list[str] = Field(default_factory=list)
    allow_fallbacks: bool = True
    deny_data_collection: bool = False

    def to_extra(self) -> dict[str, Any]:
        return self.model_dump()


class TranslateRequest(BaseModel):
    provider: str = DEFAULT_PROVIDER
    model: str = Field(min_length=1)
    title: str = ""
    is_series: bool = False
    story_context: str = ""
    tmdb_context: str = ""
    extra_instruction: str = ""
    batch_size: int = Field(default=300, ge=1, le=2000)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=0)
    thinking: bool = True
    thinking_budget: int = Field(default=2048, ge=0, le=24576)
    reasoning_effort: str | None = None
    streaming: bool = True
    timeout: float = Field(default=600.0, ge=10.0, le=3600.0)
    language_code: str = "ko"
    start_index: int = Field(default=0, ge=0)
    strip_trailing_period: bool = True
    routing: RoutingOptions = Field(default_factory=RoutingOptions)

    @field_validator("language_code")
    @classmethod
    def validate_language_code(cls, value: str) -> str:
        value = value.strip() or "ko"
        if not re.fullmatch(r"[a-zA-Z]{2,8}", value):
            raise ValueError("출력 언어 코드는 2~8자의 영문자로 입력하세요.")
        return value.lower()

    def resolve_context(self) -> tuple[str, str]:
        if manual := self.story_context.strip():
            return manual, "manual"
        if tmdb := self.tmdb_context.strip():
            return tmdb, "tmdb"
        return "", "none"

    def settings(self) -> dict[str, Any]:
        """작품별 정보와 이어하기 위치를 제외한 재사용 설정."""
        values = self.model_dump(exclude={
            "model", "title", "is_series", "story_context", "tmdb_context", "start_index",
        })
        values["models"] = {self.provider: self.model}
        return values


def default_output_path(source: SourceFile, language_code: str = "ko") -> Path:
    target = source.path.with_name(build_output_name(source.name, language_code))
    if target == source.path:
        target = target.with_stem(f"{target.stem}.translated")
    return target


def validate_output_path(target: Path, source: SourceFile) -> Path:
    """자동 저장과 다른 이름으로 저장에서 원본 덮어쓰기를 차단함."""
    target = target.expanduser().resolve()
    if target == source.path.resolve() or (
        target.exists() and source.path.exists() and target.samefile(source.path)
    ):
        raise ValueError("출력 파일이 원본 자막과 같습니다. 다른 이름을 지정하세요.")
    if target.suffix.lower() != ".srt":
        raise ValueError("출력 파일의 확장자는 .srt여야 합니다.")
    if not target.parent.is_dir():
        raise ValueError("출력 폴더가 존재하지 않습니다.")
    if target.is_dir():
        raise ValueError("출력 경로에 폴더가 지정되어 있습니다.")
    return target


def save_result(text: str, target: Path, source: SourceFile) -> Path:
    """같은 폴더의 임시 파일을 교체해 저장 실패 시 기존 결과를 보존함."""
    target = validate_output_path(target, source)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=target.parent,
            prefix=f".{target.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(text)
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target


def build_tmdb_context(item: TMDBItem) -> str:
    heading = f"{item.title} ({item.year})" if item.year else item.title
    lines = [f"[작품] {heading}"]
    if item.original_title and item.original_title != item.title:
        lines.append(f"[원제] {item.original_title}")
    if item.genres:
        lines.append(f"[장르] {', '.join(item.genres)}")
    if item.overview:
        lines.append(f"\n[줄거리]\n{item.overview}")
    cast = [f"- {member['character']} (배우: {member['name']})" for member in item.cast if member.get("character")]
    if cast:
        lines.append("\n[등장인물]\n" + "\n".join(cast))
    return "\n".join(lines)


def run_translation(
    job: Job, *, source: SourceFile, payload: TranslateRequest, api_key: str,
    output_path: Path | None = None,
) -> None:
    """백그라운드 작업에서 번역하고 완료 또는 중단 결과를 보관함."""
    job.set_status("running")
    job.log("info", f"원본: {source.name}")
    job.log("info", f"프로바이더: {payload.provider} / 모델: {payload.model}")
    engine: TranslationEngine | None = None
    try:
        if job.cancel_event.is_set():
            raise TranslationCancelled()
        if output_path is not None:
            output_path = validate_output_path(output_path, source)
        subtitles = parse_subtitles(source.text)
        if payload.start_index >= len(subtitles):
            raise ValueError("시작 위치가 자막 개수를 벗어났습니다.")
        if not payload.model.strip():
            raise ValueError("모델을 선택하세요.")
        job.set_progress(payload.start_index, len(subtitles))
        params = GenerationParams(
            temperature=payload.temperature, top_p=payload.top_p, top_k=payload.top_k,
            thinking=payload.thinking, thinking_budget=payload.thinking_budget,
            reasoning_effort=payload.reasoning_effort, streaming=payload.streaming,
            timeout=payload.timeout, extra=payload.routing.to_extra(),
        )
        capabilities = model_capabilities(payload.provider, payload.model)
        ignored = capabilities.unsupported(params)
        if ignored:
            job.log("warning", f"모델이 지원하지 않아 제외하는 설정: {', '.join(ignored)}")
            for name in ignored:
                setattr(params, name, None)
        for note in capabilities.notes:
            job.log("info", note)
        provider = create_provider(payload.provider, api_key=api_key, model=payload.model, params=params)
        _log_routing(job, payload, provider)
        context, context_source = payload.resolve_context()
        instruction = build_system_instruction(
            story_context=context, title=payload.title, is_series=payload.is_series,
            extra_instruction=payload.extra_instruction, thinking=payload.thinking,
            thinking_supported=payload.provider == "gemini" and capabilities.thinking,
        )
        if context_source == "manual":
            job.log("info", "직접 입력한 상세 줄거리 및 등장인물 정보를 반영했습니다.")
        elif context_source == "tmdb":
            job.log("info", "TMDB에서 가져온 줄거리/출연진 정보를 사용합니다.")
        else:
            job.log("warning", "줄거리/등장인물 정보 없이 번역합니다. 호칭 일관성이 떨어질 수 있습니다.")
        engine = TranslationEngine(
            provider=provider, system_instruction=instruction,
            options=EngineOptions(
                batch_size=payload.batch_size, start_index=payload.start_index,
                strip_trailing_period=payload.strip_trailing_period,
            ),
            on_progress=job.set_progress, on_log=job.log, cancel_event=job.cancel_event,
        )
        result = engine.translate(subtitles)
        _store_result(job, result, source, output_path)
        job.set_status("completed")
    except TranslationCancelled:
        job.log("warning", "번역이 취소되었습니다.")
        _store_partial(job, engine, source, output_path)
        job.set_status("cancelled")
    except Exception as exc:
        job.log("error", str(exc))
        _store_partial(job, engine, source, output_path)
        job.set_status("failed", str(exc))


def _log_routing(job: Job, payload: TranslateRequest, provider: Any) -> None:
    if payload.provider != "openrouter":
        return
    effective = getattr(provider, "effective_model", None)
    model_id = effective() if callable(effective) else payload.model
    if model_id != payload.model:
        job.log("info", f"라우팅 변형을 적용했습니다: {model_id}")
    if hasattr(provider, "_resolve_effort"):
        from .providers.openrouter import _model_meta

        effort = provider._resolve_effort(_model_meta(payload.model))
        if effort:
            job.log("info", f"추론 강도: {effort}")
    options = provider._routing_options() if hasattr(provider, "_routing_options") else {}
    if order := options.get("order"):
        fallback = "허용" if options.get("allow_fallbacks", True) else "차단"
        job.log("info", f"지정한 제공자 순서: {', '.join(order)} (다른 제공자로 전환 {fallback})")
    if sort := options.get("sort"):
        job.log("info", f"제공자 정렬 기준: {sort}")
    if options.get("data_collection") == "deny":
        job.log("info", "데이터를 수집하지 않는 제공자만 사용합니다.")


def _store_result(job: Job, result: TranslationResult, source: SourceFile, target: Path | None) -> None:
    text = result.compose()
    job.set_result(text)
    if target is not None:
        try:
            saved = save_result(text, target, source)
            job.set_result(text, str(saved))
            job.log("success", f"저장 완료: {saved}")
        except (OSError, ValueError) as exc:
            job.log("error", f"파일 저장 실패: {exc}. '다른 이름으로 저장'으로 결과를 보관하세요.")


def _store_partial(job: Job, engine: TranslationEngine | None, source: SourceFile, target: Path | None) -> None:
    if engine is None:
        return
    done = engine.completed_count
    job.set_progress(done)
    partial = engine.partial_result()
    if partial is None:
        return
    job.log("info", f"{done}번 자막까지 번역된 결과를 보관합니다.")
    _store_result(job, partial, source, target)
    if done < partial.total:
        job.log("info", f"부분 결과 파일을 열고 시작 자막 번호를 {done + 1}로 설정하면 이어서 번역할 수 있습니다.")
