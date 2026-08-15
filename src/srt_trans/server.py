"""FastAPI 기반 웹 서버."""

from __future__ import annotations

import asyncio
import functools
import json
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__
from .config import ConfigManager
from .jobs import Job, job_manager
from .prompts import build_system_instruction
from .providers import (
    DEFAULT_PROVIDER,
    GenerationParams,
    ProviderError,
    create_provider,
    get_provider_class,
    list_providers,
)
from .srt_utils import (
    SUBTITLE_EXTENSIONS,
    build_output_name,
    decode_bytes_with_fallback,
    extract_title_and_year,
    looks_like_series,
    read_text_with_fallback,
)
from .tmdb import TMDBClient, TMDBError
from .translator import (
    EngineOptions,
    TranslationCancelled,
    TranslationEngine,
    TranslationFailed,
    parse_subtitles,
)

STATIC_DIR = Path(__file__).parent / "static"
# 업로드/로드한 원본을 메모리에 유지하는 시간(초)
SOURCE_TTL_SECONDS = 7200
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


config_manager = ConfigManager()


# --- 원본 파일 저장소 ----------------------------------------------------


@dataclass
class SourceFile:
    """번역 대상 원본 자막."""

    id: str
    name: str
    text: str
    local_path: Path | None = None
    created_at: float = field(default_factory=time.time)

    def info(self) -> dict[str, Any]:
        title, year = extract_title_and_year(self.name)
        try:
            count = len(parse_subtitles(self.text))
        except TranslationFailed:
            count = 0
        return {
            "file_id": self.id,
            "name": self.name,
            "subtitle_count": count,
            "title": title,
            "year": year or "",
            "is_series": looks_like_series(self.name),
            "local_path": str(self.local_path) if self.local_path else None,
            "preview": self.text[:1500],
        }


class SourceStore:
    """원본 자막을 메모리에 보관함."""

    def __init__(self) -> None:
        self._items: dict[str, SourceFile] = {}
        self._lock = threading.RLock()

    def add(self, name: str, text: str, local_path: Path | None = None) -> SourceFile:
        self._cleanup()
        source = SourceFile(id=uuid.uuid4().hex[:12], name=name, text=text, local_path=local_path)
        with self._lock:
            self._items[source.id] = source
        return source

    def get(self, file_id: str) -> SourceFile | None:
        with self._lock:
            return self._items.get(file_id)

    def _cleanup(self) -> None:
        now = time.time()
        with self._lock:
            stale = [
                key
                for key, item in self._items.items()
                if now - item.created_at > SOURCE_TTL_SECONDS
            ]
            for key in stale:
                del self._items[key]


source_store = SourceStore()


# --- 요청 모델 -----------------------------------------------------------


class ConfigUpdate(BaseModel):
    provider: str | None = None
    api_keys: dict[str, str | None] | None = None
    models: dict[str, str] | None = None
    tmdb_api_key: str | None = None
    batch_size: int | None = Field(default=None, ge=1, le=2000)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=0)
    thinking: bool | None = None
    thinking_budget: int | None = Field(default=None, ge=0, le=24576)
    streaming: bool | None = None
    language_code: str | None = None
    story_context: str | None = None
    extra_instruction: str | None = None


class ModelsRequest(BaseModel):
    provider: str = DEFAULT_PROVIDER
    api_key: str | None = None


class LocalFileRequest(BaseModel):
    path: str


class TMDBSearchRequest(BaseModel):
    query: str
    is_series: bool = False
    year: str | None = None
    api_key: str | None = None


class TMDBDetailsRequest(BaseModel):
    tmdb_id: int
    is_series: bool = False
    api_key: str | None = None


class TranslateRequest(BaseModel):
    file_id: str
    provider: str = DEFAULT_PROVIDER
    model: str
    title: str = ""
    is_series: bool = False
    # 사용자가 직접 입력한 상세 줄거리 및 등장인물 정보 (우선 사용)
    story_context: str = ""
    # TMDB에서 가져온 줄거리/출연진 정보 (story_context가 비어 있을 때만 사용)
    tmdb_context: str = ""
    extra_instruction: str = ""
    batch_size: int = Field(default=300, ge=1, le=2000)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=0)
    thinking: bool = True
    thinking_budget: int = Field(default=2048, ge=0, le=24576)
    streaming: bool = True
    language_code: str = "ko"
    start_index: int = Field(default=0, ge=0)
    save_to_source_dir: bool = True

    def resolve_context(self) -> tuple[str, str]:
        """실제 사용할 컨텍스트와 그 출처를 반환함.

        수동 입력이 있으면 그것만 사용하고 TMDB 정보는 무시함.
        """
        manual = (self.story_context or "").strip()
        if manual:
            return manual, "manual"
        tmdb = (self.tmdb_context or "").strip()
        if tmdb:
            return tmdb, "tmdb"
        return "", "none"


# --- 앱 구성 -------------------------------------------------------------


def create_app() -> FastAPI:
    app = FastAPI(title="srt-trans", version=__version__, docs_url=None, redoc_url=None)

    # --- 정적 파일 ---
    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    # --- 메타 ---
    @app.get("/api/providers")
    async def providers() -> dict[str, Any]:
        return {
            "providers": [
                {
                    "id": info.id,
                    "label": info.label,
                    "api_key_url": info.api_key_url,
                    "supports_thinking": info.supports_thinking,
                    "supports_streaming": info.supports_streaming,
                    "default_model": info.default_model,
                }
                for info in list_providers()
            ],
            "default": DEFAULT_PROVIDER,
        }

    # --- 설정 ---
    @app.get("/api/config")
    async def get_config() -> dict[str, Any]:
        return config_manager.masked()

    @app.post("/api/config")
    async def save_config(payload: ConfigUpdate) -> dict[str, Any]:
        values = payload.model_dump(exclude_unset=True)
        config_manager.update(values)
        if not config_manager.save():
            raise HTTPException(status_code=500, detail="설정 파일을 저장하지 못했습니다.")
        return config_manager.masked()

    # --- 모델 목록 ---
    @app.post("/api/models")
    async def models(payload: ModelsRequest) -> dict[str, Any]:
        api_key = (payload.api_key or "").strip() or config_manager.get_api_key(payload.provider)
        if not api_key:
            raise HTTPException(status_code=400, detail="API 키를 먼저 입력하세요.")
        try:
            provider_cls = get_provider_class(payload.provider)
            model_list = await asyncio.to_thread(provider_cls.list_models, api_key)
        except ProviderError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"models": model_list}

    # --- TMDB ---
    def _tmdb_client(api_key: str | None) -> TMDBClient:
        key = (api_key or "").strip() or str(config_manager.get("tmdb_api_key") or "")
        try:
            return TMDBClient(key)
        except TMDBError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/tmdb/search")
    async def tmdb_search(payload: TMDBSearchRequest) -> dict[str, Any]:
        client = _tmdb_client(payload.api_key)
        try:
            items = await asyncio.to_thread(
                functools.partial(
                    client.search, payload.query, is_series=payload.is_series, year=payload.year
                )
            )
        except TMDBError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"results": [item.to_dict() for item in items]}

    @app.post("/api/tmdb/details")
    async def tmdb_details(payload: TMDBDetailsRequest) -> dict[str, Any]:
        client = _tmdb_client(payload.api_key)
        try:
            item = await asyncio.to_thread(
                functools.partial(client.details, payload.tmdb_id, is_series=payload.is_series)
            )
        except TMDBError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return item.to_dict()

    # --- 파일 입력 ---
    @app.post("/api/upload")
    async def upload(file: UploadFile) -> dict[str, Any]:
        filename = file.filename or "subtitle.srt"
        if Path(filename).suffix.lower() not in SUBTITLE_EXTENSIONS:
            raise HTTPException(status_code=400, detail="SRT 파일만 업로드할 수 있습니다.")

        data = await file.read()
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=400, detail="파일이 너무 큽니다(최대 20MB).")
        if not data:
            raise HTTPException(status_code=400, detail="빈 파일입니다.")

        text = decode_bytes_with_fallback(data)
        source = source_store.add(name=Path(filename).name, text=text)
        return _validated_source_info(source)

    @app.post("/api/local-file")
    async def local_file(payload: LocalFileRequest) -> dict[str, Any]:
        raw_path = (payload.path or "").strip().strip('"')
        if not raw_path:
            raise HTTPException(status_code=400, detail="경로를 입력하세요.")

        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists() or not path.is_file():
            raise HTTPException(status_code=400, detail=f"파일을 찾을 수 없습니다: {path}")
        if path.suffix.lower() not in SUBTITLE_EXTENSIONS:
            raise HTTPException(status_code=400, detail="SRT 파일만 지정할 수 있습니다.")
        if path.stat().st_size > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=400, detail="파일이 너무 큽니다(최대 20MB).")

        try:
            text = await asyncio.to_thread(read_text_with_fallback, path)
        except (OSError, UnicodeDecodeError) as exc:
            raise HTTPException(status_code=400, detail=f"파일을 읽지 못했습니다: {exc}") from exc

        source = source_store.add(name=path.name, text=text, local_path=path)
        return _validated_source_info(source)

    # --- 번역 ---
    @app.post("/api/translate")
    async def translate(payload: TranslateRequest) -> dict[str, Any]:
        source = source_store.get(payload.file_id)
        if not source:
            raise HTTPException(status_code=404, detail="원본 파일 정보가 만료되었습니다. 다시 불러오세요.")

        api_key = config_manager.get_api_key(payload.provider)
        if not api_key:
            raise HTTPException(status_code=400, detail="API 키를 먼저 설정에 저장하세요.")
        if not payload.model.strip():
            raise HTTPException(status_code=400, detail="모델을 선택하세요.")

        try:
            subtitles = parse_subtitles(source.text)
        except TranslationFailed as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if payload.start_index >= len(subtitles):
            raise HTTPException(status_code=400, detail="시작 위치가 자막 개수를 벗어났습니다.")

        output_name = build_output_name(source.name, payload.language_code)
        job = job_manager.create(source_name=source.name, output_name=output_name)
        job.set_progress(payload.start_index, len(subtitles))

        runner = functools.partial(_run_translation, source=source, payload=payload)
        job_manager.start(job, runner)
        return {"job_id": job.id, "output_name": output_name, "total": len(subtitles)}

    @app.get("/api/jobs/{job_id}")
    async def job_status(job_id: str) -> dict[str, Any]:
        job = job_manager.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다.")
        return job.snapshot()

    @app.post("/api/jobs/{job_id}/cancel")
    async def job_cancel(job_id: str) -> dict[str, Any]:
        if not job_manager.cancel(job_id):
            raise HTTPException(status_code=400, detail="취소할 수 없는 작업입니다.")
        return {"ok": True}

    @app.get("/api/jobs/{job_id}/events")
    async def job_events(job_id: str, request: Request) -> StreamingResponse:
        job = job_manager.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다.")
        return StreamingResponse(
            _event_stream(job, request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/api/jobs/{job_id}/download")
    async def job_download(job_id: str) -> Response:
        job = job_manager.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다.")
        if job.result_text is None:
            raise HTTPException(status_code=400, detail="아직 결과가 없습니다.")

        content = job.result_text.encode("utf-8")
        # 비ASCII 파일명은 filename*(RFC 5987)로 전달하고, 폴백용 이름은 ASCII로 치환함
        quoted = "".join(char if ord(char) < 128 else "_" for char in job.output_name)
        return Response(
            content=content,
            media_type="application/x-subrip; charset=utf-8",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="{quoted}"; '
                    f"filename*=UTF-8''{_url_quote(job.output_name)}"
                )
            },
        )

    @app.exception_handler(ProviderError)
    async def provider_error_handler(_: Request, exc: ProviderError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    return app


def _validated_source_info(source: SourceFile) -> dict[str, Any]:
    info = source.info()
    if info["subtitle_count"] == 0:
        raise HTTPException(status_code=400, detail="유효한 SRT 자막 항목을 찾지 못했습니다.")
    return info


def _url_quote(text: str) -> str:
    from urllib.parse import quote

    return quote(text)


def _sse(event: dict[str, Any]) -> str:
    return f"event: {event.get('type', 'message')}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"


async def _event_stream(job: Job, request: Request):
    """작업 이벤트를 SSE로 전달함."""
    subscriber = job.subscribe()
    loop = asyncio.get_running_loop()
    try:
        yield _sse({"type": "snapshot", **job.snapshot()})
        while True:
            if await request.is_disconnected():
                break
            try:
                event = await loop.run_in_executor(
                    None, functools.partial(subscriber.get, timeout=1.0)
                )
            except queue.Empty:
                yield ": keepalive\n\n"
                continue

            yield _sse(event)
            if event.get("type") == "end":
                break
    finally:
        job.unsubscribe(subscriber)


def _run_translation(job: Job, *, source: SourceFile, payload: TranslateRequest) -> None:
    """백그라운드 스레드에서 실제 번역을 수행함."""
    job.set_status("running")
    job.log("info", f"원본: {source.name}")
    job.log("info", f"프로바이더: {payload.provider} / 모델: {payload.model}")

    try:
        subtitles = parse_subtitles(source.text)
        job.set_progress(payload.start_index, len(subtitles))

        params = GenerationParams(
            temperature=payload.temperature,
            top_p=payload.top_p,
            top_k=payload.top_k,
            thinking=payload.thinking,
            thinking_budget=payload.thinking_budget,
            streaming=payload.streaming,
        )
        provider = create_provider(
            payload.provider,
            api_key=config_manager.get_api_key(payload.provider),
            model=payload.model,
            params=params,
        )

        capabilities = provider.capabilities()
        context, context_source = payload.resolve_context()
        instruction = build_system_instruction(
            story_context=context,
            title=payload.title,
            is_series=payload.is_series,
            extra_instruction=payload.extra_instruction,
            thinking=payload.thinking,
            thinking_supported=capabilities.thinking,
        )
        if context_source == "manual":
            job.log("info", "직접 입력한 상세 줄거리 및 등장인물 정보를 번역 프롬프트에 반영했습니다.")
        elif context_source == "tmdb":
            job.log("info", "입력한 줄거리가 없어 TMDB에서 가져온 줄거리/출연진 정보를 사용합니다.")
        else:
            job.log("warning", "줄거리/등장인물 정보 없이 번역합니다. 호칭 일관성이 떨어질 수 있습니다.")

        engine = TranslationEngine(
            provider=provider,
            system_instruction=instruction,
            options=EngineOptions(
                batch_size=payload.batch_size,
                start_index=payload.start_index,
            ),
            on_progress=job.set_progress,
            on_log=job.log,
            cancel_event=job.cancel_event,
        )

        result = engine.translate(subtitles)
        output_text = result.compose()

        output_path: str | None = None
        if payload.save_to_source_dir and source.local_path is not None:
            target = source.local_path.parent / job.output_name
            try:
                target.write_text(output_text, encoding="utf-8")
                output_path = str(target)
                job.log("success", f"저장 완료: {target}")
            except OSError as exc:
                job.log("error", f"파일 저장 실패: {exc}. 다운로드 버튼으로 받을 수 있습니다.")

        job.set_result(output_text, output_path)
        job.set_status("completed")

    except TranslationCancelled:
        job.log("warning", "번역이 취소되었습니다.")
        job.set_status("cancelled")
    except (TranslationFailed, ProviderError) as exc:
        job.log("error", str(exc))
        job.set_status("failed", str(exc))
    except Exception as exc:  # noqa: BLE001
        job.log("error", f"예상치 못한 오류: {exc}")
        job.set_status("failed", str(exc))


app = create_app()
