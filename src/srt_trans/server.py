"""FastAPI 기반 웹 서버."""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import threading
import time
import uuid
from contextlib import asynccontextmanager
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
    model_capabilities,
)
from .providers.openrouter import ROUTE_VARIANTS, fetch_endpoints
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
    TranslationResult,
    parse_subtitles,
)

STATIC_DIR = Path(__file__).parent / "static"
# 정적 파일 캐시 무효화에 사용할 에셋 목록
_ASSET_FILES = ("style.css", "app.js")

# 서버 종료 시 열려 있는 SSE 스트림을 정리하기 위한 신호
_shutdown = asyncio.Event()


def _quiet_shutdown_noise(loop: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
    """브라우저를 닫거나 서버를 종료할 때 나는 무해한 예외를 감춤.

    - WinError 10054: ProactorEventLoop가 이미 끊긴 소켓을 shutdown 할 때 발생
    - _start_serving 콜백의 AssertionError: 종료 중 남아 있던 accept가 완료되면서
      이미 닫힌 서버에 transport를 붙이려 할 때 발생하는 asyncio 내부 레이스
    동작에는 영향이 없고 콘솔만 지저분해짐.
    """
    exception = context.get("exception")
    if isinstance(exception, ConnectionResetError | ConnectionAbortedError):
        return
    if isinstance(exception, AssertionError) and "_start_serving" in repr(context.get("handle")):
        return
    loop.default_exception_handler(context)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    _shutdown.clear()
    asyncio.get_running_loop().set_exception_handler(_quiet_shutdown_noise)
    try:
        yield
    finally:
        _shutdown.set()
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
    reasoning_effort: str | None = None
    streaming: bool | None = None
    timeout: float | None = Field(default=None, ge=10.0, le=3600.0)
    strip_trailing_period: bool | None = None
    language_code: str | None = None
    extra_instruction: str | None = None
    routing: dict[str, Any] | None = None


class ModelsRequest(BaseModel):
    provider: str = DEFAULT_PROVIDER
    api_key: str | None = None


class ModelInfoRequest(BaseModel):
    provider: str = DEFAULT_PROVIDER
    model: str = ""


class ModelEndpointsRequest(BaseModel):
    provider: str = DEFAULT_PROVIDER
    model: str = ""


class RoutingOptions(BaseModel):
    """OpenRouter 전용 라우팅 설정."""

    # "" | "nitro" | "floor" | "exacto"
    route_variant: str = ""
    # 우선 사용할 제공자 슬러그 목록
    providers: list[str] = Field(default_factory=list)
    # 지정한 제공자가 실패하면 다른 곳으로 넘어갈지 여부
    allow_fallbacks: bool = True
    # 데이터 수집을 하지 않는 제공자만 사용
    deny_data_collection: bool = False

    def to_extra(self) -> dict[str, Any]:
        return {
            "route_variant": self.route_variant,
            "providers": self.providers,
            "allow_fallbacks": self.allow_fallbacks,
            "deny_data_collection": self.deny_data_collection,
        }


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
    # OpenAI 추론 모델용 추론 강도 (Gemini에서는 무시됨)
    reasoning_effort: str | None = None
    streaming: bool = True
    # 요청 하나가 끝나기를 기다리는 최대 시간(초)
    timeout: float = Field(default=600.0, ge=10.0, le=3600.0)
    language_code: str = "ko"
    start_index: int = Field(default=0, ge=0)
    save_to_source_dir: bool = True
    strip_trailing_period: bool = True
    routing: RoutingOptions = Field(default_factory=RoutingOptions)

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


def asset_version() -> str:
    """정적 파일 내용으로 만든 짧은 해시.

    브라우저가 옛 CSS/JS를 캐시해서 변경이 반영되지 않는 문제를 막기 위해
    HTML의 에셋 URL 뒤에 붙임. 내용이 같으면 해시도 같으므로 불필요한
    재다운로드가 생기지 않음.
    """
    digest = hashlib.sha256()
    for name in _ASSET_FILES:
        digest.update(name.encode("utf-8"))
        try:
            digest.update((STATIC_DIR / name).read_bytes())
        except OSError:
            digest.update(b"missing")
    return digest.hexdigest()[:10]


class AssetStaticFiles(StaticFiles):
    """정적 파일 캐시 정책.

    - `?v=<내용해시>` 가 붙은 요청: 내용이 바뀌면 URL이 바뀌므로 오래 캐시해도 안전함
    - 쿼리 없이 직접 접근한 경우: 매번 재검증해 옛 파일이 남지 않게 함
    """

    async def get_response(self, path: str, scope: Any) -> Response:
        response = await super().get_response(path, scope)
        query = scope.get("query_string", b"").decode("latin-1")
        if "v=" in query:
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        else:
            response.headers["Cache-Control"] = "no-cache, must-revalidate"
        return response


def create_app() -> FastAPI:
    app = FastAPI(
        title="srt-trans",
        version=__version__,
        docs_url=None,
        redoc_url=None,
        lifespan=_lifespan,
    )

    # --- 정적 파일 ---
    @app.get("/")
    async def index() -> Response:
        html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        html = html.replace("__ASSET_V__", asset_version())
        return Response(
            content=html,
            media_type="text/html; charset=utf-8",
            headers={"Cache-Control": "no-cache, must-revalidate"},
        )

    app.mount("/static", AssetStaticFiles(directory=STATIC_DIR), name="static")

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

    @app.post("/api/model-info")
    async def model_info(payload: ModelInfoRequest) -> dict[str, Any]:
        """선택한 모델이 지원하는 파라미터를 반환함(UI 입력란 활성화 판단용)."""
        try:
            capabilities = model_capabilities(payload.provider, payload.model)
        except ProviderError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "model": payload.model,
            "thinking": capabilities.thinking,
            "thinking_control": capabilities.thinking_control,
            "effort_choices": capabilities.effort_choices,
            "temperature": capabilities.temperature,
            "top_p": capabilities.top_p,
            "top_k": capabilities.top_k,
            "streaming": capabilities.streaming,
            "token_counting": capabilities.token_counting,
            "notes": capabilities.notes,
            # OpenRouter만 제공자 선택/라우팅 변형을 지원함
            "supports_routing": payload.provider == "openrouter",
            "route_variants": (
                [
                    {"value": key, "label": item["label"]}
                    for key, item in ROUTE_VARIANTS.items()
                ]
                if payload.provider == "openrouter"
                else []
            ),
        }

    @app.post("/api/model-endpoints")
    async def model_endpoints(payload: ModelEndpointsRequest) -> dict[str, Any]:
        """모델을 제공하는 제공자(엔드포인트) 목록을 반환함."""
        if payload.provider != "openrouter":
            return {"endpoints": []}
        if not payload.model.strip():
            raise HTTPException(status_code=400, detail="모델을 먼저 선택하세요.")
        try:
            endpoints = await asyncio.to_thread(fetch_endpoints, payload.model)
        except ProviderError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"endpoints": endpoints}

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
    """작업 이벤트를 SSE로 전달함.

    블로킹 큐를 스레드로 기다리면 서버 종료 시 그 스레드가 남아 프로세스가
    끝나지 않으므로, asyncio 큐로 대기함.
    """
    subscriber = job.subscribe(asyncio.get_running_loop())
    try:
        yield _sse({"type": "snapshot", **job.snapshot()})
        while not _shutdown.is_set():
            if await request.is_disconnected():
                break
            try:
                event = await asyncio.wait_for(subscriber.queue.get(), timeout=1.0)
            except TimeoutError:
                yield ": keepalive\n\n"
                continue

            yield _sse(event)
            if event.get("type") == "end":
                break
    except asyncio.CancelledError:
        # 서버 종료 등으로 스트림이 취소된 경우 조용히 빠져나감
        raise
    finally:
        job.unsubscribe(subscriber)


def _run_translation(job: Job, *, source: SourceFile, payload: TranslateRequest) -> None:
    """백그라운드 스레드에서 실제 번역을 수행함."""
    job.set_status("running")
    job.log("info", f"원본: {source.name}")
    job.log("info", f"프로바이더: {payload.provider} / 모델: {payload.model}")

    engine: TranslationEngine | None = None
    try:
        subtitles = parse_subtitles(source.text)
        job.set_progress(payload.start_index, len(subtitles))

        params = GenerationParams(
            temperature=payload.temperature,
            top_p=payload.top_p,
            top_k=payload.top_k,
            thinking=payload.thinking,
            thinking_budget=payload.thinking_budget,
            reasoning_effort=payload.reasoning_effort,
            streaming=payload.streaming,
            timeout=payload.timeout,
            extra=payload.routing.to_extra(),
        )

        # 모델이 지원하지 않는 파라미터는 전송 전에 제거하고 사용자에게 알림
        capabilities = model_capabilities(payload.provider, payload.model)
        ignored = capabilities.unsupported(params)
        if ignored:
            job.log(
                "warning",
                f"{payload.model} 모델이 지원하지 않아 다음 설정을 제외합니다: {', '.join(ignored)}",
            )
            for name in ignored:
                setattr(params, name, None)
        for note in capabilities.notes:
            job.log("info", note)

        provider = create_provider(
            payload.provider,
            api_key=config_manager.get_api_key(payload.provider),
            model=payload.model,
            params=params,
        )
        _log_routing(job, payload, provider)
        context, context_source = payload.resolve_context()
        # 사고 지시문은 프롬프트로 사고를 제어하는 방식(Gemini)에서만 넣음.
        # OpenAI 추론 모델은 reasoning_effort 파라미터가 그 역할을 하므로 제외함.
        prompt_controls_thinking = capabilities.thinking_control in ("budget", "on_off")
        instruction = build_system_instruction(
            story_context=context,
            title=payload.title,
            is_series=payload.is_series,
            extra_instruction=payload.extra_instruction,
            thinking=payload.thinking,
            thinking_supported=prompt_controls_thinking,
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
                strip_trailing_period=payload.strip_trailing_period,
            ),
            on_progress=job.set_progress,
            on_log=job.log,
            cancel_event=job.cancel_event,
        )

        result = engine.translate(subtitles)
        _store_result(job, result, source, payload)
        job.set_status("completed")

    except TranslationCancelled:
        job.log("warning", "번역이 취소되었습니다.")
        _store_partial(job, engine, source, payload)
        job.set_status("cancelled")
    except (TranslationFailed, ProviderError) as exc:
        job.log("error", str(exc))
        _store_partial(job, engine, source, payload)
        job.set_status("failed", str(exc))
    except Exception as exc:  # noqa: BLE001
        job.log("error", f"예상치 못한 오류: {exc}")
        _store_partial(job, engine, source, payload)
        job.set_status("failed", str(exc))


def _log_routing(job: Job, payload: TranslateRequest, provider: Any) -> None:
    """OpenRouter 라우팅 설정을 실제 요청 기준으로 알림."""
    if payload.provider != "openrouter":
        return

    effective = getattr(provider, "effective_model", None)
    model_id = effective() if callable(effective) else payload.model
    if model_id != payload.model:
        job.log("info", f"라우팅 변형을 적용했습니다: {model_id}")

    # 실제로 전송되는 추론 강도를 알림(요청값과 다를 수 있음)
    if hasattr(provider, "_resolve_effort"):
        from .providers.openrouter import _model_meta

        effort = provider._resolve_effort(_model_meta(payload.model))
        if effort == "none":
            job.log("info", "추론을 끄고 요청합니다 (reasoning effort: none).")
        elif effort:
            job.log("info", f"추론 강도: {effort}")
        elif payload.reasoning_effort:
            job.log(
                "warning",
                f"이 모델은 추론을 끌 수 없어 '{payload.reasoning_effort}' 설정을 적용하지 못했습니다.",
            )

    options = provider._routing_options() if hasattr(provider, "_routing_options") else {}
    if order := options.get("order"):
        fallback = "허용" if options.get("allow_fallbacks", True) else "차단"
        job.log("info", f"지정한 제공자 순서: {', '.join(order)} (다른 제공자로 전환 {fallback})")
    if sort := options.get("sort"):
        job.log("info", f"제공자 정렬 기준: {sort}")
    if options.get("data_collection") == "deny":
        job.log("info", "데이터를 수집하지 않는 제공자만 사용합니다.")


def _store_result(
    job: Job, result: TranslationResult, source: SourceFile, payload: TranslateRequest
) -> None:
    """번역 결과를 저장하고 다운로드 가능한 상태로 만듦."""
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


def _store_partial(
    job: Job, engine: TranslationEngine | None, source: SourceFile, payload: TranslateRequest
) -> None:
    """중단된 경우에도 여기까지 번역된 내용을 저장해 이어서 할 수 있게 함."""
    if engine is None:
        return
    partial = engine.partial_result()
    if partial is None:
        return

    done = engine.completed_count
    job.log("info", f"{done}번 자막까지 번역된 결과를 저장합니다.")
    job.set_progress(done)
    _store_result(job, partial, source, payload)
    if done < partial.total:
        job.log(
            "info",
            f"이어서 하려면 고급 설정의 시작 자막 번호를 {done + 1}로 두고 다시 실행하세요.",
        )


app = create_app()
