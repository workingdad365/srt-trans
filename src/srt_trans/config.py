"""애플리케이션 설정 저장/로드.

설정은 플랫폼별 사용자 설정 디렉터리의 JSON 파일에 보관함.
- Windows: %APPDATA%\\srt-trans\\config.json
- macOS:   ~/Library/Application Support/srt-trans/config.json
- Linux:   ${XDG_CONFIG_HOME:-~/.config}/srt-trans/config.json
"""

from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path
from typing import Any

APP_NAME = "srt-trans"
CONFIG_FILENAME = "config.json"

# API 키처럼 응답에서 마스킹해야 하는 항목
SECRET_KEYS = ("api_keys", "tmdb_api_key")

DEFAULT_CONFIG: dict[str, Any] = {
    # 프로바이더별 API 키 { "gemini": "..." }
    "api_keys": {},
    # 현재 선택된 프로바이더
    "provider": "gemini",
    # 프로바이더별 선택 모델 { "gemini": "gemini-2.5-flash" }
    "models": {},
    "tmdb_api_key": "",
    # 번역 파라미터
    "batch_size": 300,
    "temperature": None,
    "top_p": None,
    "top_k": None,
    "thinking": True,
    # Gemini 계열: 사고 예산(토큰 수)
    "thinking_budget": 2048,
    # OpenAI/OpenRouter 추론 모델 계열: 추론 강도 (모델별 허용값이 다름).
    # 자막 번역은 깊은 추론이 필요하지 않아 가장 낮은 단계를 기본으로 둠.
    # 추론을 끌 수 없는 모델에서는 UI가 자동으로 그다음 낮은 단계를 고름
    "reasoning_effort": "none",
    "streaming": True,
    # 번역 결과에서 종결 마침표를 제거함 (모델이 놓친 경우 대비)
    "strip_trailing_period": True,
    # OpenRouter 라우팅 설정 { route_variant, providers, allow_fallbacks, deny_data_collection }
    "routing": {},
    # 번역 대상 언어 코드 (출력 파일명은 movie.kor.srt 처럼 3글자로 정규화됨)
    "language_code": "ko",
    # 상세 줄거리/등장인물 정보는 작품마다 다르므로 저장하지 않음
    # 마지막으로 사용한 추가 지시문 (작품과 무관한 지시라 유지함)
    "extra_instruction": "",
}


def get_config_dir() -> Path:
    """설정 디렉터리 경로를 반환함.

    SRT_TRANS_CONFIG_DIR 환경변수가 있으면 그 경로를 사용함.
    (테스트가 사용자의 실제 설정 파일을 건드리지 않도록 하기 위한 장치)
    """
    override = os.getenv("SRT_TRANS_CONFIG_DIR")
    if override:
        return Path(override)

    if os.name == "nt":
        base = os.getenv("APPDATA")
        root = Path(base) if base else Path.home() / "AppData" / "Roaming"
    elif sys.platform == "darwin":
        root = Path.home() / "Library" / "Application Support"
    else:
        base = os.getenv("XDG_CONFIG_HOME")
        root = Path(base) if base else Path.home() / ".config"
    return root / APP_NAME


class ConfigManager:
    """설정 파일의 로드/저장을 담당함. 스레드 안전."""

    def __init__(self, config_dir: Path | None = None) -> None:
        self.config_dir = config_dir or get_config_dir()
        self.config_path = self.config_dir / CONFIG_FILENAME
        self._lock = threading.RLock()
        self._data: dict[str, Any] = dict(DEFAULT_CONFIG)
        self.load()

    def load(self) -> dict[str, Any]:
        with self._lock:
            try:
                if self.config_path.exists():
                    loaded = json.loads(self.config_path.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        # 더 이상 쓰지 않는 키(예: 예전 story_context)는 버림
                        known = {k: v for k, v in loaded.items() if k in DEFAULT_CONFIG}
                        self._data = {**DEFAULT_CONFIG, **known}
            except (OSError, json.JSONDecodeError):
                # 설정 파일이 깨져 있으면 기본값으로 동작하되, 원본은 백업해 둠
                self._backup_broken_config()
                self._data = dict(DEFAULT_CONFIG)
            return dict(self._data)

    def _backup_broken_config(self) -> None:
        """읽지 못한 설정 파일을 덮어쓰기 전에 .bak으로 보존함."""
        try:
            if self.config_path.exists():
                backup = self.config_path.with_suffix(".json.bak")
                self.config_path.replace(backup)
        except OSError:
            pass

    def save(self) -> bool:
        with self._lock:
            try:
                self.config_dir.mkdir(parents=True, exist_ok=True)
                tmp = self.config_path.with_suffix(".json.tmp")
                tmp.write_text(
                    json.dumps(self._data, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                tmp.replace(self.config_path)
                return True
            except OSError:
                return False

    def all(self) -> dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._data))

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value

    def update(self, values: dict[str, Any]) -> None:
        """알려진 키만 병합함. api_keys/models는 하위 병합."""
        with self._lock:
            for key, value in values.items():
                if key not in DEFAULT_CONFIG:
                    continue
                if key in ("api_keys", "models") and isinstance(value, dict):
                    current = dict(self._data.get(key) or {})
                    for sub_key, sub_value in value.items():
                        if sub_value is None:
                            current.pop(sub_key, None)
                        else:
                            current[sub_key] = sub_value
                    self._data[key] = current
                else:
                    self._data[key] = value

    def get_api_key(self, provider: str) -> str:
        with self._lock:
            keys = self._data.get("api_keys") or {}
            return str(keys.get(provider) or "")

    def get_model(self, provider: str) -> str:
        with self._lock:
            models = self._data.get("models") or {}
            return str(models.get(provider) or "")

    def masked(self) -> dict[str, Any]:
        """API 키를 마스킹한 사본을 반환함(UI 전송용)."""
        data = self.all()
        keys = data.get("api_keys") or {}
        data["api_keys"] = {k: mask_secret(v) for k, v in keys.items()}
        data["api_keys_set"] = {k: bool(v) for k, v in keys.items()}
        data["tmdb_api_key"] = mask_secret(data.get("tmdb_api_key", ""))
        data["tmdb_api_key_set"] = bool((self.get("tmdb_api_key") or "").strip())
        data["config_path"] = str(self.config_path)
        return data


def mask_secret(value: str | None) -> str:
    """비밀 값을 앞 4자 + 별표로 마스킹함."""
    if not value:
        return ""
    text = str(value)
    if len(text) <= 8:
        return "*" * len(text)
    return f"{text[:4]}{'*' * 8}{text[-4:]}"
