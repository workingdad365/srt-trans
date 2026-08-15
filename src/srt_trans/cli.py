"""srt-trans 명령행 진입점.

`srt-trans` 를 실행하면 로컬 웹 서버를 띄우고 기본 브라우저를 엶.
"""

from __future__ import annotations

import argparse
import socket
import sys
import threading
import time
import webbrowser

import uvicorn

from . import __version__

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8420


def _port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def _find_free_port(host: str, start: int, attempts: int = 20) -> int:
    """사용 가능한 포트를 찾음. 모두 사용 중이면 시작 포트를 그대로 반환함."""
    for offset in range(attempts):
        candidate = start + offset
        if not _port_in_use(host, candidate):
            return candidate
    return start


def _open_browser_when_ready(url: str, host: str, port: int, timeout: float = 15.0) -> None:
    """서버가 응답할 때까지 기다렸다가 브라우저를 엶."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _port_in_use(host, port):
            webbrowser.open(url)
            return
        time.sleep(0.2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="srt-trans",
        description="SRT 자막을 한국어로 번역하는 웹 UI를 실행합니다.",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"바인딩 주소 (기본값: {DEFAULT_HOST})")
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help=f"포트 번호 (기본값: {DEFAULT_PORT})"
    )
    parser.add_argument("--no-browser", action="store_true", help="브라우저를 자동으로 열지 않음")
    parser.add_argument("--reload", action="store_true", help="개발용 자동 리로드")
    parser.add_argument("--version", action="version", version=f"srt-trans {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    host = args.host
    port = args.port
    if _port_in_use(host, port):
        new_port = _find_free_port(host, port + 1)
        print(f"[srt-trans] 포트 {port}이(가) 사용 중이라 {new_port}으로 실행합니다.")
        port = new_port

    display_host = "localhost" if host in ("0.0.0.0", "127.0.0.1") else host
    url = f"http://{display_host}:{port}/"

    print(f"[srt-trans] v{__version__}")
    print(f"[srt-trans] 웹 UI: {url}")
    print("[srt-trans] 종료하려면 Ctrl+C 를 누르세요.")

    if not args.no_browser:
        threading.Thread(
            target=_open_browser_when_ready,
            args=(url, host, port),
            daemon=True,
        ).start()

    try:
        uvicorn.run(
            "srt_trans.server:app",
            host=host,
            port=port,
            reload=args.reload,
            log_level="warning",
            access_log=False,
        )
    except KeyboardInterrupt:
        print("\n[srt-trans] 종료합니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
