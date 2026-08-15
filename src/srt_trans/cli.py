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
# Ctrl+C 후 열린 연결을 정리하며 기다리는 최대 시간(초)
GRACEFUL_SHUTDOWN_SECONDS = 3


def _loopback_sockets(port: int) -> list[socket.socket]:
    """IPv4/IPv6 루프백 양쪽에 바인딩한 소켓을 만듦.

    Windows에서 localhost는 ::1(IPv6)로 먼저 해석되는데 IPv4에만 바인딩되어 있으면
    요청마다 연결 타임아웃 후 폴백하느라 매우 느려짐. 양쪽 모두 받아 이를 없앰.
    """
    sockets: list[socket.socket] = []
    for family, address in ((socket.AF_INET, "127.0.0.1"), (socket.AF_INET6, "::1")):
        try:
            sock = socket.socket(family, socket.SOCK_STREAM)
            if family == socket.AF_INET6:
                sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            sock.bind((address, port))
            sock.listen(2048)
            sock.set_inheritable(True)
            sockets.append(sock)
        except OSError:
            # IPv6 미지원 등으로 실패하면 해당 계열만 건너뜀
            continue
    return sockets


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

    # localhost는 IPv6(::1)로 먼저 해석되는데 서버는 IPv4에 바인딩되므로,
    # 요청마다 연결 타임아웃(약 2초) 후 폴백하게 됨. IP를 직접 사용해 이를 피함.
    display_host = "127.0.0.1" if host == "0.0.0.0" else host
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

    # 기본 로컬 실행일 때만 IPv4/IPv6 루프백 양쪽에 바인딩함
    sockets = _loopback_sockets(port) if host == DEFAULT_HOST and not args.reload else []

    try:
        if sockets:
            config = uvicorn.Config(
                "srt_trans.server:app",
                log_level="warning",
                access_log=False,
                # SSE 같은 지속 연결이 열려 있어도 Ctrl+C 후 확실히 종료되도록 함
                timeout_graceful_shutdown=GRACEFUL_SHUTDOWN_SECONDS,
            )
            uvicorn.Server(config).run(sockets=sockets)
        else:
            uvicorn.run(
                "srt_trans.server:app",
                host=host,
                port=port,
                reload=args.reload,
                log_level="warning",
                access_log=False,
                timeout_graceful_shutdown=GRACEFUL_SHUTDOWN_SECONDS,
            )
    except KeyboardInterrupt:
        pass
    # 소켓 정리는 uvicorn이 처리함. 여기서 또 닫으면 종료 중인 accept 콜백이 깨짐
    print("\n[srt-trans] 종료합니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
