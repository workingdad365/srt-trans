"""srt-trans 데스크톱 앱 진입점."""

from __future__ import annotations

import argparse
import sys

from . import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="srt-trans",
        description="SRT 자막을 한국어로 번역하는 데스크톱 앱을 실행합니다.",
    )
    parser.add_argument("file", nargs="?", help="앱에서 바로 열 SRT 파일 경로")
    parser.add_argument("--version", action="version", version=f"srt-trans {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # 도움말과 버전 조회에는 Qt를 불러오지 않음
    from .gui import main as gui_main

    return gui_main(args.file)


if __name__ == "__main__":
    sys.exit(main())
