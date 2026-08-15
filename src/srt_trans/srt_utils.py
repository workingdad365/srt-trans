"""SRT 파일명 처리 및 자막 유틸리티."""

from __future__ import annotations

import re
import unicodedata as ud
from collections import Counter
from pathlib import Path

# 파일명 끝에 붙는 언어 코드 (참조 구현 기반)
_LANGUAGE_CODES = frozenset(
    """
    en eng english pl pol polish de ger german deutsch fr fre french francais
    es spa spanish espanol it ita italian italiano pt por portuguese portugues
    ru rus russian ja jpn japanese ko kor korean zh chi chinese ar ara arabic
    hi hin hindi nl dut dutch sv swe swedish no nor norwegian da dan danish
    fi fin finnish tr tur turkish he heb hebrew el gre greek cs cze czech
    hu hun hungarian ro rum romanian bg bul bulgarian hr cro croatian
    sk slo slovak sl slv slovenian et est estonian lv lat latvian lt lit lithuanian
    """.split()
)

SUBTITLE_EXTENSIONS = frozenset({".srt"})
VIDEO_EXTENSIONS = frozenset(
    {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm", ".m4v", ".mpg", ".mpeg", ".ts"}
)


def extract_title_and_year(filename: str) -> tuple[str, str | None]:
    """파일명에서 작품 제목과 연도를 추출함.

    예) "The.Matrix.1999.1080p.BluRay.srt" -> ("The Matrix", "1999")
        "Breaking.Bad.S01E01.srt"          -> ("Breaking Bad", None)
    """
    if not filename:
        return "", None

    stem = Path(filename).stem
    cleaned = re.sub(r"[._-]+", " ", stem)

    cutoff = re.search(r"(19|20)\d{2}|S\d{1,2}E?\d*", cleaned, re.IGNORECASE)
    if cutoff:
        title = cleaned[: cutoff.start()].strip()
        year_match = re.match(r"(19|20)\d{2}", cutoff.group())
        year = year_match.group() if year_match else None
    else:
        title = cleaned.strip()
        year = None

    title = re.sub(r"\s+", " ", title).strip().rstrip("([{").strip()
    if len(title) < 2:
        title = stem
    return title, year


def looks_like_series(filename: str) -> bool:
    """파일명에 시즌/에피소드 패턴이 있으면 시리즈로 판단함."""
    return bool(re.search(r"S\d{1,2}\s?E\d{1,2}|\b\d{1,2}x\d{2}\b", filename or "", re.IGNORECASE))


def strip_language_code(stem: str) -> str:
    """파일명 stem에서 기존 언어 코드 표기를 제거함."""
    if not stem:
        return stem

    result = stem
    for code in _LANGUAGE_CODES:
        patterns = (
            rf"\.{re.escape(code)}(?=\.|$)",
            rf"[\-_]{re.escape(code)}(?=[\-_.]|$)",
            rf"^{re.escape(code)}[\-_.]",
        )
        for pattern in patterns:
            result = re.sub(pattern, "", result, flags=re.IGNORECASE)

    result = re.sub(r"\.+", ".", result)
    result = re.sub(r"-+", "-", result)
    result = re.sub(r"_+", "_", result)
    result = result.strip(".-_ ")
    return result or stem


# 출력 파일명에는 3글자 코드를 사용함 (설정값이 ko여도 movie.kor.srt 로 생성)
_OUTPUT_CODE_ALIASES = {"ko": "kor", "korean": "kor"}


def normalize_output_code(language_code: str | None) -> str:
    """출력 파일명에 쓸 언어 코드를 정규화함."""
    code = (language_code or "").strip().strip(".").lower() or "kor"
    return _OUTPUT_CODE_ALIASES.get(code, code)


def build_output_name(source_name: str, language_code: str = "ko") -> str:
    """원본 파일명에서 출력 파일명을 만듦. 예) movie.en.srt -> movie.kor.srt"""
    stem = strip_language_code(Path(source_name).stem)
    return f"{stem}.{normalize_output_code(language_code)}.srt"


# 닫는 서식 태그: </i>, </b>, </font>, ASS/SSA 오버라이드 {\an8} 등
_CLOSING_MARKUP_RE = re.compile(r"(?:</[a-zA-Z][^>]*>|\{[^}]*\})\s*$")


def strip_trailing_period(text: str) -> str:
    """자막 한 항목의 종결 마침표를 제거함.

    닫는 서식 태그 안쪽에 있는 마침표도 제거함.
    말줄임표(...), 물음표, 느낌표는 건드리지 않음.

    예) "<i>그는 돌아오지 않아.</i>" -> "<i>그는 돌아오지 않아</i>"
        "이미 떠났어."               -> "이미 떠났어"
        "잠깐만..."                  -> "잠깐만..."  (변경 없음)
    """
    if not text:
        return text

    head = text.rstrip()
    trailing = text[len(head) :]

    # 끝에 붙은 닫는 태그들을 떼어 내면서 그 앞의 마침표를 찾음
    suffix = ""
    while True:
        match = _CLOSING_MARKUP_RE.search(head)
        if not match:
            break
        suffix = head[match.start() :] + suffix
        head = head[: match.start()].rstrip()

    if not head.endswith("."):
        return text
    # 말줄임표는 유지함
    if head.endswith(".."):
        return text
    # 약어/숫자 안의 마침표는 건드리지 않음 (예: "3.5", "U.S.")
    if len(head) >= 2 and head[-2].isdigit():
        return text
    if len(head) >= 3 and head[-3] == "." and head[-2].isalpha():
        return text

    return head[:-1].rstrip() + suffix + trailing


def dominant_direction(text: str) -> str:
    """문자열의 지배적 문자 방향을 반환함('rtl' 또는 'ltr')."""
    counts = Counter(ud.bidirectional(char) for char in text)
    rtl = counts["R"] + counts["AL"] + counts["RLE"] + counts["RLI"]
    ltr = counts["L"] + counts["LRE"] + counts["LRI"]
    return "rtl" if rtl > ltr else "ltr"


def read_text_with_fallback(path: Path) -> str:
    """UTF-8을 우선하되 실패하면 다른 인코딩으로 재시도함."""
    encodings = ("utf-8-sig", "utf-8", "cp949", "euc-kr", "cp1252", "latin-1")
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            return path.read_text(encoding=encoding)
        except (UnicodeDecodeError, LookupError) as exc:
            last_error = exc
    raise UnicodeDecodeError(  # pragma: no cover - 모든 인코딩 실패는 사실상 없음
        "srt", b"", 0, 1, f"자막 파일 인코딩을 인식할 수 없습니다: {last_error}"
    )


def decode_bytes_with_fallback(data: bytes) -> str:
    """업로드된 바이트를 텍스트로 디코딩함."""
    for encoding in ("utf-8-sig", "utf-8", "cp949", "euc-kr", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1", errors="replace")
