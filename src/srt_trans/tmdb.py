"""TMDB API 클라이언트.

작품 제목/연도/장르 등 번역 컨텍스트 보조 정보를 조회함.
API Key(v3)와 Bearer Token(v4) 두 방식을 자동으로 구분해 사용함.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx

BASE_URL = "https://api.themoviedb.org/3"
IMAGE_BASE = "https://image.tmdb.org/t/p/w300"
TIMEOUT = 10.0


class TMDBError(Exception):
    """TMDB 호출 실패."""


@dataclass
class TMDBItem:
    """검색/상세 결과 공통 표현."""

    id: int
    title: str
    original_title: str = ""
    year: str = ""
    overview: str = ""
    poster_url: str = ""
    vote_average: float = 0.0
    genres: list[str] = field(default_factory=list)
    is_series: bool = False
    cast: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "original_title": self.original_title,
            "year": self.year,
            "overview": self.overview,
            "poster_url": self.poster_url,
            "vote_average": self.vote_average,
            "genres": self.genres,
            "is_series": self.is_series,
            "cast": self.cast,
        }


def _is_bearer_token(api_key: str) -> bool:
    """JWT 형식이면 v4 Bearer 토큰으로 판단함."""
    return api_key.count(".") == 2 and len(api_key) > 100


def _year_of(date_string: str | None) -> str:
    if not date_string:
        return ""
    return date_string.split("-")[0]


class TMDBClient:
    """TMDB REST API 래퍼."""

    def __init__(self, api_key: str, language: str = "ko-KR") -> None:
        api_key = (api_key or "").strip()
        if not api_key:
            raise TMDBError("TMDB API 키가 설정되지 않았습니다.")
        self.api_key = api_key
        self.language = language
        self._bearer = _is_bearer_token(api_key)

    def _request(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        params = dict(params or {})
        headers: dict[str, str] = {}
        if self._bearer:
            headers["Authorization"] = f"Bearer {self.api_key}"
        else:
            params["api_key"] = self.api_key

        try:
            response = httpx.get(
                f"{BASE_URL}{path}", params=params, headers=headers, timeout=TIMEOUT
            )
        except httpx.RequestError as exc:
            raise TMDBError(f"TMDB 연결 실패: {exc}") from exc

        if response.status_code == 401:
            raise TMDBError("TMDB API 키가 유효하지 않습니다.")
        if response.status_code == 404:
            raise TMDBError("TMDB에서 해당 항목을 찾을 수 없습니다.")
        if response.status_code >= 400:
            raise TMDBError(f"TMDB 오류 (HTTP {response.status_code})")

        try:
            return response.json()
        except ValueError as exc:
            raise TMDBError("TMDB 응답을 해석할 수 없습니다.") from exc

    def validate(self) -> bool:
        """API 키 유효성을 확인함."""
        self._request("/configuration")
        return True

    def search(
        self, query: str, *, is_series: bool = False, year: str | None = None, limit: int = 8
    ) -> list[TMDBItem]:
        """제목으로 영화/시리즈를 검색함."""
        query = (query or "").strip()
        if not query:
            raise TMDBError("검색할 제목을 입력하세요.")

        params: dict[str, Any] = {
            "query": query,
            "language": self.language,
            "include_adult": "false",
        }
        if year and str(year).isdigit():
            params["first_air_date_year" if is_series else "year"] = int(year)

        path = "/search/tv" if is_series else "/search/movie"
        data = self._request(path, params)

        items: list[TMDBItem] = []
        for raw in (data.get("results") or [])[:limit]:
            items.append(self._to_item(raw, is_series=is_series))
        return items

    def details(self, tmdb_id: int, *, is_series: bool = False) -> TMDBItem:
        """TMDB ID로 상세 정보와 주요 출연진을 조회함."""
        path = f"/tv/{tmdb_id}" if is_series else f"/movie/{tmdb_id}"
        raw = self._request(path, {"language": self.language})
        item = self._to_item(raw, is_series=is_series)
        item.genres = [genre.get("name", "") for genre in (raw.get("genres") or [])]

        try:
            credits = self._request(f"{path}/credits", {"language": self.language})
            for member in (credits.get("cast") or [])[:15]:
                item.cast.append(
                    {
                        "name": member.get("name", ""),
                        "character": member.get("character", ""),
                    }
                )
        except TMDBError:
            # 출연진 조회 실패는 치명적이지 않음
            pass

        return item

    @staticmethod
    def _to_item(raw: dict[str, Any], *, is_series: bool) -> TMDBItem:
        if is_series:
            title = raw.get("name") or "제목 없음"
            original = raw.get("original_name") or ""
            date = raw.get("first_air_date") or ""
        else:
            title = raw.get("title") or "제목 없음"
            original = raw.get("original_title") or ""
            date = raw.get("release_date") or ""

        poster = raw.get("poster_path") or ""
        return TMDBItem(
            id=int(raw.get("id") or 0),
            title=title,
            original_title=original,
            year=_year_of(date),
            overview=raw.get("overview") or "",
            poster_url=f"{IMAGE_BASE}{poster}" if poster else "",
            vote_average=float(raw.get("vote_average") or 0.0),
            is_series=is_series,
        )
