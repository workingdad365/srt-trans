# srt-trans

SRT 자막 파일을 한국어 자막으로 번역하는 웹 기반 도구.

`gemini-srt-translator` 와 `Gemini-SRT-translator-GUI` 의 번역 로직/프롬프트를 계승하되,
다중 LLM 프로바이더를 염두에 둔 구조로 재작성함. 현재는 Google Gemini만 구현되어 있음.

## 특징

- 웹 UI (Streamlit 미사용, FastAPI + 순수 HTML/JS)
- `srt-trans` 명령 한 번으로 서버 기동 + 브라우저 자동 실행
- 프로바이더 추상화 계층: 새 프로바이더는 `providers/` 에 클래스 하나만 추가하면 됨
- UI에서 프로바이더 / API 키 / 모델 목록 조회 및 선택
- TMDB API 키 설정 및 작품 검색
- **상세 줄거리 및 등장인물 정보** 입력란 → 번역 프롬프트에 반영되어 호칭·어투·고유명사 표기를 일관되게 유지
  - 비워 두면 TMDB에서 선택한 작품의 줄거리·출연진 정보가 대신 사용됨
- 한국어 특화 번역 지시문 (의역, 존댓말/반말 일관성, 문말 마침표 제거, 자막 서식 규칙)
- 배치 번역 + 스트리밍 진행률 표시 + 취소
- 자막 파일 입력: 드래그앤드롭 업로드 / 서버 로컬 경로 지정
- 결과 저장: 원본과 같은 폴더에 `원본명.kor.srt` 저장 또는 브라우저 다운로드

## 요구 사항

- Python 3.13 (3.13 고정)
- [uv](https://docs.astral.sh/uv/)

## 설치

전역 명령(`srt-trans`)으로 설치하는 방법:

```bash
cd G:\project\python\srt-trans
uv tool install --python 3.13 .
```

설치 후 아무 디렉터리에서나 실행:

```bash
srt-trans
```

`uv tool` 의 실행 파일 경로가 PATH에 없다면 한 번 실행:

```bash
uv tool update-shell
```

코드를 수정하며 쓰려면 편집 가능 모드로 설치:

```bash
uv tool install --python 3.13 --editable .
```

### 개발용 실행 (설치 없이)

```bash
cd G:\project\python\srt-trans
uv sync
uv run srt-trans
```

## 사용법

```
srt-trans                      # http://localhost:8420 에서 웹 UI 실행
srt-trans --port 9000          # 포트 지정
srt-trans --no-browser         # 브라우저 자동 실행 안 함
srt-trans --host 0.0.0.0       # 외부 접속 허용 (주의: API 키가 저장된 서버임)
```

### 진행 순서

1. **API 설정** — 프로바이더 선택 → API 키 입력 후 저장 → `모델 목록 불러오기` → 모델 선택
   TMDB API 키는 작품 검색용(선택 사항)
2. **자막 파일** — SRT 파일을 드래그앤드롭하거나, 로컬 경로를 입력해 불러옴
3. **작품 정보 및 번역 컨텍스트** — 제목 확인 후 **상세 줄거리 및 등장인물 정보**를 입력
4. **고급 설정** (선택) — 배치 크기, temperature, thinking 등
5. **번역 실행** — 진행률과 로그를 실시간으로 확인, 완료 후 저장 경로 확인 또는 다운로드

### 번역 컨텍스트 우선순위

1. **상세 줄거리 및 등장인물 정보**를 직접 입력한 경우 → 그 내용만 사용함 (TMDB 정보는 무시)
2. 입력란이 비어 있고 TMDB에서 작품을 선택한 경우 → TMDB의 줄거리·장르·주요 등장인물(배역명) 정보를 사용함
3. 둘 다 없는 경우 → 작품 제목만 전달하고 경고를 남김

### 상세 줄거리 및 등장인물 정보

이 입력란의 내용은 번역 시스템 프롬프트에 포함되어 다음 용도로 사용됨.

- 생략된 주어·대명사 해석
- 인물명/지명/고유명사의 한국어 표기 통일
- 인물 관계·나이·지위에 근거한 존댓말/반말 결정 및 일관성 유지
- 인물별 어투와 성격 반영

작성 예:

```
[줄거리]
1999년, 프로그래머 토마스 앤더슨은 자신이 사는 세계가 가상현실임을 알게 된다. ...

[등장인물]
- 네오(토마스 앤더슨): 주인공. 모피어스에게는 존댓말, 트리니티와는 반말.
- 모피어스: 네오의 스승. 40대. 항상 격식 있는 말투.
- 트리니티: 네오와 동년배. 간결하고 단정적인 어투.
```

## 설정 파일 위치

- Windows: `%APPDATA%\srt-trans\config.json`
- macOS: `~/Library/Application Support/srt-trans/config.json`
- Linux: `${XDG_CONFIG_HOME:-~/.config}/srt-trans/config.json`

API 키가 평문으로 저장되므로 파일 권한에 유의할 것.

`SRT_TRANS_CONFIG_DIR` 환경변수를 지정하면 해당 경로를 설정 디렉터리로 사용함.
테스트나 실험 시에는 이 변수를 지정해 실제 설정 파일을 건드리지 않도록 할 것.

### 저장 시점

| 항목 | 저장 시점 |
| --- | --- |
| API 키, TMDB API 키 | `저장` 버튼을 누른 즉시 |
| 프로바이더, 모델명 | 모델 선택 변경 시 / `모델 목록 불러오기` 직후 |
| 배치 크기, temperature, thinking, 줄거리, 추가 지시문 등 | `번역 시작`을 누를 때 |

설정 파일을 읽을 수 없으면 기본값으로 동작하며, 덮어쓰기 전에 원본을 `config.json.bak` 으로 보존함.

## 프로바이더 추가 방법

1. `src/srt_trans/providers/` 에 `base.LLMProvider` 를 상속한 클래스를 작성함
   - `info` (ProviderInfo), `list_models()`, `generate()` 는 필수
   - `capabilities()`, `count_tokens()` 는 선택
2. `providers/__init__.py` 의 `_PROVIDERS` 에 등록함

UI의 프로바이더 목록과 모델 조회는 자동으로 반영됨.

## 프로젝트 구조

```
src/srt_trans/
  cli.py            # srt-trans 진입점 (서버 기동 + 브라우저)
  server.py         # FastAPI 라우트 (설정/모델/TMDB/업로드/번역/SSE)
  translator.py     # 프로바이더 독립 배치 번역 엔진
  prompts.py        # 한국어 특화 시스템 프롬프트 + 줄거리/등장인물 반영
  providers/
    base.py         # 프로바이더 추상 인터페이스
    gemini.py       # Gemini 구현
  jobs.py           # 작업 상태 및 SSE 이벤트
  config.py         # 설정 저장/로드
  tmdb.py           # TMDB 클라이언트
  srt_utils.py      # 파일명 파싱, 언어 코드 처리, 인코딩 폴백
  static/           # 웹 UI (HTML/CSS/JS)
```

## 원본 대비 변경 사항

- 오디오 추출 및 오디오 기반 성별 추론 기능 제외
- 영상 파일에서 자막 추출(FFmpeg) 기능 제외
- Overview 입력란을 "상세 줄거리 및 등장인물 정보"로 대체하고, 번역 프롬프트에서 이를
  적극 참조하도록 지시문 추가
- 대상 언어는 한국어로 고정
- 무료 할당량 관련 처리(pro 모델 요청 간 30초 지연, 보조 API 키 전환) 제거 — 유료 사용 전제.
  분당 요청 한도(429)에 걸렸을 때의 대기·재시도만 오류 처리로 유지함

## 라이선스

MIT
