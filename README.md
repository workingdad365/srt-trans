# srt-trans

SRT 자막 파일을 한국어 자막으로 번역하는 웹 기반 도구.

`gemini-srt-translator` 와 `Gemini-SRT-translator-GUI` 의 번역 로직/프롬프트를 계승하되,
다중 LLM 프로바이더를 염두에 둔 구조로 재작성함. 현재 Google Gemini와 OpenAI를 지원함.

## 특징

- 웹 UI (Streamlit 미사용, FastAPI + 순수 HTML/JS)
- `srt-trans` 명령 한 번으로 서버 기동 + 브라우저 자동 실행
- 프로바이더 추상화 계층: 새 프로바이더는 `providers/` 에 클래스 하나만 추가하면 됨
- UI에서 프로바이더 / API 키 / 모델 목록 조회 및 선택
- TMDB API 키 설정 및 작품 검색
- **상세 줄거리 및 등장인물 정보** 입력란 → 번역 프롬프트에 반영되어 호칭·어투·고유명사 표기를 일관되게 유지
  - 비워 두면 TMDB에서 선택한 작품의 줄거리·출연진 정보가 대신 사용됨
- 한국어 특화 번역 지시문 (의역, 존댓말/반말 일관성, 문말 마침표 제거, 자막 서식 규칙)
  - `<i>` 등 서식 태그 안쪽에 남는 마침표까지 제거하며, 모델이 놓친 경우 후처리로 한 번 더 정리함
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

## 문장 끝 마침표 제거

자막 관행에 맞춰 번역 결과의 종결 마침표를 제거함. 두 단계로 처리함.

1. **프롬프트**: 마침표 제거를 별도 항목으로 강조하고, `<i>`, `<b>`, `<font>`, `{\an8}` 같은
   서식 태그 안쪽에 있는 마침표도 제거하도록 예시와 함께 지시함
2. **후처리**: 모델이 놓친 경우를 대비해 결과에서 한 번 더 제거함
   (고급 설정의 `문장 끝 마침표 제거` 체크박스, 기본 켜짐)

후처리가 건드리지 않는 경우:

| 입력 | 출력 | 이유 |
| --- | --- | --- |
| `<i>그는 돌아오지 않아.</i>` | `<i>그는 돌아오지 않아</i>` | 태그 안쪽 마침표 제거 |
| `어디 있어?` / `안 돼!` | 그대로 | 물음표·느낌표는 유지 |
| `잠깐만...` / `그러니까…` | 그대로 | 말줄임표는 유지 |
| `안녕하세요. 반갑습니다.` | `안녕하세요. 반갑습니다` | 마지막 마침표만 제거 |
| `가격은 3.5` / `여긴 U.S.` | 그대로 | 숫자·약어 내부 마침표는 유지 |

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

## 지원 프로바이더

| 프로바이더 | 사고 제어 | temperature / top_p | top_k | 토큰 계산 |
| --- | --- | --- | --- | --- |
| Google Gemini | `thinking` + `thinking_budget`(2.5 이상) | 지원 | 지원 | 지원 |
| OpenAI (GPT-5 이상 / o 시리즈) | `reasoning_effort` | **미지원**(1 고정) | 미지원 | 미지원 |
| OpenAI (GPT-4o 등 이전 모델) | 없음 | 지원 | 미지원 | 미지원 |

모델을 고르면 UI가 해당 모델이 지원하는 입력란만 활성화하고, 지원하지 않는 값이 요청에 섞여 있으면
번역 시작 시 로그로 알린 뒤 전송 대상에서 제외함.

### 프로바이더별로 다른 점 (구현 참고)

- **응답 형식**: Gemini는 최상위 배열 스키마를 받지만, OpenAI 구조화 출력(strict)은 최상위가 객체여야 하고
  모든 필드가 `required`, `additionalProperties: false`여야 함. OpenAI 쪽은 `{"translations": [...]}`로
  감싸고 `output_format_instruction()`으로 그 사실을 프롬프트에 덧붙임. 엔진은 두 형태를 모두 처리함
- **사고 지시문**: "Think deeply..." 문구는 Gemini처럼 프롬프트로 사고를 제어하는 모델에만 들어감.
  OpenAI 추론 모델은 `reasoning_effort` 파라미터가 그 역할을 하므로 프롬프트에 넣지 않음
- **`reasoning_effort` 허용값**: GPT-5는 `minimal/low/medium/high`, GPT-5.1 이상은 `none/low/medium/high`,
  o 시리즈는 `low/medium/high`. 최소 단계 명칭(`minimal`/`none`) 차이는 자동으로 맞춰 줌
- **역할 이름**: OpenAI 추론 모델은 `system` 대신 `developer` 역할을 사용함
- **안전 설정**: `safety_settings`는 Gemini 전용이며 OpenAI에는 전송하지 않음
- **배치 크기 자동 축소**: 토큰 계산 API가 있는 Gemini에서만 동작함. OpenAI는 응답이 길이 한도에 걸리면
  (`finish_reason=length`) 오류로 알리고 배치 크기를 줄이도록 안내함

## 프로바이더 추가 방법

1. `src/srt_trans/providers/` 에 `base.LLMProvider` 를 상속한 클래스를 작성함
   - `info` (ProviderInfo), `list_models()`, `generate()` 는 필수
   - `model_capabilities()`, `output_token_limit()`, `count_tokens()`,
     `output_format_instruction()` 은 선택
2. `providers/__init__.py` 의 `_PROVIDERS` 에 등록함

UI의 프로바이더 목록, 모델 조회, 지원 파라미터 활성화는 자동으로 반영됨.

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
    openai_provider.py  # OpenAI 구현 (GPT-5 이상/이전 모델 파라미터 분기)
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
