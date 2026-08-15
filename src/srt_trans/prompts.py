"""번역 시스템 프롬프트 구성.

참조 구현(gemini-srt-translator/helpers.py, Gemini-SRT-translator-GUI/cli_runner.py)의
한국어 특화 지시문과 자막 서식 규칙을 그대로 계승하고,
'상세 줄거리 및 등장인물 정보'를 번역 근거로 활용하도록 지시문을 추가함.
"""

from __future__ import annotations

TARGET_LANGUAGE = "Korean"

# 프로바이더 공통 응답 스키마(JSON Schema). 각 프로바이더가 자신의 형식으로 변환함.
RESPONSE_SCHEMA: dict = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "index": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["index", "content"],
    },
}

_BASE_INSTRUCTION = (
    "You are an assistant that translates subtitles from any language to Korean.\n"
    "You will receive a list of objects, each with these fields:\n\n"
    "- index: a string identifier\n"
    "- content: the text to translate\n"
    "\nTranslate the 'content' field of each object.\n"
    "If the 'content' field is empty, leave it as is.\n"
    "As this is a film subtitle translation, make sure that the characters' lines sound natural "
    "and the style of the dialogue is consistent.\n"
    "Try to paraphrase naturally, not rigidly, especially in Korean, and omit or replace definite "
    "articles such as he, she, they, etc. that do not need to be translated.\n"
    "Concentrate on turning the whole sentence into a natural Korean sentence rather than "
    "translating it word for word.\n"
    "Korean dialogue requires consistent use of honorifics and semi-speech when speaking between "
    "characters.\n"
    "Try to infer relationships between characters as much as possible to ensure that the use of "
    "honourifics and disrespect is consistent.\n"
    "Remove the period at the end of the sentence.\n"
    "Do NOT move or merge 'content' between objects.\n"
    "Do NOT add or remove any objects.\n"
    "Do NOT alter the 'index' field.\n"
)

# 서식 태그가 붙은 줄에서 종결 마침표가 남는 사례가 잦아 별도 규칙으로 강조함
_PERIOD_RULE = (
    "\n[CRITICAL RULE] Removing the trailing period. Apply this to EVERY entry, with no exception:\n"
    "1. The translated text must NOT end with a period ('.').\n"
    "2. This applies even when the text is wrapped in formatting tags such as <i>, <b>, <u>, "
    "<font ...>, or ASS/SSA overrides like {\\an8}. The period that sits INSIDE the closing tag "
    "must be removed too. The tags themselves must be kept exactly as they are.\n"
    "   - \"<i>He is not coming back.</i>\"  ->  \"<i>그는 돌아오지 않아</i>\"   (NOT \"<i>그는 돌아오지 않아.</i>\")\n"
    "   - \"<i>Call me later.</i>\"          ->  \"<i>나중에 전화해</i>\"\n"
    "   - \"She left already.\"              ->  \"이미 떠났어\"\n"
    "   - \"<i>Line one.\\nLine two.</i>\"     ->  \"<i>첫 줄이야.\\n둘째 줄이야</i>\"\n"
    "3. Lines in italics (a voice over the phone, a radio, narration, song lyrics) are where this "
    "rule is missed most often. Check every line containing a tag one more time before answering.\n"
    "4. Do NOT remove '?', '!', '...' or '…'. Only the plain period is removed.\n"
    "5. If an entry holds several sentences, keep the periods between them and remove only the "
    "very last one.\n"
    "   - \"안녕하세요. 반갑습니다.\"  ->  \"안녕하세요. 반갑습니다\"\n"
    "6. Never touch periods that belong inside an abbreviation or a number (Mr., 3.5, U.S.).\n"
)

_FORMAT_RULES = (
    "\nWhen translating text, follow these formatting rules:\n"
    "1. Line length: Keep lines to 40-50 characters when possible, breaking at natural phrase "
    "boundaries or punctuation marks.\n"
    "2. Dialogue formatting: When text contains dialogue between multiple speakers, format each "
    "speaker's lines separately, starting each with a dash (-).\n"
    "3. Spacing: Ensure proper spacing between words and after punctuation marks.\n"
    "4. Sentence breaks: If a sentence continues on the next line, maintain proper spacing between "
    "the end of one line and the beginning of the next.\n"
    "5. Preserve line breaks, formatting tags and special characters that exist in the source. "
    "The only exception is the sentence-ending period, which must always be removed (see the "
    "critical rule below).\n"
)

_THINKING_ON = "\nThink deeply and reason as much as possible before returning the response.\n"
_THINKING_OFF = "\nDo NOT think or reason.\n"


def _story_context_block(story_context: str, title: str = "", is_series: bool = False) -> str:
    """상세 줄거리 및 등장인물 정보를 번역 지침으로 감싸 반환함."""
    context = (story_context or "").strip()
    title = (title or "").strip()
    if not context and not title:
        return ""

    content_type = "TV series" if is_series else "movie"
    block = "\n[Detailed synopsis and character information]\n"
    if title:
        block += f"The subtitles below belong to a {content_type} titled \"{title}\".\n"
    if context:
        block += (
            "The following is a detailed synopsis and a description of the characters of this "
            f"{content_type}. Use it as the primary reference while translating:\n"
            "- Resolve ambiguous pronouns, subjects and omitted words using this information.\n"
            "- Keep the Korean notation of character names, places and proper nouns exactly as "
            "written here, and keep it consistent across the whole subtitle.\n"
            "- Decide the Korean speech level (honorific vs. casual, 존댓말/반말) for every line "
            "from the relationships, age gap and social status described here, and keep each "
            "character pair's speech level consistent from start to end.\n"
            "- Reflect each character's tone, personality and manner of speaking.\n"
            "- Use the plot information to disambiguate homonyms and context-dependent wording, "
            "but do NOT add any content that is not in the source line.\n\n"
            f"{context}\n"
        )
    return block


def build_system_instruction(
    *,
    story_context: str = "",
    title: str = "",
    is_series: bool = False,
    extra_instruction: str = "",
    thinking: bool = True,
    thinking_supported: bool = True,
) -> str:
    """번역용 시스템 지시문 전체를 생성함.

    thinking_supported는 '프롬프트로 사고를 제어하는 모델인지'를 뜻함.
    OpenAI 추론 모델처럼 파라미터(reasoning_effort)로 제어하는 경우에는 False를 넘겨
    사고 관련 문구가 들어가지 않도록 함.
    """
    instruction = _BASE_INSTRUCTION + _FORMAT_RULES + _PERIOD_RULE
    instruction += _story_context_block(story_context, title=title, is_series=is_series)

    if thinking_supported:
        instruction += _THINKING_ON if thinking else _THINKING_OFF

    extra = (extra_instruction or "").strip()
    if extra:
        instruction += f"\n\nAdditional user instruction:\n\n{extra}\n"

    return instruction
