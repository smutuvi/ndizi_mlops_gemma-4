# src/utils/constants.py — shared column names and ASR prompt defaults.
from __future__ import annotations

SRC_DATASETS = ["smutuvi/ndizi-1", "smutuvi/ndizi-1-2025"]

# Sunflower (Sunbird) leakage-safe Swahili recipe.
SUNFLOWER_MODEL_ID = "Sunbird/Sunflower-Gemma4-E2B"
SUNFLOWER_SYSTEM_PROMPT = (
    "You are Sunflower, a helpful assistant made by Sunbird AI who knows many African languages."
)
# Train ASR only on Ndizi (plus optional de-duped ALFFA). Never FLEURS / CV / SALT / Waxal.
SUNFLOWER_TRAIN_ASR = ["smutuvi/ndizi-1", "smutuvi/ndizi-1-2025"]
SUNFLOWER_OPTIONAL_ASR = "nickdee96/ALFFA-Swahili-News"
# Eval-only: Ndizi test is in-domain; FLEURS is OOD and must not be trained on.
SUNFLOWER_EVAL_ASR = [
    "smutuvi/ndizi-1:test",
    "smutuvi/ndizi-1-2025:test",
    "google/fleurs:sw_ke:test",
]
AUDIO_COLUMN = "audio"
TEXT_COLUMN = "text"
SPEAKER_COLUMN = "speaker_id"
PREPARED_REPO = "smutuvi/ndizi-merged-asr"
WHISPER_REF_ID = "openai/whisper-large-v3"
TARGET_SR = 16_000
MAX_AUDIO_SEC = 30.0

# Must match LiteRT / on-device ASR (conversion_scripts/quick_test.py, .litertlm bundles).
ONDEVICE_ASR_INSTRUCTION = "Andika maneno unayosikia katika sauti hii."

# Short English variant (training only; do not use for LiteRT eval).
SHORT_ASR_INSTRUCTION = (
    "Transcribe the Swahili audio exactly as spoken. "
    "Output only the transcript text, no explanations."
)

ASR_INSTRUCTION = (
    "Transcribe the following speech segment in Swahili into Swahili text.\n\n"
    "Follow these specific instructions for formatting the answer:\n"
    "* Only output the transcription, with no newlines.\n"
    "* When transcribing numbers, write the digits, i.e. write 1.7 and not "
    "one point seven, and write 3 instead of three."
)

# Stricter punctuation variant — use at inference when the model under-punctuates.
PUNCTUATION_ASR_INSTRUCTION = (
    "Transcribe the following speech segment in Swahili into Swahili text.\n\n"
    "Follow these specific instructions for formatting the answer:\n"
    "* Only output the transcription, with no newlines.\n"
    "* Use natural written Swahili capitalization: uppercase at the start of each sentence; "
    "uppercase for proper nouns and spoken labels (e.g. Aina A, Aina B) when the speaker uses them.\n"
    "* Do not write the whole transcript in lowercase; preserve uppercase and lowercase as in normal Swahili writing.\n"
    "* PUNCTUATION IS MANDATORY — a transcription with no punctuation is wrong.\n"
    "* End every declarative sentence with a period (.).\n"
    "* End every question with a question mark (?).\n"
    "* Use commas (,) to separate listed items, after introductory phrases (e.g. 'Kwa mfano,'), "
    "and at natural spoken pauses within a long sentence.\n"
    "* Example of correct punctuation: 'Aina ya kwanza ni A, aina ya pili ni B. Je, unaelewa?'\n"
    "* Example of wrong punctuation: 'Aina ya kwanza ni A aina ya pili ni B Je unaelewa'\n"
    "* Do not repeat the same word or phrase; transcribe each word once.\n"
    "* When transcribing numbers, write the digits, i.e. write 1.7 and not "
    "one point seven, and write 3 instead of three."
)

ASR_PROMPT_MAP = {
    "ondevice": ONDEVICE_ASR_INSTRUCTION,
    "short": SHORT_ASR_INSTRUCTION,
    "full": ASR_INSTRUCTION,
    "punctuation": PUNCTUATION_ASR_INSTRUCTION,
}


def resolve_asr_prompt(style: str | None) -> str:
    key = (style or "ondevice").strip().lower()
    if key not in ASR_PROMPT_MAP:
        raise ValueError(f"Unknown ASR prompt style {style!r}; expected one of {sorted(ASR_PROMPT_MAP)}")
    return ASR_PROMPT_MAP[key]
