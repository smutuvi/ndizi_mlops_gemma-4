# src/utils/constants.py — shared column names and ASR prompt defaults.
from __future__ import annotations

SRC_DATASETS = ["smutuvi/ndizi-1", "smutuvi/ndizi-1-2025"]
AUDIO_COLUMN = "audio"
TEXT_COLUMN = "text"
SPEAKER_COLUMN = "speaker_id"
PREPARED_REPO = "smutuvi/ndizi-merged-asr"
WHISPER_REF_ID = "openai/whisper-large-v3"
TARGET_SR = 16_000
MAX_AUDIO_SEC = 30.0

ASR_INSTRUCTION = (
    "Transcribe the following speech segment in Swahili into Swahili text.\n\n"
    "Follow these specific instructions for formatting the answer:\n"
    "* Only output the transcription, with no newlines.\n"
    "* When transcribing numbers, write the digits, i.e. write 1.7 and not "
    "one point seven, and write 3 instead of three."
)
