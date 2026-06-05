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

# Short variant for asr_safe / asr_moderate training examples.
# Reduces instruction-following contamination of the LM decoder.
SHORT_ASR_INSTRUCTION = (
    "Transcribe the Swahili audio exactly as spoken. "
    "Output only the transcript text, no explanations."
)

ASR_INSTRUCTION = (
    "Transcribe the following speech segment in Swahili into Swahili text.\n\n"
    "Follow these specific instructions for formatting the answer:\n"
    "* Only output the transcription, with no newlines.\n"
    "* Use natural written Swahili capitalization: uppercase at the start of each sentence; "
    "uppercase for proper nouns and spoken labels (e.g. Aina A, Aina B) when the speaker uses them.\n"
    "* Do not write the whole transcript in lowercase; preserve uppercase and lowercase as in normal Swahili writing.\n"
    "* Use standard Swahili punctuation (periods, commas, question marks) that matches the speech.\n"
    "* Do not repeat the same word or phrase; transcribe each word once.\n"
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
