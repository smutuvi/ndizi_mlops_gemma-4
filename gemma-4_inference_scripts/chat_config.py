"""Shared Swahili chat settings for Ndizi farming-assistant eval and notebooks."""
from __future__ import annotations

import re
from typing import Callable

# Farmer asks → bot gives practical advice (not survey / interviewer mode).
NDIZI_ADVISOR_SYSTEM_PROMPT = (
    "Wewe ni mshauri wa kilimo unaemsaidia mkulima kwa Kiswahili.\n"
    "Toa ushauri wa vitendo: hatua za kufuata, dalili za kuangalia, au suluhisho linalowezekana.\n"
    "Usianze kwa salamu pekee — elekea moja kwa moja kwenye maudhui ya swali.\n"
    "Tumia sentensi 2–5 za kawaida (sentensi 1–2 tu kwa shukrani).\n"
    "\n"
    "Mfano:\n"
    "Mkulima: Majani ya ndizi yameanza kukauka. Nifanye nini?\n"
    "Mshauri: Angalia kwanza unyevu wa udongo — ikiwa ni ukame, umwagilia asubuhi au jioni. "
    "Kata majani yaliyokauka ili kuzuia ugonjwa kuenea. Angalia pia kama kuna wadudu chini ya majani."
)

NDIZI_SURVEY_SYSTEM_PROMPT = (
    "Wewe ni msaidizi wa kukusanya taarifa za kilimo kutoka kwa wakulima. "
    "Unaongea Kiswahili. Unauliza maswali kuhusu shamba, mazao, na hali ya kilimo."
)

MAX_NEW_TOKENS = 200

CHAT_GEN_KWARGS = {
    "do_sample": True,
    "temperature": 0.65,
    "top_p": 0.9,
    "top_k": 40,
    "repetition_penalty": 1.2,
    "no_repeat_ngram_size": 3,
}

CHAT_RETRY_GEN_KWARGS = {
    "do_sample": True,
    "temperature": 0.85,
    "top_p": 0.95,
    "top_k": 60,
    "repetition_penalty": 1.3,
    "no_repeat_ngram_size": 4,
}

CHAT_FINAL_RETRY_GEN_KWARGS = {
    "do_sample": True,
    "temperature": 0.55,
    "top_p": 0.88,
    "top_k": 30,
    "repetition_penalty": 1.15,
    "no_repeat_ngram_size": 2,
}

_SWAHILI_TOKENS = {
    "na", "ya", "wa", "kwa", "ni", "la", "za", "katika", "kuwa", "au", "kama", "sana",
    "ndiyo", "hapana", "asante", "tafadhali", "habari", "vizuri", "sawa", "bado", "sasa",
    "kila", "ikiwa", "labda", "pia", "kisha", "zaidi", "kidogo", "mara", "hii", "hilo",
    "yake", "wako", "wetu", "wake", "yako", "hapa", "hapo", "leo", "kesho", "jana",
    "fanya", "weka", "panda", "kata", "angalia", "hakikisha", "jaribu", "weka", "tekeleza",
    "elekea", "kwanza", "pili", "tatu", "baada", "kabla", "mpaka", "wakati", "wiki", "mwezi",
    "msimu", "shamba", "mavuno", "mbolea", "mvua", "umwagiliaji", "magugu", "mmea", "ndizi",
    "majani", "mkulima", "ekari", "kukauka", "wadudu", "ugonjwa", "hatua", "dalili",
    "kupanda", "udongo", "ukame", "mizizi", "mchirizi", "mazao", "mimea", "ardhi", "maji",
    "unaweza", "ninaweza", "itabidi", "inawezekana", "unahitaji", "inahitaji", "boresha",
    "linahitaji", "yanaweza", "nitajaribu", "nimeelewa", "kusaidia", "ushauri", "swali",
}

_FARMING_KEYWORDS = {
    "shamba", "mavuno", "mbolea", "mvua", "umwagiliaji", "magugu", "mmea", "ndizi",
    "panda", "dawa", "udongo", "majani", "mkulima", "ekari", "kukauka", "wadudu",
    "ugonjwa", "hatua", "dalili", "kupanda", "kukata", "ukame", "mizizi", "mchirizi",
    "kukua", "kuvuna", "mazao", "mimea", "ardhi", "maji", "mahindi", "nyanya", "viazi",
    "mbegu", "kunyunyiza", "kukauka", "madoa", "kukauka", "umwagilia", "magonjwa",
}

_ACTION_WORDS = {
    "angalia", "weka", "panda", "kata", "fanya", "hakikisha", "jaribu", "tekeleza",
    "umwagilia", "ondoa", "tumia", "punguza", "ongeza", "linda", "tegemeza", "choma",
}

_THANKS_CATEGORIES = frozenset({"thanks", "acknowledgment"})
_RELAXED_CATEGORIES = frozenset({"thanks", "acknowledgment", "greeting"})

_FOLLOWUP_CONTEXT = (
    "[Muktadha: Mkulima na mshauri wamekuwa wakizungumza kuhusu shamba, mbolea, na wadudu. "
    "Jibu swali la ufuatiliaji kama ungekuwa na taarifa hizo tayari.]"
)

_GENERIC_GREETING_RE = re.compile(
    r"^(?:habari(?: yako| za asubuhi| za mchana)?[!?.,\s]*)*"
    r"(?:(?:ninafurahi|naweza|napenda)\s+(?:ku)?kusaidia|karibu sana|asante kwa swali lako)"
    r"[!?.,\s]*$",
    re.IGNORECASE,
)


def augment_user_prompt(prompt: str, category: str = "unknown") -> str:
    cat = (category or "unknown").lower()
    if cat == "followup":
        return f"{_FOLLOWUP_CONTEXT}\n{prompt}"
    if cat == "greeting" and "?" in prompt and len(prompt.split()) < 15:
        return f"{prompt}\n(Nina shida ya kilimo na nahitaji ushauri wa vitendo.)"
    return prompt


def category_system_suffix(category: str = "unknown") -> str:
    cat = (category or "unknown").lower()
    if cat in _THANKS_CATEGORIES:
        return "\nMkulima anakushukuru — jibu kwa ukarimu na sentensi 1–2 (bila kuuliza maswali mengi)."
    if cat == "followup":
        return "\nJibu swali la ufuatiliaji kwa kutumia muktadha uliotolewa."
    return ""


def is_generic_greeting_response(text: str, category: str = "unknown") -> bool:
    cat = (category or "unknown").lower()
    if cat in _THANKS_CATEGORIES:
        return False
    t = text.strip()
    if not t:
        return True
    if _GENERIC_GREETING_RE.match(t):
        return True
    lower = t.lower()
    words = re.findall(r"[a-zA-ZÀ-ÿ]+", lower)
    has_farming = any(k in lower for k in _FARMING_KEYWORDS)
    has_action = any(w in lower for w in _ACTION_WORDS)
    if has_farming or has_action:
        return False
    if len(words) <= 12 and "habari" in lower:
        return True
    if len(words) <= 8 and any(p in lower for p in ("ninafurahi", "naweza kukusaidia", "karibu", "msaidizi")):
        return True
    if cat in _RELAXED_CATEGORIES and len(words) >= 6 and any(w in lower for w in ("saidia", "swali", "shamba", "kilimo", "ndizi")):
        return False
    return False


def has_substantive_content(text: str, category: str = "unknown") -> bool:
    cat = (category or "unknown").lower()
    lower = text.lower()
    if cat in _THANKS_CATEGORIES:
        return any(w in lower for w in ("asante", "karibu", "sawa", "heri", "kazi", "nimeelewa", "nitajaribu"))
    if any(k in lower for k in _FARMING_KEYWORDS):
        return True
    if any(w in lower for w in _ACTION_WORDS):
        return True
    if re.search(r"\b[1-4][.)]\s|\b(kwanza|pili|tatu|hatua)\b", lower):
        return True
    return len(text.strip()) >= 50


def score_chat_response(text: str, category: str = "unknown") -> tuple[bool, list[str]]:
    """Return (passed, failed_check_names)."""
    failed: list[str] = []
    t = text.strip()
    cat = (category or "unknown").lower()
    if not t:
        failed.append("non_empty")
    if re.fullmatch(r"[\d\s.,]+", t):
        failed.append("not_digits_only")
    if any(m in t.lower() for m in ("transcribe", "speech segment", "only output the transcription")):
        failed.append("not_asr_echo")
    tokens = re.findall(r"[a-zA-ZÀ-ÿ]+", t.lower())
    if tokens:
        sw_frac = sum(1 for tok in tokens if tok in _SWAHILI_TOKENS) / len(tokens)
        min_frac = 0.08 if cat in _RELAXED_CATEGORIES else 0.10
        if sw_frac < min_frac:
            failed.append("has_swahili_content")
    else:
        failed.append("has_swahili_content")
    if is_generic_greeting_response(t, category=cat):
        failed.append("not_generic_greeting")
    if not has_substantive_content(t, category=cat):
        failed.append("has_substance")
    return len(failed) == 0, failed


def build_chat_messages(
    user_prompt: str,
    *,
    system: str | None = NDIZI_ADVISOR_SYSTEM_PROMPT,
    category: str = "unknown",
) -> list[dict]:
    sys_text = (system or "") + category_system_suffix(category)
    messages: list[dict] = []
    if sys_text.strip():
        messages.append({"role": "system", "content": [{"type": "text", "text": sys_text.strip()}]})
    messages.append({"role": "user", "content": [{"type": "text", "text": user_prompt}]})
    return messages


def retry_user_suffix(category: str = "unknown") -> str:
    cat = (category or "unknown").lower()
    if cat in _THANKS_CATEGORIES:
        return ""
    return (
        "\n\n(Mkumbushe: toa hatua 2–3 za vitendo zinazohusiana na swali langu. "
        "Anza moja kwa moja na ushauri, si salamu pekee.)"
    )


def final_retry_suffix(category: str = "unknown") -> str:
    if (category or "").lower() in _THANKS_CATEGORIES:
        return ""
    return (
        "\n\n(Jibu kwa Kiswahili. Anza na neno la kwanza kama 'Angalia', 'Weka', 'Panda', au 'Kwanza'. "
        "Toa angalau hatua mbili za vitendo.)"
    )


def pick_best_reply(candidates: list[str], category: str = "unknown") -> str:
    best, best_score = "", -1
    for reply in candidates:
        if not reply or not reply.strip():
            continue
        passed, failed = score_chat_response(reply, category=category)
        score = 100 if passed else max(0, 100 - 15 * len(failed))
        score += min(len(reply.strip()), 120) // 10
        if score > best_score:
            best, best_score = reply, score
    return best or (candidates[0] if candidates else "")


def chat_with_retries(
    prompt: str,
    *,
    category: str = "unknown",
    system: str | None = NDIZI_ADVISOR_SYSTEM_PROMPT,
    generate_fn: Callable[[list[dict], dict], str],
) -> str:
    """Generate chat reply with up to 3 attempts; return highest-scoring response."""
    augmented = augment_user_prompt(prompt, category)
    cat = category or "unknown"

    attempts: list[tuple[list[dict], dict]] = [
        (build_chat_messages(augmented, system=system, category=cat), CHAT_GEN_KWARGS),
    ]
    suffix = retry_user_suffix(cat)
    if suffix:
        attempts.append(
            (build_chat_messages(augmented + suffix, system=system, category=cat), CHAT_RETRY_GEN_KWARGS)
        )
    final_suffix = final_retry_suffix(cat)
    if final_suffix:
        attempts.append(
            (build_chat_messages(augmented + final_suffix, system=system, category=cat), CHAT_FINAL_RETRY_GEN_KWARGS)
        )

    candidates: list[str] = []
    for messages, gen_kw in attempts:
        reply = generate_fn(messages, gen_kw)
        candidates.append(reply)
        passed, _ = score_chat_response(reply, category=cat)
        if passed:
            return reply
    return pick_best_reply(candidates, category=cat)
