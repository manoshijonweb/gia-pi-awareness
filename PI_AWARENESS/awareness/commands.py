"""Exact utterance routing: never turn arbitrary conversation into an action."""
from __future__ import annotations
from dataclasses import dataclass
import unicodedata

@dataclass(frozen=True)
class Command:
    intent: str
    language: str = "en"
    value: float | None = None

ALIASES = {
    "en": {
        "scan": ("scan", "look", "objects", "what is this", "what is in front of me", "what do you see"),
        "color": ("color", "colour", "what color", "what colour", "what color is this", "what colour is this"),
        "read": ("read", "read text", "read this", "what does this say"),
        "distance": ("distance", "how far", "how far is it", "obstacle distance"),
        "help": ("help", "commands"),
        "repeat": ("repeat", "repeat that", "say again"),
        "stop": ("stop", "stop reading", "quiet"),
        "calibrate": ("calibrate", "calibrate steps", "step calibration"),
        "confirm": ("confirm", "yes", "save calibration"),
        "cancel": ("cancel", "no", "cancel calibration"),
        "language_hi": ("hindi", "speak hindi"),
        "language_en": ("english", "speak english"),
        "language_auto": ("automatic language",),
    },
    "hi": {
        "scan": ("स्कैन", "देखो", "यह क्या है", "ये क्या है", "मेरे सामने क्या है", "सामने क्या है"),
        "color": ("रंग", "कौन सा रंग", "क्या रंग है", "रंग बताओ"),
        "read": ("पढ़ो", "पढ़ कर सुनाओ", "पढ़कर सुनाओ", "इसे पढ़ो", "क्या लिखा है"),
        "distance": ("दूरी", "कितनी दूर", "कितनी दूर है", "दूरी बताओ"),
        "help": ("मदद", "सहायता"),
        "repeat": ("दोहराओ", "फिर से बोलो", "फिर सुनाओ"),
        "stop": ("रुको", "बंद करो", "चुप"),
        "calibrate": ("कदम नापो", "कदम की लंबाई बदलो"),
        "confirm": ("हाँ", "हां", "सहेजो", "पुष्टि"),
        "cancel": ("रद्द करो", "नहीं"),
        "language_hi": ("हिंदी", "हिंदी में बोलो"),
        "language_en": ("अंग्रेजी", "अंग्रेज़ी", "अंग्रेजी में बोलो"),
        "language_auto": ("दोनों भाषा",),
    },
}

HI_LENGTHS = {
    20: "बीस", 25: "पच्चीस", 30: "तीस", 35: "पैंतीस", 40: "चालीस",
    45: "पैंतालीस", 50: "पचास", 55: "पचपन", 60: "साठ", 65: "पैंसठ",
    70: "सत्तर", 75: "पचहत्तर", 80: "अस्सी", 85: "पचासी", 90: "नब्बे",
    95: "पचानवे", 100: "सौ", 105: "एक सौ पांच", 110: "एक सौ दस",
    115: "एक सौ पंद्रह", 120: "एक सौ बीस",
}

def english_number(n: int) -> str:
    ones = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
    tens = "zero ten twenty thirty forty fifty sixty seventy eighty ninety".split()
    if n < 20:
        return ones[n]
    if n < 100:
        return tens[n // 10] + (" " + ones[n % 10] if n % 10 else "")
    return "one hundred" + (" " + english_number(n - 100) if n > 100 else "")

def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    # Combining marks MUST be preserved for Devanagari.
    text = "".join(" " if unicodedata.category(ch).startswith("P") else ch for ch in text)
    return " ".join(text.split())

def command_table(language: str) -> dict[str, Command]:
    table = {normalize(phrase): Command(intent, language)
             for intent, aliases in ALIASES[language].items() for phrase in aliases}
    lengths = {n: english_number(n) for n in range(20, 121)} if language == "en" else HI_LENGTHS
    for cm, word in lengths.items():
        prefixes = ("step length",) if language == "en" else ("कदम की लंबाई", "कदम")
        for prefix in prefixes:
            for number in (word, str(cm)):
                table[normalize(f"{prefix} {number}")] = Command("set_step", language, cm * 10.0)
    return table

TABLES = {lang: command_table(lang) for lang in ALIASES}
PREFIXES = {"en": "hello", "hi": "सुनो"}

def parse_command(text: str, language: str | None = None, require_prefix: bool = False) -> Command | None:
    text = normalize(text)
    for lang in ((language,) if language else ("en", "hi")):
        candidate = text
        prefix = PREFIXES[lang] + " "
        if candidate.startswith(prefix):
            candidate = candidate[len(prefix):]
        elif require_prefix:
            continue
        command = TABLES[lang].get(candidate)
        if command:
            return command
    return None

def grammar_phrases(language: str, require_prefix: bool = False) -> list[str]:
    phrases = list(TABLES[language])
    # Words rather than numeric digit alternatives are meaningful to ASR.
    phrases = [p for p in phrases if not any(ch.isdigit() for ch in p)]
    prefixed = [PREFIXES[language] + " " + p for p in phrases]
    return sorted(set(prefixed if require_prefix else phrases + prefixed)) + ["[unk]"]

def select_candidate(candidates: list[tuple[str, str, float]], threshold: float,
                     margin: float, require_prefix: bool = False) -> Command | None:
    ranked = []
    for text, language, confidence in candidates:
        command = parse_command(text, language, require_prefix)
        if command and confidence >= threshold:
            ranked.append((confidence, command))
    ranked.sort(key=lambda item: item[0], reverse=True)
    if not ranked:
        return None
    if len(ranked) > 1:
        best, second = ranked[0], ranked[1]
        if ((best[1].intent, best[1].value) != (second[1].intent, second[1].value)
                and best[0] - second[0] < margin):
            return None  # different language models' scores are NOT calibrated probabilities
    return ranked[0][1]
