"""Turns written French into text that reads well out loud.

Safety net for the small model, which doesn't always follow the prompt: numbers, hours, prices and
ordinals are spelled out, abbreviations expanded, and parentheses, dashes and slashes replaced by
punctuation that the speech synthesis turns into pauses.
"""

from __future__ import annotations

import re

_UNITS = ["zéro", "un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf", "dix",
          "onze", "douze", "treize", "quatorze", "quinze", "seize"]
_TENS = {2: "vingt", 3: "trente", 4: "quarante", 5: "cinquante", 6: "soixante"}


def _below_100(n: int) -> str:
    if n <= 16:
        return _UNITS[n]
    if n < 20:
        return "dix-" + _UNITS[n - 10]
    tens, unit = divmod(n, 10)
    if tens in (7, 9):  # soixante-dix…, quatre-vingt-dix…
        base = "soixante" if tens == 7 else "quatre-vingt"
        rest = 10 + unit
        sep = " et " if (tens == 7 and unit == 1) else "-"
        return base + sep + _below_100(rest)
    if tens == 8:
        return "quatre-vingts" if unit == 0 else "quatre-vingt-" + _UNITS[unit]
    if unit == 0:
        return _TENS[tens]
    if unit == 1:
        return _TENS[tens] + " et un"
    return _TENS[tens] + "-" + _UNITS[unit]


def _below_1000(n: int) -> str:
    hundreds, rest = divmod(n, 100)
    if hundreds == 0:
        return _below_100(rest)
    head = "cent" if hundreds == 1 else _UNITS[hundreds] + " cent"
    if rest == 0:
        return head + ("s" if hundreds > 1 else "")
    return head + " " + _below_100(rest)


def number_to_words(n: int) -> str:
    if n < 0:
        return "moins " + number_to_words(-n)
    if n < 1000:
        return _below_1000(n)
    parts = []
    for value, name in ((10**9, "milliard"), (10**6, "million")):
        q, n = divmod(n, value)
        if q:
            parts.append(f"{number_to_words(q)} {name}{'s' if q > 1 else ''}")
    q, n = divmod(n, 1000)
    if q:
        parts.append("mille" if q == 1 else _below_1000(q).removesuffix("s") + " mille")
    if n:
        parts.append(_below_1000(n))
    return " ".join(parts)


def ordinal_to_words(n: int, feminine: bool = False) -> str:
    if n == 1:
        return "première" if feminine else "premier"
    word = number_to_words(n)
    if word.endswith("cents") or word.endswith("vingts"):
        word = word[:-1]
    if word.endswith("cinq"):
        return word + "uième"
    if word.endswith("neuf"):
        return word[:-1] + "vième"
    if word.endswith("e"):
        word = word[:-1]
    return word + "ième"


def _hour(h: str, m: str | None) -> str:
    hours = int(h)
    out = "minuit" if hours == 0 else ("une heure" if hours == 1 else f"{number_to_words(hours)} heures")
    if m and int(m):
        out += " " + number_to_words(int(m))
    return out


def _price(units: str, cents: str | None) -> str:
    out = f"{number_to_words(int(units))} euro{'s' if int(units) > 1 else ''}"
    if cents and int(cents):
        out += " " + number_to_words(int(cents.ljust(2, "0")))
    return out


def _decimal(units: str, decimals: str) -> str:
    if int(decimals) == 0:
        return number_to_words(int(units))
    return f"{number_to_words(int(units))} virgule {number_to_words(int(decimals))}"


_ABBREVIATIONS = [
    (r"\bav\.", "avenue"), (r"\bbd\b\.?", "boulevard"), (r"\bbld\b\.?", "boulevard"),
    (r"\bpl\.", "place"), (r"\bSt\b\.?", "Saint"), (r"\bSte\b\.?", "Sainte"),
    (r"\bn°\s*", "numéro "), (r"\bN°\s*", "numéro "), (r"\betc\.", "et cetera"),
    (r"\bM\.(?=\s+[A-Z])", "monsieur"), (r"\bMme\b\.?", "madame"),
    (r"\bmin\b\.?", "minutes"), (r"\bkm\b", "kilomètres"), (r"\bkg\b", "kilos"),
    (r"\bmn\b", "minutes"),
]


def to_oral(text: str) -> str:
    """Rewrites `text` so that the speech synthesis says it naturally. Expects a single paragraph."""
    t = text
    # Punctuation that has no voice: turned into pauses or removed.
    t = re.sub(r"\(\s*\d{5}\s*\)", "", t)                       # (75009)
    t = re.sub(r",?\s*\b75\d{3}\b(?!\s?(?:€|euro))", "", t)       # postal codes elsewhere
    t = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", t)                   # MB1 -> MB 1
    t = re.sub(r"\s*\(([^()]*)\)", r", \1,", t)                  # (aside) -> , aside,
    t = re.sub(r"\s+[-–—]\s+", ", ", t)                         # dash between words
    t = re.sub(r"(?<=[^\W\d])\s*/\s*(?=[^\W\d])", " ou ", t)    # Yakyoshi/Yakiochi
    t = re.sub(r"[«»\"“”]", "", t)
    t = re.sub(r"\s*:\s+", ", ", t)                             # "voici : a, b" -> "voici, a, b"
    t = t.replace("&", " et ").replace("%", " pour cent").replace("+", " plus ")

    # Hours and durations: 15h, 15 h 00, 11h30, 15:00.
    t = re.sub(r"\b(\d{1,2})\s?(?:h|H)(?:\s?(\d{2}))?\b", lambda m: _hour(m[1], m[2]), t)
    t = re.sub(r"\b(\d{1,2}):(\d{2})\b", lambda m: _hour(m[1], m[2]), t)
    # Prices: 10€, 13,50 €, 13.50€, 388 euros 10.
    t = re.sub(r"(\d+)(?:[.,](\d{1,2}))?\s?(?:€|euros?\b)(?:\s?(\d{1,2})\b)?",
               lambda m: _price(m[1], m[2] or m[3]), t)
    # Ordinals: 8e, 8ème, 18e, 1er, 1re, 1ère, XXe is left alone.
    t = re.sub(r"\b1(?:er)\b", "premier", t)
    t = re.sub(r"\b1(?:re|ère)\b", "première", t)
    t = re.sub(r"\b(\d+)\s?(?:e|ème|eme|è)\b", lambda m: ordinal_to_words(int(m[1])), t)
    # Decimals, then whole numbers (thousands separators first: 12 500, 12.500).
    t = re.sub(r"\b(\d{1,3})(?:[   .](\d{3}))+\b",
               lambda m: re.sub(r"\D", "", m[0]), t)
    t = re.sub(r"\b(\d+)[.,](\d+)\b", lambda m: _decimal(m[1], m[2]), t)
    t = re.sub(r"\d+", lambda m: number_to_words(int(m[0])), t)

    for pattern, word in _ABBREVIATIONS:
        t = re.sub(pattern, word, t)

    # Tidy up what the rewrites left behind.
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\s+([,.!?…])", r"\1", t)
    t = re.sub(r",\s*([,.!?…])", r"\1", t)
    t = re.sub(r"([,.!?…])(?=\w)", r"\1 ", t)
    t = t.strip(" ,")
    if t and t[-1] not in ".!?…":
        t += "."
    return t
