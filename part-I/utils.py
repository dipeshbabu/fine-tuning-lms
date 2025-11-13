import os
import random
from typing import List
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)
_RNG = random.Random(3407)

# ----- lexicons -----
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "so", "very", "really", "just", "quite", "rather", "some", "any",
    "many", "much", "more", "most", "also", "still", "even", "though", "while", "however", "indeed", "like"
}
SENTIMENT_POLAR = {
    "great", "excellent", "amazing", "awesome", "good", "fantastic", "favorite",
    "bad", "terrible", "awful", "horrible", "poor", "boring", "worst"
}
NEGATION_GUARDS = {"not", "n't", "never", "no"}

US_UK_MAP = {
    "color": "colour", "colors": "colours", "favorite": "favourite", "honor": "honour",
    "theater": "theatre", "center": "centre", "gray": "grey", "realize": "realise",
    "organize": "organise", "apologize": "apologise", "catalog": "catalogue", "program": "programme"
}
UK_US_MAP = {v: k for k, v in US_UK_MAP.items()}

KEYBOARD_NEIGHBORS = {
    "a": "qs", "s": "awed", "d": "sfe", "f": "drg", "g": "fty", "h": "gju", "j": "huik", "k": "jil", "l": "k",
    "q": "wa", "w": "qes", "e": "wrd", "r": "etf", "t": "ryg", "y": "tuh", "u": "yij", "i": "uoj", "o": "ipk", "p": "o",
    "z": "x", "x": "zc", "c": "xv", "v": "cb", "b": "vn", "n": "bm", "m": "n"
}

# Unicode look-alikes (Latin → Cyrillic)
CONFUSABLES = {
    "a": "а",  # U+0430
    "e": "е",  # U+0435
    "o": "о",  # U+043E
    "p": "р",  # U+0440
    "c": "с",  # U+0441
    "y": "у",  # U+0443
    "x": "х",  # U+0445
    "A": "А",  # U+0410
    "E": "Е",  # U+0415
    "O": "О",  # U+041E
    "P": "Р",  # U+0420
    "C": "С",  # U+0421
    "Y": "У",  # U+0423
    "X": "Х",  # U+0425
}

# ---- defaults (env-overridable) ----
SYN_P = float(os.getenv("SYN_P",   0.15))
TYPO_P = float(os.getenv("TYPO_P",  0.10))
DROP_P = float(os.getenv("DROP_P",  0.08))
LOCALE_P = float(os.getenv("LOCALE_P", 0.20))
PUNC_P = float(os.getenv("PUNC_P",  0.10))
# case jitter prob per eligible token
CASE_P = float(os.getenv("CASE_P",  0.25))
# confusable swap prob per eligible char
CONF_P = float(os.getenv("CONF_P",  0.20))


def _maybe_download_wordnet_once() -> None:
    try:
        _ = wordnet.synsets("test")
    except LookupError:
        import nltk
        try:
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
        except Exception:
            pass


_maybe_download_wordnet_once()


def example_transform(example: dict) -> dict:
    example["text"] = example.get("text", "").lower()
    return example


def custom_transform(example: dict) -> dict:
    """Label-preserving OOD transform with synonym, typo, locale, stopword drop,
    punctuation jitter + (NEW) case jitter & Unicode confusables."""
    def _preserve_case(orig: str, rep: str) -> str:
        return rep.capitalize() if orig[:1].isupper() else rep

    def _synonym(tok: str) -> str:
        if not tok.isalpha():
            return tok
        low = tok.lower()
        if low in SENTIMENT_POLAR or low in NEGATION_GUARDS:
            return tok
        if _RNG.random() > SYN_P:
            return tok
        syns = wordnet.synsets(low)
        lemmas: List[str] = []
        for s in syns:
            for l in s.lemmas():
                cand = l.name().replace("_", " ")
                if cand.isalpha() and cand.lower() != low:
                    lemmas.append(cand)
        if not lemmas:
            return tok
        return _preserve_case(tok, _RNG.choice(lemmas))

    def _typo(tok: str) -> str:
        if len(tok) < 3 or _RNG.random() > TYPO_P:
            return tok
        r = _RNG.random()
        if r < 0.6:
            i = _RNG.randint(1, len(tok)-2)
            c = tok[i].lower()
            if c in KEYBOARD_NEIGHBORS:
                repl = _RNG.choice(list(KEYBOARD_NEIGHBORS[c]))
                return tok[:i] + repl + tok[i+1:]
            return tok
        elif r < 0.8:
            i = _RNG.randint(1, len(tok)-2)
            return tok[:i] + tok[i+1:]
        else:
            i = _RNG.randint(1, len(tok)-2)
            return tok[:i] + tok[i] + tok[i:]

    def _locale_swap(tok: str) -> str:
        low = tok.lower()
        if _RNG.random() > LOCALE_P:
            return tok
        rep = US_UK_MAP.get(low) or UK_US_MAP.get(low)
        return _preserve_case(tok, rep) if rep else tok

    def _case_jitter(tok: str) -> str:
        # Skip polar/negation and short tokens
        if tok.lower() in SENTIMENT_POLAR or tok.lower() in NEGATION_GUARDS or len(tok) < 3:
            return tok
        if _RNG.random() > CASE_P:
            return tok
        chars = list(tok)
        for i in range(1, len(chars)-1):
            if chars[i].isalpha() and _RNG.random() < 0.35:
                chars[i] = chars[i].lower(
                ) if chars[i].isupper() else chars[i].upper()
        return "".join(chars)

    def _confusables(tok: str) -> str:
        # Swap a few letters with Unicode look-alikes (skip polar/negation words)
        if tok.lower() in SENTIMENT_POLAR or tok.lower() in NEGATION_GUARDS:
            return tok
        # Apply per-character with low probability
        out = []
        for ch in tok:
            if ch in CONFUSABLES and _RNG.random() < CONF_P:
                out.append(CONFUSABLES[ch])
            else:
                out.append(ch)
        return "".join(out)

    text = example.get("text", "")
    try:
        toks = word_tokenize(text)
    except LookupError:
        import nltk
        try:
            nltk.download("punkt", quiet=True)
        except Exception:
            pass
        toks = text.split()

    new_toks: List[str] = []
    for t in toks:
        low = t.lower()
        if low in STOPWORDS and low not in NEGATION_GUARDS and low not in SENTIMENT_POLAR:
            if _RNG.random() < DROP_P:
                continue
        t1 = _synonym(t)
        t2 = _locale_swap(t1)
        t3 = _typo(t2)
        t4 = _case_jitter(t3)
        t5 = _confusables(t4)
        new_toks.append(t5)

    detok = TreebankWordDetokenizer().detokenize(new_toks) if new_toks else ""
    if _RNG.random() < PUNC_P:
        detok = detok.replace("...", " … ").replace(
            "!", ".").replace(".", "…", 1)
    example["text"] = detok
    return example
