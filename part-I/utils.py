import os
import random
from typing import List

from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

# Global seeds / RNG
random.seed(0)
_RNG = random.Random(3407)


# Lexicons / maps
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

# Defaults (your original numbers). You can override via env vars.
# Example: SYN_P=0.2 python main.py ...
SYN_P_DEFAULT = float(os.getenv("SYN_P",   0.15))
TYPO_P_DEFAULT = float(os.getenv("TYPO_P",  0.10))
DROP_P_DEFAULT = float(os.getenv("DROP_P",  0.08))
LOCALE_P_DEFAULT = float(os.getenv("LOCALE_P", 0.20))
PUNC_P_DEFAULT = float(os.getenv("PUNC_P",  0.10))

# One-time WordNet availability check


def _maybe_download_wordnet_once() -> None:
    try:
        _ = wordnet.synsets("test")
    except LookupError:
        import nltk
        try:
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
        except Exception:
            # If offline, synonym step will silently no-op.
            pass


_maybe_download_wordnet_once()

# Public helpers


def example_transform(example: dict) -> dict:
    """
    Baseline example transform: lowercase the text.
    """
    text = example.get("text", "")
    example["text"] = text.lower()
    return example


def custom_transform(example: dict) -> dict:
    """
    Label-preserving OOD transform combining:
      - synonym replacement via WordNet (guarding sentiment & negations)
      - US<->UK spelling swaps
      - realistic typos (neighbor subst / deletion / duplication)
      - light stopword dropout
      - punctuation jitter
    Strengths are controlled by *_DEFAULT constants (or env vars) and
    are intentionally modest to avoid flipping sentiment labels.
    """
    # knobs
    SYN_P = SYN_P_DEFAULT
    TYPO_P = TYPO_P_DEFAULT
    DROP_P = DROP_P_DEFAULT
    LOCALE_P = LOCALE_P_DEFAULT
    PUNC_P = PUNC_P_DEFAULT

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
        rep = _RNG.choice(lemmas)
        return _preserve_case(tok, rep)

    def _typo(tok: str) -> str:
        if len(tok) < 3 or _RNG.random() > TYPO_P:
            return tok
        choice = _RNG.random()
        # 60% neighbor substitution, 20% deletion, 20% duplication
        if choice < 0.6:
            i = _RNG.randint(1, len(tok) - 2)
            c = tok[i].lower()
            if c in KEYBOARD_NEIGHBORS:
                repl = _RNG.choice(list(KEYBOARD_NEIGHBORS[c]))
                return tok[:i] + repl + tok[i + 1:]
            return tok
        elif choice < 0.8:
            i = _RNG.randint(1, len(tok) - 2)
            return tok[:i] + tok[i + 1:]
        else:
            i = _RNG.randint(1, len(tok) - 2)
            c = tok[i]
            return tok[:i] + c + tok[i:]

    def _locale_swap(tok: str) -> str:
        low = tok.lower()
        if _RNG.random() > LOCALE_P:
            return tok
        rep = US_UK_MAP.get(low) or UK_US_MAP.get(low)
        return _preserve_case(tok, rep) if rep else tok

    # Tokenize (fallback to .split if punkt missing)
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

    # Apply transforms token-wise
    new_toks: List[str] = []
    for t in toks:
        low = t.lower()
        # dropout only harmless function words (not negations / sentiment)
        if low in STOPWORDS and low not in NEGATION_GUARDS and low not in SENTIMENT_POLAR:
            if _RNG.random() < DROP_P:
                continue
        t1 = _synonym(t)
        t2 = _locale_swap(t1)
        t3 = _typo(t2)
        new_toks.append(t3)

    # Detokenize + punctuation jitter
    if new_toks:
        detok = TreebankWordDetokenizer().detokenize(new_toks)
    else:
        detok = ""  # if everything dropped (rare), keep empty string
    if _RNG.random() < PUNC_P:
        detok = detok.replace("...", " … ").replace(
            "!", ".").replace(".", "…", 1)

    example["text"] = detok
    return example
