import os
import random
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)

# ---- constants (same behavior as your snippet) ----
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "so", "very", "really", "just", "quite", "rather", "some", "any",
    "many", "much", "more", "most", "also", "still", "even", "though", "while", "however", "indeed", "like"
}
SENTIMENT_POLAR = {"great", "excellent", "amazing", "awesome", "good", "fantastic", "favorite",
                   "bad", "terrible", "awful", "horrible", "poor", "boring", "worst"}
NEGATION_GUARDS = {"not", "n't", "never", "no"}

US_UK_MAP = {
    "color": "colour", "colors": "colours", "favorite": "favourite", "honor": "honour",
    "theater": "theatre", "center": "centre", "gray": "grey", "realize": "realise",
    "organize": "organise", "apologize": "apologise", "catalog": "catalogue", "program": "programme"
}
UK_US_MAP = {v: k for k, v in US_UK_MAP.items()}

KEYBOARD_NEIGHBORS = {
    'a': 'qs', 's': 'awed', 'd': 'sfe', 'f': 'drg', 'g': 'fty', 'h': 'gju', 'j': 'huik', 'k': 'jil', 'l': 'k',
    'q': 'wa', 'w': 'qes', 'e': 'wrd', 'r': 'etf', 't': 'ryg', 'y': 'tuh', 'u': 'yij', 'i': 'uoj', 'o': 'ipk', 'p': 'o',
    'z': 'x', 'x': 'zc', 'c': 'xv', 'v': 'cb', 'b': 'vn', 'n': 'bm', 'm': 'n'
}

# --- defaults (exactly your numbers); can be overridden via env *if you ever want* ---
SYN_P_DEFAULT = float(os.getenv("SYN_P", 0.15))
TYPO_P_DEFAULT = float(os.getenv("TYPO_P", 0.10))
DROP_P_DEFAULT = float(os.getenv("DROP_P", 0.08))
LOCALE_P_DEFAULT = float(os.getenv("LOCALE_P", 0.20))
PUNC_P_DEFAULT = float(os.getenv("PUNC_P", 0.10))

# One RNG and one WordNet download, done once
_RNG = random.Random(3407)


def _maybe_download_wordnet_once():
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


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


def custom_transform(example):
    # keep your behavior
    SYN_P = SYN_P_DEFAULT
    TYPO_P = TYPO_P_DEFAULT
    DROP_P = DROP_P_DEFAULT
    LOCALE_P = LOCALE_P_DEFAULT
    PUNC_P = PUNC_P_DEFAULT

    def preserve_case(orig, rep):
        return rep.capitalize() if orig[:1].isupper() else rep

    def synonym(tok: str) -> str:
        if not tok.isalpha():
            return tok
        low = tok.lower()
        if low in SENTIMENT_POLAR or low in NEGATION_GUARDS:
            return tok
        if _RNG.random() > SYN_P:
            return tok
        syns = wordnet.synsets(low)
        lemmas = []
        for s in syns:
            for l in s.lemmas():
                cand = l.name().replace("_", " ")
                if cand.isalpha() and cand.lower() != low:
                    lemmas.append(cand)
        if not lemmas:
            return tok
        rep = _RNG.choice(lemmas)
        return preserve_case(tok, rep)

    def typo(tok: str) -> str:
        if len(tok) < 3 or _RNG.random() > TYPO_P:
            return tok
        choice = _RNG.random()
        if choice < 0.6:
            i = _RNG.randint(1, len(tok)-2)
            c = tok[i].lower()
            if c in KEYBOARD_NEIGHBORS:
                repl = _RNG.choice(list(KEYBOARD_NEIGHBORS[c]))
                return tok[:i] + repl + tok[i+1:]
            return tok
        elif choice < 0.8:
            i = _RNG.randint(1, len(tok)-2)
            return tok[:i] + tok[i+1:]
        else:
            i = _RNG.randint(1, len(tok)-2)
            c = tok[i]
            return tok[:i] + c + tok[i:]

    def locale_swap(tok: str) -> str:
        low = tok.lower()
        if _RNG.random() > LOCALE_P:
            return tok
        rep = US_UK_MAP.get(low) or UK_US_MAP.get(low)
        return preserve_case(tok, rep) if rep else tok

    try:
        toks = word_tokenize(example["text"])
    except LookupError:
        import nltk
        try:
            nltk.download("punkt", quiet=True)
        except Exception:
            pass
        toks = example["text"].split()

    new_toks = []
    for t in toks:
        low = t.lower()
        if low in STOPWORDS and low not in NEGATION_GUARDS and low not in SENTIMENT_POLAR:
            if _RNG.random() < DROP_P:
                continue
        t1 = synonym(t)
        t2 = locale_swap(t1)
        t3 = typo(t2)
        new_toks.append(t3)

    detok = TreebankWordDetokenizer().detokenize(new_toks)
    if _RNG.random() < PUNC_P:
        detok = detok.replace("...", " … ").replace(
            "!", ".").replace(".", "…", 1)
    example["text"] = detok
    return example
