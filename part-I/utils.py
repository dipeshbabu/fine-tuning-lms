import random
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


# Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


# Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


# --- small helpers / lists ---
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
    "organize": "organise", "apologize": "apologise", "catalog": "catalogue",
    "program": "programme"  # context-dependent but fine for reviews
}
UK_US_MAP = {v: k for k, v in US_UK_MAP.items()}

KEYBOARD_NEIGHBORS = {
    'a': 'qs', 's': 'awed', 'd': 'sfe', 'f': 'drg', 'g': 'fty', 'h': 'gju',
    'j': 'huik', 'k': 'jil', 'l': 'k', 'q': 'wa', 'w': 'qes', 'e': 'wrd',
    'r': 'etf', 't': 'ryg', 'y': 'tuh', 'u': 'yij', 'i': 'uoj', 'o': 'ipk',
    'p': 'o', 'z': 'x', 'x': 'zc', 'c': 'xv', 'v': 'cb', 'b': 'vn', 'n': 'bm', 'm': 'n'
}


def _maybe_download_wordnet():
    try:
        _ = wordnet.synsets("test")
    except LookupError:
        import nltk
        try:
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
        except Exception:
            pass


def custom_transform(example):
    # knobs (you can also pass via CLI and set here if you prefer)
    SYN_P = 0.30   # ↑ from 0.15
    TYPO_P = 0.15   # ↑ from 0.10
    DROP_P = 0.08   # stopword dropout (keep negations)
    LOCALE_P = 0.20  # US<->UK swap prob
    PUNC_P = 0.10  # punctuation jitter

    rng = random.Random(3407)
    _maybe_download_wordnet()

    def preserve_case(orig, rep):
        return rep.capitalize() if orig[:1].isupper() else rep

    def synonym(tok: str) -> str:
        if not tok.isalpha():
            return tok
        low = tok.lower()
        if low in SENTIMENT_POLAR or low in NEGATION_GUARDS:
            return tok
        if rng.random() > SYN_P:
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
        rep = rng.choice(lemmas)
        return preserve_case(tok, rep)

    def typo(tok: str) -> str:
        if len(tok) < 3 or rng.random() > TYPO_P:
            return tok
        choice = rng.random()
        # 60% neighbor substitution, 20% deletion, 20% insertion/dup
        if choice < 0.6:
            i = rng.randint(1, len(tok)-2)
            c = tok[i].lower()
            if c in KEYBOARD_NEIGHBORS:
                repl = rng.choice(list(KEYBOARD_NEIGHBORS[c]))
                return tok[:i] + repl + tok[i+1:]
            return tok
        elif choice < 0.8:
            i = rng.randint(1, len(tok)-2)
            return tok[:i] + tok[i+1:]
        else:
            i = rng.randint(1, len(tok)-2)
            c = tok[i]
            return tok[:i] + c + tok[i:]  # duplicate a char
        # (fallbacks keep token unchanged)

    def locale_swap(tok: str) -> str:
        low = tok.lower()
        if rng.random() > LOCALE_P:
            return tok
        rep = None
        if low in US_UK_MAP:
            rep = US_UK_MAP[low]
        elif low in UK_US_MAP:
            rep = UK_US_MAP[low]
        if rep is None:
            return tok
        return preserve_case(tok, rep)

    # tokenization
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
        # light dropout of harmless stopwords (never drop negations / sentiment)
        if low in STOPWORDS and (low not in NEGATION_GUARDS) and (low not in SENTIMENT_POLAR):
            if rng.random() < DROP_P:
                continue

        t1 = synonym(t)
        t2 = locale_swap(t1)
        t3 = typo(t2)
        new_toks.append(t3)

    # punctuation jitter on the detokenized string
    detok = TreebankWordDetokenizer().detokenize(new_toks)
    if rng.random() < PUNC_P:
        detok = detok.replace("...", " … ").replace(
            "!", ".").replace(".", "…", 1)

    example["text"] = detok
    return example
