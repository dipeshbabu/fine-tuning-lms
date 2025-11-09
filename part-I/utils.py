import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
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


def custom_transform(example):
    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation

    # --- configuration (tuned to be "reasonable") ---
    SYN_P = 0.10     # 10% chance per eligible token to get a synonym
    TYPO_P = 0.06    # 6% chance per eligible token to get a realistic typo
    rng = random.Random(3407)

    # Light guardrail: avoid directly flipping the most polar words
    SENTIMENT_POLAR = {"great", "excellent", "amazing", "awesome", "good", "fantastic",
                       "bad", "terrible", "awful", "horrible", "poor", "boring"}

    # Keyboard neighbors for realistic typos (inner letters only)
    KEYBOARD_NEIGHBORS = {
        'a': 'qs', 's': 'awed', 'd': 'sfe', 'f': 'drg', 'g': 'fty', 'h': 'gju',
        'j': 'huik', 'k': 'jil', 'l': 'k', 'q': 'wa', 'w': 'qes', 'e': 'wrd',
        'r': 'etf', 't': 'ryg', 'y': 'tuh', 'u': 'yij', 'i': 'uoj', 'o': 'ipk',
        'p': 'o', 'z': 'x', 'x': 'zc', 'c': 'xv', 'v': 'cb', 'b': 'vn', 'n': 'bm', 'm': 'n'
    }

    # Ensure NLTK data is available (fail-safe)
    try:
        _ = wordnet.synsets("test")
    except LookupError:
        import nltk
        try:
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
        except Exception:
            pass  # if offline, synonym step will just skip

    def _maybe_synonym(tok: str) -> str:
        # Skip punctuation / numerics / very short
        if not tok.isalpha():
            return tok
        low = tok.lower()
        if low in SENTIMENT_POLAR:  # keep strong polarity words intact
            return tok
        if rng.random() > SYN_P:
            return tok
        # Try to fetch a clean synonym from WordNet
        syns = wordnet.synsets(low)
        lemmas = []
        for s in syns:
            for l in s.lemmas():
                cand = l.name().replace("_", " ")
                # avoid identical, preserve simple alpha tokens
                if cand.lower() != low and cand.isalpha():
                    lemmas.append(cand)
        if not lemmas:
            return tok
        rep = rng.choice(lemmas)
        # preserve capitalization
        if tok[0].isupper():
            rep = rep.capitalize()
        return rep

    def _maybe_typo(tok: str) -> str:
        if len(tok) < 3 or rng.random() > TYPO_P:
            return tok
        # change an inner character to a neighboring key or swap
        i = rng.randint(1, len(tok) - 2)
        c = tok[i].lower()
        if c in KEYBOARD_NEIGHBORS and rng.random() < 0.7:
            repl = rng.choice(list(KEYBOARD_NEIGHBORS[c]))
            return tok[:i] + repl + tok[i+1:]
        # fallback: swap adjacent
        return tok[:i-1] + tok[i] + tok[i-1] + tok[i+1:]

    # Tokenize words (Treebank rules) and detokenize back
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
        # Apply synonym (label-preserving) then a very light typo noise
        t2 = _maybe_synonym(t)
        t3 = _maybe_typo(t2)
        new_toks.append(t3)

    detok = TreebankWordDetokenizer().detokenize(new_toks)
    example["text"] = detok

    return example
