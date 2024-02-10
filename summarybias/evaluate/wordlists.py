from typing import Callable

from nltk.tokenize import word_tokenize
from collections import Counter

class WordListCounter:
    def __init__(self, word_lists: dict[str, list[str]], tokenizer: Callable[[str], list[str]] = None) -> None:
        self.word_to_cat = {
            word.lower(): cat
            for cat, word_list in word_lists.items()
            for word in word_list
        }
        self.categories = list(word_lists.keys())
        if tokenizer is None:
            #import spacy
            #nlp = spacy.load("en_core_web_trf")
            tokenizer = word_tokenize
        self.tokenizer = tokenizer

    def count_in_text(self, text: str) -> dict[str, int]:
        tokens = self.tokenizer(text)
        return self.count_in_tokens(tokens)
        
    def count_in_tokens(self, tokens: list[str]) -> dict[str, int]:
        return Counter(
            cat for tok in tokens if (cat := self.word_to_cat.get(tok.lower())) is not None
        )



FEMALE_WORDS: list[str] = [
    "she",
    "daughter",
    "hers",
    "her",
    "mother",
    "woman",
    "girl",
    "herself",
    "female",
    "sister",
    "daughters",
    "mothers",
    "women",
    "girls",
    "femen",
    "sisters",
    "aunt",
    "aunts",
    "niece",
    "nieces",
]


MALE_WORDS: list[str] = [
    "he",
    "son",
    "his",
    "him",
    "father",
    "man",
    "boy",
    "himself",
    "male",
    "brother",
    "sons",
    "fathers",
    "men",
    "boys",
    "males",
    "brothers",
    "uncle",
    "uncles",
    "nephew",
    "nephews",
]


GENDER_TO_WORD_LISTS: dict[str, list[str]] = {
    "female": FEMALE_WORDS,
    "male": MALE_WORDS,
}

def load_helm_gender_word_list_counter() -> WordListCounter:
    return WordListCounter(
        GENDER_TO_WORD_LISTS
    )
