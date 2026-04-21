import string
import os
import pickle
from collections import defaultdict
from nltk.stem import PorterStemmer
from .search_utils import (
    CACHE_DIR,
    DEFAULT_SEARCH_LIMIT,
    load_movies,
)
from .stopword_loader import load_stopwords

from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies
from .stopword_loader import load_stopwords
from nltk.stem import PorterStemmer

class InvertedIndex:
    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")

    def build(self) -> None:
        movies = load_movies()
        for m in movies:
            doc_id = m["id"]
            doc_description = f"{m['title']} {m['description']}"
            self.docmap[doc_id] = m
            self.__add_document(doc_id, doc_description)

    def save(self) -> None:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)

    def get_documents(self, term: str) -> list[int]:
        doc_ids = self.index.get(term, set())
        return sorted(list(doc_ids))

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize_text(text)
        for token in set(tokens):
            self.index[token].add(doc_id)


def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()
    docs = idx.get_documents("merida")
    print(f"First document for token 'merida' = {docs[0]}")

def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    stop_words = load_stopwords()
    results = []
    for movie in movies:
        query_tokens = filter_stopwords(tokenize_text(query), stop_words)
        title_tokens = filter_stopwords(tokenize_text(movie["title"]), stop_words)
        stemmed_query_tokens = stem_tokens(query_tokens)
        stemmed_title_tokens = stem_tokens(title_tokens)
        if has_matching_token(stemmed_query_tokens, stemmed_title_tokens):
            results.append(movie)
            if len(results) >= limit:
                break

    return results

def stem_tokens(tokens: list[str]):
    stemmer = PorterStemmer()
    seen = set()
    for token in tokens:
        stemmed_token = stemmer.stem(token)
        if stemmed_token not in seen:
            seen.add(stemmed_token)
            
    return list(seen)

def filter_stopwords(tokens: list[str], stop_words: list[str]) -> list[str]:
    return [token for token in tokens if token not in stop_words]


def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def tokenize_text(text: str) -> list[str]:
    text = preprocess_text(text)
    tokens = text.split()
    valid_tokens = []
    for token in tokens:
        if token:
            valid_tokens.append(token)
    return valid_tokens
