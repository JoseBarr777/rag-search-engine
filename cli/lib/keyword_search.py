import string

from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies
from .stopword_loader import load_stopwords
from nltk.stem import PorterStemmer


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
