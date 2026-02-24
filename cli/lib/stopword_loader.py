from .search_utils import STOPWORDS_PATH


def load_stopwords(path: str = STOPWORDS_PATH) -> list[str]:
    with open(path) as file:
        return file.read().splitlines()
