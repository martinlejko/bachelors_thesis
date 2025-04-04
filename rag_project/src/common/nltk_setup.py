"""Downloads necessary NLTK data (punkt, stopwords, wordnet) required for text processing."""

import nltk


def main():
    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("wordnet")


if __name__ == "__main__":
    main()
