from abc import ABC, abstractmethod


class Scorer(ABC):
    """Gives each (query, passage) pair a relevance score. Higher = more relevant."""

    name: str = "scorer"
    batch_size: int = 16  # how many rows score() receives per call

    @abstractmethod
    def score(self, rows):
        """rows: list of candidate dicts. Return a list of floats, same order and length."""