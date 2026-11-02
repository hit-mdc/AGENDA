"""Replace the two providers independently; UI consumes only these contracts."""
from dataclasses import dataclass
import numpy as np


@dataclass
class Approximation:
    coefficients: np.ndarray
    reconstructed: np.ndarray
    symbols: np.ndarray


@dataclass
class SearchResult:
    ids: list[int]
    distances: list[float]
    trace: list[dict]
    metrics: dict
    nodes: list[dict]


