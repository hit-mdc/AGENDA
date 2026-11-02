"""Public interfaces for dataset, representation, and index integrations."""
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
import numpy as np
from .contracts import Approximation, SearchResult


@dataclass
class IndexRequest:
    query: np.ndarray
    representation: np.ndarray
    data: np.ndarray
    top_k: int
    exclude_id: int | None
    budgets_ms: tuple[float, ...]
    encoding_ms: float


class DSAPlugin(Protocol):
    def fit(self, data: np.ndarray, artifact_dir: Path) -> None: ...
    def load(self, artifact_dir: Path) -> None: ...
    def transform(self, rows: np.ndarray) -> np.ndarray: ...
    def approximate(self, query: np.ndarray, cardinality: int) -> Approximation: ...


class IndexPlugin(Protocol):
    def build(self, data: np.ndarray, representations: np.ndarray, artifact_dir: Path) -> None: ...
    def load(self, data: np.ndarray, representations: np.ndarray, artifact_dir: Path) -> None: ...
    def describe(self) -> dict: ...
    def search_curve(self, request: IndexRequest) -> dict[float, SearchResult]: ...
