"""
Typed wrappers that distinguish structured from unstructured data views.

SimpleView      - a single 2-D matrix  (samples x features), used for
                  Normal / Bernoulli FA likelihoods.
StructuredView  - a list of 2-D matrices (one per sample), used for
                  CTM likelihoods.
Views           - container that holds simple and structured views
                  separately, replacing the old {"M0": ..., "M1": ...} dict.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Union

import numpy as np


@dataclass
class SimpleView:
    """Unstructured data view (2-D matrix: samples x features)."""

    data: np.ndarray

    def __post_init__(self) -> None:
        if self.data.ndim != 2:
            raise ValueError(
                f"SimpleView expects a 2D numpy array, got shape {self.data.shape}"
            )

    @property
    def N(self) -> int:
        """Number of samples."""
        return self.data.shape[0]

    @property
    def D(self) -> int:
        """Number of features."""
        return self.data.shape[1]


@dataclass
class StructuredView:
    """Structured data view (list of 2-D matrices, one per sample)."""

    data: List[np.ndarray]

    def __post_init__(self) -> None:
        if not self.data:
            raise ValueError("StructuredView expects a non-empty list")
        for i, arr in enumerate(self.data):
            if arr.ndim != 2:
                raise ValueError(
                    f"StructuredView expects 2D arrays, "
                    f"got shape {arr.shape} at index {i}"
                )
        g = self.data[0].shape[1]
        for i, arr in enumerate(self.data[1:], start=1):
            if arr.shape[1] != g:
                raise ValueError(
                    f"Inconsistent feature dimension: expected {g}, "
                    f"got {arr.shape[1]} at index {i}"
                )

    @property
    def N(self) -> int:
        """Number of samples."""
        return len(self.data)

    @property
    def G(self) -> int:
        """Number of features (vocabulary / cell types)."""
        return self.data[0].shape[1]


View = Union[SimpleView, StructuredView]


@dataclass
class Views:
    """
    Container that holds simple and structured views separately.

    Replaces the old ``{"M0": array, "M1": list_of_arrays, ...}`` dict,
    eliminating all string-based M-indexing from the codebase.

    Attributes
    ----------
    simple:     list of SimpleView (Normal / Bernoulli FA likelihoods)
    structured: list of StructuredView (CTM likelihoods)
    cohorts:    optional 1D array-like of cohort labels (length N)
    """

    simple: List[SimpleView] = field(default_factory=list)
    structured: List[StructuredView] = field(default_factory=list)
    cohorts: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not self.simple and not self.structured:
            raise ValueError("Views must contain at least one view.")

        all_n = [v.N for v in self.simple] + [v.N for v in self.structured]
        if len(set(all_n)) > 1:
            raise ValueError(
                f"All views must have the same number of samples N, got: {all_n}"
            )

        if self.cohorts is not None:
            cohort_arr = np.asarray(self.cohorts)
            if cohort_arr.ndim != 1:
                raise ValueError(
                    "Views.cohorts must be a 1D array-like with one label per sample."
                )
            if len(cohort_arr) != self.N:
                raise ValueError(
                    "Views.cohorts length must match number of samples N. "
                    f"Got {len(cohort_arr)} and N={self.N}."
                )
            self.cohorts = cohort_arr

    @property
    def N(self) -> int:
        """Number of samples (consistent across all views)."""
        if self.simple:
            return self.simple[0].N
        return self.structured[0].N

    @property
    def num_simple(self) -> int:
        return len(self.simple)

    @property
    def num_structured(self) -> int:
        return len(self.structured)

    @classmethod
    def from_list(
        cls,
        data: list[Any],
        cohorts: np.ndarray | list[Any] | None = None,
    ) -> "Views":
        """
        Build a :class:`Views` from a list of arrays or lists-of-arrays.

        Each element is classified independently:

        * ``np.ndarray``       → :class:`SimpleView`
        * ``list[np.ndarray]`` → :class:`StructuredView`

        Examples
        --------
        >>> Views.from_list([rna_matrix, atac_matrix, cell_counts_per_sample])
        >>> Views.from_list([rna_matrix], cohorts=sample_cohorts)
        """
        simple = []
        structured = []
        for i, el in enumerate(data):
            if isinstance(el, np.ndarray):
                simple.append(SimpleView(el))
            elif isinstance(el, list):
                structured.append(StructuredView(el))
            else:
                raise TypeError(
                    f"Element at index {i} has unsupported type {type(el)}. "
                    "Expected np.ndarray Simple or list[np.ndarray] Structured."
                )
        cohort_arr = None if cohorts is None else np.asarray(cohorts)
        return cls(simple=simple, structured=structured, cohorts=cohort_arr)
