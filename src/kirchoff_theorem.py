# Created by Micah
# Date: 11/15/25
# Time: 12:48 PM
# Project: GraphTheory_PlatonicSolids
# File: Notebook.py

import numpy as np

from src.laplacian_matrix import build_laplacian
from adj_lists import init

solids = init()


# Kirchhoff's Matrix-Tree Theorem: any cofactor of the Laplacian equals the number of spanning trees.
# We verify that deleting different rows/cols gives the same determinant.

def spanning_tree_count_kirchhoff(adj: dict[int, list[int]], delete_idx: int = 0) -> int:
    """Return the number of spanning trees using a Laplacian cofactor determinant."""
    L = build_laplacian(adj)
    L_minor = np.delete(np.delete(L, delete_idx, axis=0), delete_idx, axis=1)

    # Prefer exact integer arithmetic when available.
    try:
        import sympy as sp

        return int(sp.Matrix(L_minor).det())
    except Exception:
        # Fallback: floating determinant, rounded to nearest int.
        det = np.linalg.det(L_minor)
        return int(round(det))


for name, adj in solids:
    L = build_laplacian(adj)
    n = L.shape[0]

    # Check a few different cofactors to confirm invariance.
    test_indices = list(range(min(3, n)))  # e.g. [0, 1, 2] when possible
    counts = [spanning_tree_count_kirchhoff(adj, k) for k in test_indices]

    print(f"\n{name}")
    for k, c in zip(test_indices, counts):
        print(f"  delete row/col {k}: {c}")

    if len(set(counts)) != 1:
        raise ValueError(f"Kirchhoff cofactors disagree for {name}: {counts}")

    print(f"  spanning trees (Kirchhoff): {counts[0]}")