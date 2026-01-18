# Created by Micah
# Date: 11/15/25
# Time: 2:39 PM
# Project: GraphTheory_PlatonicSolids
# File: Build.py
import numpy as np
import itertools

# Takes in an adjacency list, returns the laplacian matrix

def build_laplacian(adj_list):
    n = len(adj_list)
    L = np.zeros((n, n), dtype=int)
    for i in range(n):
        neighbors = adj_list[i]
        L[i, i] = len(neighbors)
        for j in neighbors:
            L[i, j] = -1
    return L


def build_adjacency_matrix(adj_list, symmetrize: bool = True) -> np.ndarray:
    """Build an (n x n) adjacency matrix A from an adjacency list.

    If `symmetrize` is True, we treat the graph as undirected and set A[i,j]=A[j,i]=1
    whenever either i lists j or j lists i.
    """
    n = len(adj_list)
    A = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in adj_list[i]:
            if i == j:
                raise ValueError(f"Self-loop at node {i}")
            A[i, j] = 1
            if symmetrize:
                A[j, i] = 1
    np.fill_diagonal(A, 0)
    return A


def permute_by_conjugation(M: np.ndarray, perm: tuple[int, ...]) -> np.ndarray:
    """Return P M P^T for the permutation `perm`.

    With our convention, this corresponds to simultaneously permuting rows/cols by `perm`:
    (P M P^T)[i,j] = M[perm[i], perm[j]].
    """
    idx = np.ix_(perm, perm)
    return M[idx]


def automorphisms_from_matrix(M: np.ndarray) -> list[tuple[int, ...]]:
    """Brute-force all permutations p with P M P^T == M."""
    n = M.shape[0]
    autos: list[tuple[int, ...]] = []
    for perm in itertools.permutations(range(n)):
        if np.array_equal(permute_by_conjugation(M, perm), M):
            autos.append(perm)
    return autos



adj_1  = {
    0: [1],
    1: [0, 2, 3, 4],
    2: [1],
    3: [1],
    4: [1, 5],
    5: [4],
}


if __name__ == "__main__":
    A = build_adjacency_matrix(adj_1, symmetrize=True)
    L = build_laplacian(adj_1)

    autos_A = automorphisms_from_matrix(A)
    autos_L = automorphisms_from_matrix(L)

    # These should match for simple graphs (A and L carry the same structure).
    print("Adjacency matrix A:\n", A)
    print("\nLaplacian L:\n", L)

    print(f"\nAutomorphisms from A: {len(autos_A)}")
    for perm in autos_A:
        print(perm)

    print(f"\nAutomorphisms from L: {len(autos_L)}")
    for perm in autos_L:
        print(perm)

    if set(autos_A) != set(autos_L):
        print("\nWARNING: Automorphism sets differ between A and L. This can happen if the adjacency list is not symmetric.")


