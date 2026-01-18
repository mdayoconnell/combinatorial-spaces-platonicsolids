import time
from datetime import datetime
import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher
from networkx.algorithms.tree import SpanningTreeIterator


def apply_edge_perm(mask: int, ep: list[int]) -> int:
    """Apply an edge permutation to an edge-bitmask.

    This implementation iterates only over set bits (edges present), which is faster than
    scanning through all bit positions up to bit_length(mask).
    """
    out = 0
    while mask:
        lsb = mask & -mask
        i = lsb.bit_length() - 1
        out |= 1 << ep[i]
        mask ^= lsb
    return out


def main() -> None:
    print(f"Run started: {datetime.now().isoformat(timespec='seconds')}")
    t0 = time.perf_counter()

    G = nx.icosahedral_graph()
    t_graph = time.perf_counter()

    # Fix a consistent edge ordering so trees can be represented as bitmasks.
    edges = [tuple(sorted(e)) for e in G.edges()]
    edges.sort()
    edge_index = {e: i for i, e in enumerate(edges)}
    t_index = time.perf_counter()

    # Enumerate automorphisms (graph symmetries).
    GM = GraphMatcher(G, G)
    automorphisms = list(GM.isomorphisms_iter())
    t_autos = time.perf_counter()
    print("computed number of dual graph automorphisms = {}".format(len(automorphisms)))  
    # Convert each node automorphism into an induced edge permutation.
    def edge_perm_from_auto(auto: dict[int, int]) -> list[int]:
        perm: list[int] = []
        for (u, v) in edges:
            u2, v2 = auto[u], auto[v]
            e2 = tuple(sorted((u2, v2)))
            perm.append(edge_index[e2])
        return perm

    edge_perms = [edge_perm_from_auto(a) for a in automorphisms]
    t_edgeperms = time.perf_counter()

    # Enumerate spanning trees, convert each to a bitmask, and canonicalize/dedup on the fly.
    t_loop = time.perf_counter()

    n_trees = 0
    classes: set[int] = set()
    for T in SpanningTreeIterator(G):
        mask = 0
        for (u, v) in T.edges():
            e = tuple(sorted((u, v)))
            mask |= 1 << edge_index[e]

        n_trees += 1

        # Canonicalize each tree by taking the minimum image under the automorphism group.
        canon_mask = min(apply_edge_perm(mask, ep) for ep in edge_perms)
        classes.add(canon_mask)

    t_loop_done = time.perf_counter()

    print("computed number of spanning trees = {}".format(n_trees))
    print(f"computed number of equivalence classes = {len(classes)}")

    print("\nTiming (seconds):")
    print(f"  build graph:            {t_graph - t0:.6f}")
    print(f"  edge indexing:          {t_index - t_graph:.6f}")
    print(f"  automorphisms:          {t_autos - t_index:.6f}")
    print(f"  edge perms:             {t_edgeperms - t_autos:.6f}")
    print(f"  trees + canon/dedup:    {t_loop_done - t_loop:.6f}")
    print(f"  total:                 {t_loop_done - t0:.6f}")


if __name__ == "__main__":
    main()