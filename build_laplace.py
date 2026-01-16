# Created by Micah
# Date: 11/15/25
# Time: 2:39 PM
# Project: GraphTheory_PlatonicSolids
# File: Build.py
import numpy as np

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