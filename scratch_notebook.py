# Created by Micah
# Date: 11/15/25
# Time: 12:48 PM
# Project: GraphTheory_PlatonicSolids
# File: Notebook.py

import numpy as np

from build_laplace import build_laplacian

adj_icosa = {
    0: [2, 4, 6, 8, 10],
    1: [3, 4, 6, 9, 11],
    2: [0, 5, 7, 8, 10],
    3: [1, 5, 7, 9, 11],
    4: [0, 1, 6, 8, 9],
    5: [2, 3, 7, 8, 9],
    6: [0, 1, 4, 10, 11],
    7: [2, 3, 5, 10, 11],
    8: [0, 2, 4, 5, 9],
    9: [1, 3, 4, 5, 8],
    10: [0, 2, 6, 7, 11],
    11: [1, 3, 6, 7, 10],
}


# Showing that we get the same result for deleting any index

L = build_laplacian(adj_icosa)

print(L)

L_minor = np.delete(np.delete(L, 1, axis=0), 1, axis=1)

print('Deleting row and column {0}'.format(1))
print("Our minor matrix is:")
print(L_minor)

det = np.linalg.det(L_minor)
tree_count = round(det)   # det is floating, so round to nearest int

print('Determinant of matrix is:')
print(det)
print(tree_count)