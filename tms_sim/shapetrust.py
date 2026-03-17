# import numpy as np
# from copy import copy

# def internal_value(i: int, trust_matrix: np.ndarray) -> float:
#     '''Compute the internal value of peer i based on the trust matrix.'''
#     # filter np.inf values to avoid issues with summation
#     row_finite_mask = np.isfinite(trust_matrix[i, :])
#     row_sum = np.sum(trust_matrix[i, row_finite_mask])  # sum of row i
#     column_finite_mask = np.isfinite(trust_matrix[:, i])
#     column_sum = np.sum(trust_matrix[column_finite_mask, i])  # sum of column i
#     return 0.5 * (row_sum + column_sum)

# def external_value(i: int, neighbors_of_i: np.ndarray, incoming_values: dict[int, np.ndarray]) -> float:
#     first_summand = 0.0
#     second_summand = 0.0
#     i_incoming_sorted = np.sort(incoming_values[i], axis=0)
#     i_values = np.array([x[0] for x in i_incoming_sorted])
    
#     mi = len(i_values)
#     for t in range(mi):
#         b_i_t = i_values[t]
#         b_i_t_minus_1 = 0.0 if t == 0 else i_values[t-1]
#         first_summand += (b_i_t - b_i_t_minus_1) / (t + 1)
        
#     if mi > 0:
#         first_summand -= i_values[mi-1] / (mi + 1)
        
#     for j in neighbors_of_i:
#         j_incoming_sorted = incoming_values[j][np.argsort(incoming_values[j][:, 0])]
#         j_values = np.array([x[0] for x in j_incoming_sorted])
#         mj = len(j_values)
#         rj = np.where(j_incoming_sorted[:, 1] == i)[0][0]  # index of i in j's incoming sorted list
#         if rj is None:
#             continue
        
#         for t in range(rj + 1, mj):
#             b_j_t = j_values[t]
#             b_j_t_minus_1 = 0.0 if t == 0 else j_values[t-1]
#             second_summand += (b_j_t - b_j_t_minus_1) / (t + 1)
            
#         if mj > 0:
#             second_summand -= j_values[mj-1] / (mj + 1)
    
#     return first_summand + second_summand
    
    
# def shapetrust(trust_matrix: np.ndarray) -> np.ndarray:
#     n = trust_matrix.shape[0]
#     shapley_values = np.zeros(n)
#     for i in range(n):
#         shapley_values[i] = internal_value(i, trust_matrix)
#         neighbors_of_i = np.where(np.isfinite(trust_matrix[i, :]))[0]
#         neighbors_with_i = copy(neighbors_of_i)
#         neighbors_with_i = np.append(neighbors_with_i, i)
#         incoming_values = {}
#         for j in neighbors_with_i:
#             in_idx = np.where(np.isfinite(trust_matrix[:, j]))[0]
#             incoming_values[j] = np.array([[trust_matrix[p,j], p] for p in in_idx])
            
#         shapley_values[i] += external_value(i, neighbors_of_i, incoming_values)
        
#     return shapley_values
import numpy as np

def internal_value(i: int, A: np.ndarray) -> float:
    row_sum = np.sum(A[i, np.isfinite(A[i, :])])
    col_sum = np.sum(A[np.isfinite(A[:, i]), i])
    return 0.5 * (row_sum + col_sum)


def precompute_incoming(A):
    """Precompute sorted incoming values for all nodes."""
    n = A.shape[0]
    incoming = {}

    for j in range(n):
        idx = np.where(np.isfinite(A[:, j]))[0]
        vals = A[idx, j]

        # sort by values
        order = np.argsort(vals)
        incoming[j] = (vals[order], idx[order])  # tuple of arrays

    return incoming


def external_value_fast(i, neighbors_of_i, incoming):
    first_summand = 0.0
    second_summand = 0.0

    # --- i part ---
    i_vals, _ = incoming[i]
    mi = len(i_vals)

    if mi > 0:
        diffs = np.diff(np.concatenate(([0.0], i_vals)))
        first_summand += np.sum(diffs / np.arange(1, mi + 1))
        first_summand -= i_vals[-1] / (mi + 1)

    # --- neighbors ---
    for j in neighbors_of_i:
        j_vals, j_idx = incoming[j]
        mj = len(j_vals)

        # find position of i in sorted list
        matches = np.where(j_idx == i)[0]
        if len(matches) == 0:
            continue
        rj = matches[0]

        if rj + 1 < mj:
            diffs = np.diff(np.concatenate(([0.0], j_vals)))
            second_summand += np.sum(
                diffs[rj + 1:] / np.arange(rj + 2, mj + 1)
            )

        if mj > 0:
            second_summand -= j_vals[-1] / (mj + 1)

    return first_summand + second_summand


def shapetrust(A: np.ndarray) -> np.ndarray:
    n = A.shape[0]
    phi = np.zeros(n)
    incoming = precompute_incoming(A)

    for i in range(n):
        phi[i] = internal_value(i, A)

        neighbors_of_i = np.where(np.isfinite(A[i, :]))[0]
        phi[i] += external_value_fast(i, neighbors_of_i, incoming)

    return phi

import numpy as np
from numba import njit, prange

@njit
def precompute_incoming_numba(A):
    n = A.shape[0]
    incoming_vals = []
    incoming_idx = []

    for j in range(n):
        idx = np.where(np.isfinite(A[:, j]))[0]
        vals = A[idx, j]

        order = np.argsort(vals)
        incoming_vals.append(vals[order])
        incoming_idx.append(idx[order])

    return incoming_vals, incoming_idx


@njit
def internal_value_numba(i, A):
    n = A.shape[0]
    row_sum = 0.0
    col_sum = 0.0

    for j in range(n):
        if np.isfinite(A[i, j]):
            row_sum += A[i, j]
        if np.isfinite(A[j, i]):
            col_sum += A[j, i]

    return 0.5 * (row_sum + col_sum)


@njit
def external_value_numba(i, neighbors, incoming_vals, incoming_idx):
    first_summand = 0.0
    second_summand = 0.0

    # --- i ---
    i_vals = incoming_vals[i]
    mi = len(i_vals)

    if mi > 0:
        prev = 0.0
        for t in range(mi):
            diff = i_vals[t] - prev
            first_summand += diff / (t + 1)
            prev = i_vals[t]
        first_summand -= i_vals[-1] / (mi + 1)

    # --- neighbors ---
    for k in range(len(neighbors)):
        j = neighbors[k]
        j_vals = incoming_vals[j]
        j_idx = incoming_idx[j]
        mj = len(j_vals)

        # find rj
        rj = -1
        for t in range(mj):
            if j_idx[t] == i:
                rj = t
                break

        if rj == -1:
            continue

        prev = 0.0
        for t in range(mj):
            if t == 0:
                prev = 0.0
            diff = j_vals[t] - prev
            if t > rj:
                second_summand += diff / (t + 1)
            prev = j_vals[t]

        if mj > 0:
            second_summand -= j_vals[-1] / (mj + 1)

    return first_summand + second_summand


@njit(parallel=True)
def shapetrust_numba(A):
    n = A.shape[0]
    phi = np.zeros(n)
    
    incoming_vals, incoming_idx = precompute_incoming_numba(A)

    for i in prange(n):  # 🔥 parallel
        phi_i = internal_value_numba(i, A)

        # neighbors
        neighbors = []
        for j in range(n):
            if np.isfinite(A[i, j]):
                neighbors.append(j)

        phi_i += external_value_numba(i, neighbors, incoming_vals, incoming_idx)
        phi[i] = phi_i

    return phi