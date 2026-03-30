import numpy as np

def internal_value(i: int, A: np.ndarray) -> float:
    row_sum = np.sum(A[i, np.isfinite(A[i, :])])
    col_sum = np.sum(A[np.isfinite(A[:, i]), i])
    return 0.5 * (row_sum + col_sum)


def precompute_incoming(A: np.ndarray, max: bool = False):
    """Precompute sorted incoming values for all nodes."""
    n = A.shape[0]
    incoming = {}

    for j in range(n):
        idx = np.where(np.isfinite(A[:, j]))[0]
        vals = A[idx, j]

        # sort by values
        order = np.argsort(vals)
        if max:
            order = order[::-1]  # sort in descending order
            
        incoming[j] = (vals[order], idx[order])  # tuple of arrays

    return incoming


def external_value_fast(i : int, neighbors_of_i: np.ndarray, incoming: dict) -> float:
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


def shapetrust(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = A.shape[0]
    internal_phi = np.zeros(n)
    external_phi = np.zeros(n)
    incoming = precompute_incoming(A)

    for i in range(n):
        internal_phi[i] = internal_value(i, A)

        neighbors_of_i = np.where(np.isfinite(A[i, :]))[0]
        external_phi[i] += external_value_fast(i, neighbors_of_i, incoming)

    return internal_phi, external_phi

from numba import njit, prange # type: ignore

@njit
def precompute_incoming_numba(A, max= False):
    n = A.shape[0]
    incoming_vals = []
    incoming_idx = []

    for j in range(n):
        idx = np.where(np.isfinite(A[:, j]))[0]
        vals = A[idx, j]

        order = np.argsort(vals)  # sort in descending order
        if max:
            order = order[::-1]
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
def shapetrust_numba(A, max=False):
    n = A.shape[0]
    internal_phi = np.zeros(n)
    external_phi = np.zeros(n)

    incoming_vals, incoming_idx = precompute_incoming_numba(A, max=max)

    for i in prange(n):
        internal_phi[i] = internal_value_numba(i, A)

        # neighbors
        neighbors = []
        for j in range(n):
            if np.isfinite(A[i, j]):
                neighbors.append(j)

        external_phi[i] = external_value_numba(i, np.array(neighbors), incoming_vals, incoming_idx)

    return internal_phi, external_phi