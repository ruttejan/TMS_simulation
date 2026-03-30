import numpy as np


def _as_square_matrix(C: np.ndarray) -> np.ndarray:
	"""Validate and return C as a float square matrix."""
	C = np.asarray(C, dtype=float)
	if C.ndim != 2:
		raise ValueError("Input must be a 2D matrix.")
	m, n = C.shape
	if m != n:
		raise ValueError("Dimension of given matrix are wrong. (n != m)")
	return C


def normalize_trust_matrix(C: np.ndarray, pretrusted: list[int] | np.ndarray) -> np.ndarray:
	"""Row-normalize trust matrix and handle zero rows.

	For rows with zero outgoing trust, fallback to uniform distribution over
	pretrusted peers (if provided), otherwise to uniform over all peers.

	Args:
		C: Trust matrix.
		pretrusted: Indices of pretrusted peers (0-based).
	"""

	C = _as_square_matrix(C).copy()
	n = C.shape[0]

	pretrusted = np.asarray(pretrusted, dtype=int)
	fallback = np.zeros(n, dtype=float)
	if pretrusted.size > 0:
		fallback[pretrusted] = 1.0 / pretrusted.size
	else:
		fallback = np.ones(n, dtype=float) / n

	C = np.maximum(C, 0)  # Ensure non-negative trust values
	row_sums = C.sum(axis=1)
	for i in range(n):
		if row_sums[i] > 0:
			C[i, :] = C[i, :] / row_sums[i]
		else:
			C[i, :] = fallback
	return C

def eigentrust_iteration(
	C: np.ndarray,
	trust_v: np.ndarray | None = None,
	eps: float = 1e-10,
	max_iter: int = 100_000,
) -> np.ndarray:
	"""EigenTrust iteration without pretrusted peers."""
	if eps < 0:
		raise ValueError("Epsilon is a negative number.")

	C = _as_square_matrix(C)
	n = C.shape[0]
 
	C = normalize_trust_matrix(C, pretrusted=[])  # Normalize without pretrusted fallback

	if trust_v is None:
		trust_v = np.ones(n, dtype=float) / n
	else:
		trust_v = np.asarray(trust_v, dtype=float)
		if trust_v.ndim != 1 or trust_v.shape[0] != n:
			raise ValueError("Initial trust_v must be a vector of length n.")

	for i in range(max_iter):
		trust_v_new = C.T @ trust_v
		delta = np.linalg.norm(trust_v_new - trust_v)
		trust_v = trust_v_new
		if delta < eps:
			break

	return trust_v


def eigentrust(
	C: np.ndarray,
	pretrusted: list[int] | np.ndarray,
	alpha: float,
	eps: float = 1e-10,
	max_iter: int = 100_000,
) -> np.ndarray:
	"""Basic EigenTrust implementation with pretrusted peers."""
	if eps < 0:
		raise ValueError("Epsilon is a negative number.")

	C = _as_square_matrix(C)
	n = C.shape[0]

	pretrusted = np.asarray(pretrusted, dtype=int)
	num_pretrusted = pretrusted.size
	if num_pretrusted == 0 or alpha == 0.0:
		return eigentrust_iteration(C, eps=eps, max_iter=max_iter)

	C = normalize_trust_matrix(C, pretrusted)
	# check if rows sum to 1
	row_sums = C.sum(axis=1)
	if not np.allclose(row_sums, 1.0):
		raise ValueError("Row normalization failed: rows do not sum to 1.")

	rho = np.zeros(n, dtype=float)
	rho[pretrusted] = 1.0 / num_pretrusted
	

	trust_v = rho.copy()
	for i in range(max_iter):
      	# check if C contains infs or nans
		if np.isnan(C).any() or np.isinf(C).any():
			raise ValueError("Trust matrix contains NaN or Inf values.")
		# check if trust_v contains infs or nans
		if np.isnan(trust_v).any() or np.isinf(trust_v).any():
			raise ValueError("Trust vector contains NaN or Inf values.")

		trust_v_new = C.T @ trust_v
		trust_v_new = (1.0 - alpha) * trust_v_new + alpha * rho
		delta = np.linalg.norm(trust_v_new - trust_v)
		trust_v = trust_v_new
		if delta < eps:
			break

	return trust_v