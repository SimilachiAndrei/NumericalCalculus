import numpy as np


def is_diagonally_dominant(A):
    # Dominanță pe linii
    row_dom = all(2 * np.abs(A[i, i]) > np.sum(np.abs(A[i, :])) for i in range(A.shape[0]))
    # Dominanță pe coloane
    col_dom = all(2 * np.abs(A[j, j]) > np.sum(np.abs(A[:, j])) for j in range(A.shape[0]))
    return row_dom or col_dom


def is_spd(A):
    if not np.allclose(A, A.T):
        return False
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False


def compute_v0(A):
    n = A.shape[0]

    # Metoda 2: Diagonale dominante
    if is_diagonally_dominant(A):
        return np.diag(1 / A.diagonal())

    # Metoda 3: Matrice simetrică pozitiv definită
    if is_spd(A):
        norm_F = np.linalg.norm(A, 'fro')
        return np.eye(n) / norm_F

    # Metoda 4: V0 = alpha*A^T
    alpha_max = 2 / (np.linalg.norm(A, 2) ** 2)
    alpha = 0.9 * alpha_max  # factor de siguranță
    if alpha > 0:
        return alpha * A.T

    # Metoda 5: V0 = alpha*I
    eigvals = np.linalg.eigvals(A)
    alpha = 1 / (np.max(np.abs(eigvals)) + 1e-6)
    if np.max(np.abs(1 - eigvals * alpha)) < 1:
        return alpha * np.eye(n)

    # Metoda 1 (default): normele 1 și infinit
    norm_1 = np.max(np.sum(np.abs(A), axis=0))
    norm_inf = np.max(np.sum(np.abs(A), axis=1))
    return (A.T) / (norm_1 * norm_inf)


def iterative(method, A, eps, kmax):
    n = A.shape[0]
    V0 = compute_v0(A)
    k = 0

    while k < kmax:
        if method == 'schultz':
            V1 = V0 @ (2 * np.eye(n) - A @ V0)
        elif method == 'li1':
            AV0 = A @ V0
            V1 = V0 @ (3 * np.eye(n) - AV0 @ (3 * np.eye(n) - AV0))
        elif method == 'li2':
            V0A = V0 @ A
            term = (np.eye(n) - V0A) @ (3 * np.eye(n) - V0A) ** 2
            V1 = (np.eye(n) + 0.25 * term) @ V0

        delta = np.linalg.norm(V1 - V0, 'fro')
        if delta < eps or delta > 1e10:
            break
        V0 = V1.copy()
        k += 1

    return V1, k


def schultz_rectangular(A, eps, kmax):
    m, n = A.shape
    alpha = 1.9 / (np.linalg.norm(A, 2) ** 2 + 1e-12)
    V0 = alpha * A.T
    for k in range(kmax):
        V1 = V0 @ (2 * np.eye(m) - A @ V0)
        if np.linalg.norm(V1 - V0, 'fro') < eps:
            break
        V0 = V1.copy()
    return V1, k + 1


# Testare matrice  4x4
A_hilbert = np.array([[1 / (i + j + 1) for j in range(4)] for i in range(4)])
print("\n=== Test matrice Hilbert 4x4 ===")
for method in ['schultz', 'li1', 'li2']:
    V_approx, k = iterative(method, A_hilbert, 1e-8, 1000)
    error = np.linalg.norm(A_hilbert @ V_approx - np.eye(4))
    print(f"{method}: {k} iterații, Eroare: {error:.2e}")

# Test matrice diagonal dominantă
A_diag_dom = np.array([[5, 1, 0], [1, 6, 2], [0, 2, 7]])
print("\nV0 pentru matrice diagonal dominantă:", compute_v0(A_diag_dom).diagonal())