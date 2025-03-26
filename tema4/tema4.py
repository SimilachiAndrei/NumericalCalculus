import numpy as np

def is_diagonally_dominant(A):
    n = A.shape[0]
    row_dom = all(2 * abs(A[i, i]) > sum(abs(A[i, :])) for i in range(n))
    col_dom = all(2 * abs(A[j, j]) > sum(abs(A[:, j])) for j in range(n))
    return row_dom or col_dom


def is_spd(A):
    if not np.allclose(A, A.T):
        return False
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False


def compute_alpha_for_method5(A):
    eigvals = np.linalg.eigvals(A)
    return 1 / (np.max(np.abs(eigvals)) + 1e-12)


def compute_v0(A):
    n = A.shape[0]

    # Metoda 2: Diagonale dominante
    if is_diagonally_dominant(A):
        return np.diag(1 / A.diagonal())

    # Metoda 3: Matrice SPD
    if is_spd(A):
        norm_F = np.linalg.norm(A, 'fro')
        return np.eye(n) / norm_F

    # Metoda 4: V0 = alpha*A^T
    norm_2 = np.linalg.norm(A, 2)
    alpha_max = 2 / (norm_2 ** 2 + 1e-12)
    if alpha_max > 0:
        return 0.9 * alpha_max * A.T  # 0.9 pentru stabilitate

    # Metoda 5: V0 = alpha*I_n
    alpha = compute_alpha_for_method5(A)
    if np.max(np.abs(1 - alpha * np.linalg.eigvals(A))) < 1:
        return alpha * np.eye(n)

    # Metoda 1 (default): Norme 1 și infinit
    norm_1 = np.max(np.sum(np.abs(A), axis=0))
    norm_inf = np.max(np.sum(np.abs(A), axis=1))
    return A.T / (norm_1 * norm_inf)


def iterative_method(method, A, eps=1e-8, kmax=10000):
    n = A.shape[0]
    V0 = compute_v0(A)
    k = 0

    for k in range(kmax):
        if method == 'schultz':
            V1 = V0 @ (2 * np.eye(n) - A @ V0)
        elif method == 'li1':
            AV0 = A @ V0
            inner = 3 * np.eye(n) - AV0
            V1 = V0 @ (3 * np.eye(n) - AV0 @ inner)
        elif method == 'li2':
            V0A = V0 @ A
            term = (np.eye(n) - V0A) @ (3 * np.eye(n) - V0A) ** 2
            V1 = (np.eye(n) + 0.25 * term) @ V0

        delta = np.linalg.norm(V1 - V0, 'fro')
        if delta < eps or delta > 1e10 or k == kmax - 1:
            break
        V0 = V1.copy()

    return V1, k + 1


def schultz_rectangular(A, eps=1e-8, kmax=1000):
    m, n = A.shape
    alpha = 1.9 / (np.linalg.norm(A, 2) ** 2 + 1e-12)
    V0 = alpha * A.T

    for k in range(kmax):
        V1 = V0 @ (2 * np.eye(m) - A @ V0)
        if np.linalg.norm(V1 - V0, 'fro') < eps:
            break
        V0 = V1.copy()

    return V1, k + 1


print("=== Test 1: Matrice diagonal dominantă (Metoda 2) ===")
A_diag_dom = np.array([
    [5, 1, 0],
    [1, 6, 2],
    [0, 2, 7]
])
V0_expected = np.diag([1 / 5, 1 / 6, 1 / 7])
V0_actual = compute_v0(A_diag_dom)
print("V0 așteptat:\n", V0_expected)
print("V0 calculat:\n", np.round(V0_actual, 4))
print("Potrivire:", np.allclose(V0_actual, V0_expected, atol=1e-4))

print("\n=== Test 2: Matrice SPD (Metoda 3) ===")
A_spd = np.array([
    [4, 1, 1],
    [1, 5, 2],
    [1, 2, 6]
])
norm_F = np.linalg.norm(A_spd, 'fro')
V0_expected = np.eye(3) / norm_F
V0_actual = compute_v0(A_spd)
print("V0 așteptat:\n", np.round(V0_expected, 4))
print("V0 calculat:\n", np.round(V0_actual, 4))
print("Potrivire:", np.allclose(V0_actual, V0_expected, atol=1e-4))

print("\n=== Test 3: Metoda 4 (V0 = alpha*A^T) ===")
A_random = np.random.rand(3, 3) * 0.5 + np.eye(3) * 5
V0_actual = compute_v0(A_random)
alpha = 0.9 * 2 / (np.linalg.norm(A_random, 2) ** 2)
V0_expected = alpha * A_random.T
print("Alpha calculat:", alpha)
print("Potrivire:", np.allclose(V0_actual, V0_expected, atol=1e-4))

print("\n=== Test 4: Metoda 5 (V0 = alpha*I) ===")
A_eig = np.array([[2, 0.5], [0.5, 3]])
alpha = compute_alpha_for_method5(A_eig)
V0_expected = alpha * np.eye(2)
V0_actual = compute_v0(A_eig)
print("Alpha calculat:", alpha)
print("V0 calculat:\n", V0_actual)
print("Condiție satisfăcută:", np.max(np.abs(1 - alpha * np.linalg.eigvals(A_eig))) < 1)

print("\n=== Test 5: Metoda default (Norme 1/inf) ===")
A_gen = np.array([[1, 2], [3, 4]])
norm_1 = max(4, 6)
norm_inf = max(3, 7)
V0_expected = A_gen.T / (norm_1 * norm_inf)
V0_actual = compute_v0(A_gen)
print("V0 așteptat:\n", V0_expected)
print("V0 calculat:\n", V0_actual)
print("Potrivire:", np.allclose(V0_actual, V0_expected))

print("\n=== Test 6: Convergență algoritmi iterativi ===")
methods = ['schultz', 'li1', 'li2']
for method in methods:
    V_inv, k = iterative_method(method, A_spd)
    error = np.linalg.norm(A_spd @ V_inv - np.eye(3))
    print(f"{method.upper()}: {k} iterații, Eroare: {error:.2e}")

print("\n=== Test 7: Bonus - Matrice nepătratică ===")
A_rect = np.array([[1, 2], [3, 4], [5, 6]])
V_pseudo, k = schultz_rectangular(A_rect)
A_pinv = np.linalg.pinv(A_rect)
error = np.linalg.norm(V_pseudo - A_pinv)
print(f"Pseudoinversă aproximată ({k} iterații):\n", np.round(V_pseudo, 4))
print("Eroare față de pseudoinversa exactă:", error)