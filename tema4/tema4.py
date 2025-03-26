import numpy as np

def compute_v0(A):
    norm_1 = np.max(np.sum(np.abs(A), axis=0))
    norm_inf = np.max(np.sum(np.abs(A), axis=1))
    return (A.T) / (norm_1 * norm_inf)

def iterativa(method, A, eps, kmax):
    V0 = compute_v0(A)
    n = A.shape[0]
    k = 0
    while k < kmax:
        if method == 'schultz':
            V1 = V0 @ (2 * np.eye(n) - A @ V0)
        elif method == 'li1':
            AV0 = A @ V0
            V1 = V0 @ (3 * np.eye(n) - AV0 @ (3 * np.eye(n) - AV0))
        elif method == 'li2':
            V0A = V0 @ A
            term = (np.eye(n) - V0A) @ (3 * np.eye(n) - V0A)**2
            V1 = (np.eye(n) + 0.25 * term) @ V0
        delta = np.linalg.norm(V1 - V0, 'fro')
        if delta < eps:
            break
        V0 = V1.copy()
        k += 1
    return V1, k

def schultz_rectangular(A, eps, kmax):
    m, n = A.shape
    alpha = 1.9 / (np.linalg.norm(A, 2)**2)
    V0 = alpha * A.T
    for k in range(kmax):
        V1 = V0 @ (2 * np.eye(m) - A @ V0)
        if np.linalg.norm(V1 - V0, 'fro') < eps:
            break
        V0 = V1.copy()
    return V1, k+1


# Exemplu complex matrice pătratică 3x3 (simetrică, pozitiv definită)
A_square = np.array([
    [4, 1, 1],
    [1, 5, 2],
    [1, 2, 6]
], dtype=float)

# Exemplu complex matrice nepătratică 2x3
A_rect = np.array([
    [1, 2, 3],
    [4, 5, 6]
], dtype=float)

# Test pentru matrice pătratică
print("=== Matrice pătratică 3x3 ===")
methods = ['schultz', 'li1', 'li2']
for method in methods:
    V_approx, k = iterativa(method, A_square, eps=1e-8, kmax=10000)
    norm_error = np.linalg.norm(A_square @ V_approx - np.eye(3))
    print(f"{method.capitalize()}: {k} iterații, Normă eroare: {norm_error:.2e}")

# Comparație cu inversa exactă
A_inv_exact = np.linalg.inv(A_square)
print("\nInversa exactă:\n", A_inv_exact)
print("Aproximare Schultz:\n", np.round(iterativa('schultz', A_square, 1e-8, 10000)[0], 4))

# Test matrice nepătratică
print("\n=== Matrice nepătratică 2x3 ===")
V_pseudo, k = schultz_rectangular(A_rect, eps=1e-6, kmax=1000)
print(f"Pseudoinversă aproximată (Schultz adaptat) după {k} iterații:\n", np.round(V_pseudo, 4))

# Comparație cu pseudoinversa Moore-Penrose
A_pinv_exact = np.linalg.pinv(A_rect)
print("\nPseudoinversă exactă (Moore-Penrose):\n", np.round(A_pinv_exact, 4))
print("Norma diferență:", np.linalg.norm(V_pseudo - A_pinv_exact))