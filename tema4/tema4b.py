import numpy as np


# ----------- Funcții auxiliare -----------
def norm_inf(M):
    """Calculează norma infinită a unei matrice."""
    return np.max(np.sum(np.abs(M), axis=1))


def norm_diff(M1, M2):
    """Calculează norma infinită a diferenței a două matrice."""
    return norm_inf(M1 - M2)


def initial_guess(A):
    """
    Calculează V0 conform formulei (5)/(6):
    V0 = A^T / (||A||_1 * ||A||_∞)
    """
    norm1 = np.max(np.sum(np.abs(A), axis=0))  # Norma 1 (max pe coloane)
    norm_inf_val = norm_inf(A)  # Norma infinită (max pe linii)
    return A.T / (norm1 * norm_inf_val)


# ----------- Metode iterative -----------
import numpy as np


def method_schultz(A, eps=1e-6, kmax=10000):
    n = A.shape[0]
    V = initial_guess(A)
    k = 0
    while k < kmax:
        AV = A @ V
        # Construim 2I - AV prin ajustare diagonală
        C = -AV.copy()
        np.fill_diagonal(C, 2 + (-AV.diagonal()))  # 2I - AV = -AV + 2I
        V_next = V @ C
        delta = norm_diff(V_next, V)
        k += 1
        if delta < eps:
            break
        V = V_next
    residual = norm_inf(np.eye(n) - A @ V)
    return V, k, residual


def method_Li_LI1(A, eps=1e-6, kmax=10000):
    n = A.shape[0]
    I = np.eye(n)
    V = initial_guess(A)
    k = 0
    while k < kmax:
        AV = A @ V

        # Compute 3I - AV
        inner_term = -AV.copy()
        np.fill_diagonal(inner_term, inner_term.diagonal() + 3)  # 3I - AV

        # Compute 3I - AV@inner_term
        AV_inner = AV @ inner_term
        outer_term = -AV_inner
        np.fill_diagonal(outer_term, outer_term.diagonal() + 3)
        V_next = V @ outer_term
        delta = norm_diff(V_next, V)
        k += 1
        if delta < eps:
            break
        V = V_next
    residual = norm_inf(I - A @ V_next)
    return V_next, k, residual



def method_Li_LI2(A, eps=1e-6, kmax=10000):
    n = A.shape[0]
    I = np.eye(n)
    V = initial_guess(A)
    k = 0
    while k < kmax:
        VA = V @ A

        # Compute (I - VA)
        I_minus_VA = -VA.copy()
        np.fill_diagonal(I_minus_VA, I_minus_VA.diagonal() + 1)

        # Compute (3I - VA)
        threeI_minus_VA = -VA.copy()
        np.fill_diagonal(threeI_minus_VA, threeI_minus_VA.diagonal() + 3)

        # Multiply terms
        temp = threeI_minus_VA @ threeI_minus_VA  # (3I-VA)Â² first
        term = I_minus_VA @ temp

        V_next = (I + 0.25 * term) @ V
        delta = norm_diff(V_next, V)
        k += 1
        if delta < eps:
            break
        V = V_next
    residual = norm_inf(np.eye(n) - A @ V)
    return V, k, residual


# ----------- Partea 3: Matricea specială -----------
def create_special_matrix(n):
    """Generează matricea cu 1 pe diagonală și 2 pe prima superdiagonală."""
    A = np.eye(n)
    for i in range(n - 1):
        A[i, i + 1] = 2
    return A


def inverse_special_matrix(n):
    """Calculează inversa exactă prin formulă inductivă."""
    A_inv = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            A_inv[i, j] = (-2) ** (j - i)
    return A_inv


# ----------- Bonus: Matrice nepătrată -----------
def bonus_pseudoinverse(A, eps=1e-6, kmax=10000):
    m, n = A.shape
    I_m = np.eye(m)
    V = A.T / (np.max(np.sum(np.abs(A), axis=0)) * norm_inf(A))
    k = 0
    while k < kmax:
        AV = A @ V  # Precompute once (most expensive operation)

        # Compute (2I - AV)
        twoI_minus_AV = -AV.copy()  # Start with -AV
        np.fill_diagonal(twoI_minus_AV, twoI_minus_AV.diagonal() + 2)  # Add 2 to diagonal

        V_next = V @ twoI_minus_AV
        delta = norm_diff(V_next, V)
        k += 1
        if delta < eps:
            break
        V = V_next
    residual = norm_inf(I_m - A @ V_next)
    return V_next, k, residual


# ----------- Testare -----------
if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    # -------------------------------
    # Partea 1 și 2: Test pe matrice generică
    # -------------------------------
    np.random.seed(0)
    n = 4
    A = np.random.rand(n, n) + np.eye(n)  # Matrice nesingulară
    eps = 1e-6
    kmax = 10000

    print("\n" + "=" * 50)
    print("Testare metode iterative pe matrice generică:")
    print("Matricea A:\n", A)

    # Metoda Schultz
    V_schultz, k_schultz, res_schultz = method_schultz(A, eps, kmax)
    print("\nMetoda Schultz:")
    print(f"Iterații: {k_schultz}, Rezidual: {res_schultz:.4e}")

    # Metoda Li & Li (1)
    V_li1, k_li1, res_li1 = method_Li_LI1(A, eps, kmax)
    print("\nMetoda Li & Li (1):")
    print(f"Iterații: {k_li1}, Rezidual: {res_li1:.4e}")

    # Metoda Li & Li (2)
    V_li2, k_li2, res_li2 = method_Li_LI2(A, eps, kmax)
    print("\nMetoda Li & Li (2):")
    print(f"Iterații: {k_li2}, Rezidual: {res_li2:.4e}")

    # -------------------------------
    # Partea 3: Matricea specială
    # -------------------------------
    n_special = 4
    A_special = create_special_matrix(n_special)
    Ainv_exact = inverse_special_matrix(n_special)

    print("\n" + "=" * 50)
    print("Testare partea 3 (matrice specială):")
    print("\nMatricea A specială:\n", A_special)
    print("\nInversa exactă:\n", Ainv_exact)

    # Aproximare cu Schultz
    V_special, k_special, res_special = method_schultz(A_special, eps, kmax)
    diff_norm = norm_diff(Ainv_exact, V_special)
    print("\nAproximare cu Schultz:")
    print(f"Iterații: {k_special}, Rezidual: {res_special:.4e}")
    print(f"Norma diferenței față de exact: {diff_norm:.4e}")

    # -------------------------------
    # Bonus: Matrice nepătrată (3x2)
    # -------------------------------
    A_rect = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
    print("\n" + "=" * 50)
    print("Testare bonus (matrice nepătrată 3x2):")
    print("Matricea A:\n", A_rect)

    V_pseudo, k_pseudo, res_pseudo = bonus_pseudoinverse(A_rect, eps, kmax)
    print("\nPseudoinversa aproximată:\n", V_pseudo)
    print(f"Iterații: {k_pseudo}, Rezidual: {res_pseudo:.4e}")