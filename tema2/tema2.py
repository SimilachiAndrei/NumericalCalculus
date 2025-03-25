import numpy as np

n = 10
epsilon = 1.0000000000000001e-16
A_initR = np.random.uniform(-10,10,(n,n))
bR = np.random.uniform(-10,10,n)
dUR = np.random.uniform(-10,10,n)

dUR[dUR == 0] = np.random.choice([-10,10], size = np.count_nonzero(dUR == 0))

def main():
    # input
    A_init = A_initR.copy()
    b = bR.copy()
    dU = dUR.copy()

    # copiaza matricea initiala
    A = A_init.copy()

    # descompunerea LU
    for p in range(n):
        # calculeaza liniile din L p (coloanele de la 0 la p)
        for i in range(p + 1):
            sum_L = np.dot(A[p, :i], A[:i, i])
            if np.abs(dU[i]) < epsilon:
                print("Cannot perform LU decomposition: division by zero in L at i={}".format(i))
                return
            A[p, i] = (A_init[p, i] - sum_L) / dU[i]

        # calculeaza liniile din U p (coloanele de la p + 1 la n - 1)
        for j in range(p + 1, n):
            sum_U = np.dot(A[p, :p], A[:p, j])
            if np.abs(A[p, p]) < epsilon:
                print("Cannot perform LU decomposition: division by zero in U at p={}, j={}".format(p, j))
                return
            A[p, j] = (A_init[p, j] - sum_U) / A[p, p]

    # calculeaza determinantul
    det_L = np.prod(np.diag(A))
    det_U = np.prod(dU)
    det_A = det_L * det_U
    print("Determinant of A:", det_A)

    # metoda substitutiei directe (Ly = b)
    y = np.zeros(n)
    for i in range(n):
        sum_ly = np.dot(A[i, :i], y[:i])
        if np.abs(A[i, i]) < epsilon:
            print("Singular matrix in forward substitution at i={}".format(i))
            return
        y[i] = (b[i] - sum_ly) / A[i, i]

    # metoda substitutiei inverse (Ux = y)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        sum_ux = np.dot(A[i, i +  1:], x[i + 1:])
        if np.abs(dU[i]) < epsilon:
            print("Singular matrix in backward substitution at i={}".format(i))
            return
        x[i] = (y[i] - sum_ux) / dU[i]
    print("xLU solution:", x)

    # norma reziduala = ||Ainit * xLU - b||2
    residual = A_init.dot(x) - b
    residual_norm = np.linalg.norm(residual, 2)
    print("Residual norm ||A_init xLU - b||_2:", residual_norm)

    try:
        #aici practic rezolva cu numpy Ax = b si dupa compara cu x obtinut prin descomp. LU folosind norma euclidiana
        x_lib = np.linalg.solve(A_init, b)
        print("Numpy solution:", x_lib)
        norm_diff = np.linalg.norm(x - x_lib, 2)
        print("||xLU - x_lib||_2:", norm_diff)

        #aici compara solutia obtinuta prin descomp. LU cu A^(-1)*b  de la Ax = b  A^(-1)|  <=> x = A^(-1)*b
        A_inv = np.linalg.inv(A_init)
        A_inv_b = A_inv.dot(b)
        norm_diff_inv = np.linalg.norm(x - A_inv_b, 2)
        print("||xLU - A_inv b||_2:", norm_diff_inv)
    except np.linalg.LinAlgError:
        print("Numpy could not solve the system (matrix is singular).")


def main_bonus():
    A_init = A_initR.copy()
    b = bR.copy()
    dU = dUR.copy()

    L_flat = np.zeros(n * (n + 1) // 2)
    U_flat_size = (n - 1) * n // 2
    U_flat = np.zeros(U_flat_size)

    for p in range(n):
        for i in range(p + 1):
            idx_L = p * (p + 1) // 2 + i
            sum_L = 0.0
            for k in range(i):
                # Calculate index for U_flat[k, i]
                idx_U_k_i = (k * (2*(n-1) - k +1)) // 2 + (i - k -1)
                sum_L += L_flat[p * (p + 1) // 2 + k] * U_flat[idx_U_k_i]
            if np.abs(dU[i]) < epsilon:
                print("Nu se poate calcula LU: diviziune cu zero în L la i={}".format(i))
                return
            L_flat[idx_L] = (A_init[p, i] - sum_L) / dU[i]

        for j in range(p + 1, n):
            idx_U = (p * (2*(n-1) - p +1)) // 2 + (j - p -1)
            sum_U = 0.0
            for k in range(p):
                # Calculate index for U_flat[k, j]
                idx_U_k_j = (k * (2*(n-1) - k +1)) // 2 + (j - k -1)
                sum_U += L_flat[p * (p + 1) // 2 + k] * U_flat[idx_U_k_j]
            if np.abs(L_flat[p * (p + 1) // 2 + p]) < epsilon:
                print("Nu se poate calcula LU: diviziune cu zero în U la p={}, j={}".format(p, j))
                return
            U_flat[idx_U] = (A_init[p, j] - sum_U) / L_flat[p * (p + 1) // 2 + p]

    y = np.zeros(n)
    for i in range(n):
        idx = i * (i + 1) // 2
        sum_ly = sum(L_flat[idx + j] * y[j] for j in range(i))
        y[i] = (b[i] - sum_ly) / L_flat[idx + i]

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        sum_ux = 0.0
        for j in range(i + 1, n):
            idx_U = (i * (2*(n-1) - i +1)) // 2 + (j - i -1)
            sum_ux += U_flat[idx_U] * x[j]
        x[i] = (y[i] - sum_ux) / dU[i]

    # Reconstruct L and U matrices
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            idx = i * (i + 1) // 2 + j
            L[i, j] = L_flat[idx]

    U = np.zeros((n, n))
    for i in range(n):
        U[i, i] = dU[i]
        for j in range(i + 1, n):
            idx = (i * (2*(n-1) - i +1)) // 2 + (j - i -1)
            U[i, j] = U_flat[idx]

    LU = L @ U

    print("xLU (bonus):", x)
    print("Matricea LU (bonus):\n", LU)
    print("Matricea A originală:\n", A_init)
    print("Eroarea ||A - LU||_2:", (np.linalg.norm(A_init - LU, 2)))

if __name__ == "__main__":
    main()
    print("BONUS:")
    main_bonus()
