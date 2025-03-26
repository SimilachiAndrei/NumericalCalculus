import os
import random
import numpy as np

def generare_matrice_rara(n, density=0.005):
    d = [0.0] * n
    vrar = [{} for _ in range(n)]

    for i in range(n):
        d[i] = random.uniform(1, 10)
        for j in range(i + 1, n):
            if random.random() < density:
                value = random.uniform(1, 10)
                vrar[i][j] = value
                vrar[j][i] = value

    vrar_list = []
    for row in vrar:
        vrar_list.append([(j, val) for j, val in row.items()])
    return n, d, vrar_list


def print_matrice(n, d, vrar_list):
    print("Dimensiune matrice:", n)
    for i in range(n):
        print(f"{d[i]}, {i}, {i}")
    for i, row in enumerate(vrar_list):
        for j, val in row:
            print(f"{val}, {i}, {j}")

###############################################
# Cerința 2: Metoda puterii pentru matrice simetrice sparse
###############################################
def inmultire_cu_scalar(d, vrar_list, x):
    n = len(d)
    y = np.empty(n)
    for i in range(n):
        s = d[i] * x[i]
        for (j, val) in vrar_list[i]:
            s += val * x[j]
        y[i] = s
    return y


def metoda_puterii(d, vrar_list, epsilon, kmax=1000000):
    n = len(d)
    v = np.random.rand(n)
    v = v / np.linalg.norm(v)

    for k in range(kmax):
        w = inmultire_cu_scalar(d, vrar_list, v)
        lam = np.dot(w, v)  # coeficientul Rayleigh
        # Verificare convergență: ||w - λ*v||
        if np.linalg.norm(w - lam * v) <= n * epsilon:
            return lam, v, k + 1
        v = w / np.linalg.norm(w)
    raise Exception("Metoda puterii nu a convergit în numărul maxim de iterații.")


def is_symmetric(d, vrar_list, epsilon):
    n = len(d)
    for i in range(n):
        for (j, val) in vrar_list[i]:
            if i == j:
                continue
            aji = 0.0
            found = False
            for (k, v_val) in vrar_list[j]:
                if k == i:
                    aji = v_val
                    found = True
                    break
            if not found:
                aji = 0.0
            if abs(val - aji) > epsilon:
                return False
    return True


def citire_matrice_A_var1(fisier_matrice_A, epsilon, verificare_zero=True):
    if not os.path.exists(fisier_matrice_A):
        print(f"{fisier_matrice_A} nu există.")
        exit()

    try:
        with open(fisier_matrice_A, 'r') as infile:
            lines = infile.readlines()
            n = int(lines[0].strip())
            d = [0.0] * n
            vrar = [{} for _ in range(n)]
            for line in lines[1:]:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',')
                if len(parts) < 3:
                    continue
                val = float(parts[0].strip())
                i = int(parts[1].strip())
                j = int(parts[2].strip())
                if i < 0 or i >= n or j < 0 or j >= n:
                    print(f"Indici invalizi (i={i}, j={j}) în fișier.")
                    exit()
                if i == j:
                    d[i] += val
                else:
                    if j in vrar[i]:
                        vrar[i][j] += val
                    else:
                        vrar[i][j] = val
            for i in range(n):
                if abs(d[i]) < epsilon and verificare_zero:
                    print(f"Element diagonal nul la linia {i}.")
                    exit()
            vrar_list = []
            for row in vrar:
                vrar_list.append([(j, val) for j, val in row.items()])
            return n, d, vrar_list
    except Exception as e:
        print(f"Eroare la citirea matricei: {e}")
        exit()


def svd_analysis(A, b, epsilon=1e-9):

    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    print("\nValorile singulare ale lui A:")
    print(s)

    rank = np.sum(s > epsilon)
    print("Rangul matricei A:", rank)

    # Calculăm numărul de condiționare: σ_max / σ_min (pentru σ_min > epsilon)
    s_pos = s[s > epsilon]
    if s_pos.size > 0:
        cond_number = np.max(s_pos) / np.min(s_pos)
    else:
        cond_number = np.inf
    print("Numărul de condiționare al lui A:", cond_number)

    # Calculăm pseudoinversa:
    S_inv = np.diag([1 / x if x > epsilon else 0 for x in s])
    A_pinv = Vt.T @ S_inv @ U.T
    print("\nPseudoinversa Moore-Penrose A^I:")
    print(A_pinv)

    # Calculăm soluția xI și norma reziduală:
    xI = A_pinv @ b
    residual = np.linalg.norm(b - A @ xI)
    print("\nSoluția xI a sistemului Ax = b:")
    print(xI)
    print("Norma ||b - A*xI||2 =", residual)




epsilon = 1e-10

n_random = 600
p_random = n_random
n_gen, d_gen, vrar_gen = generare_matrice_rara(n_random, density=0.005)
print_matrice(n_gen, d_gen, vrar_gen)

try:
    lam_gen, v_gen, it_gen = metoda_puterii(d_gen, vrar_gen, epsilon)
    print("\nMetoda puterii pentru matricea generată:")
    print("Aproximarea valorii proprii:", lam_gen)
    print("Vectorul propriu aproximat:", v_gen)

    w_gen = inmultire_cu_scalar(d_gen, vrar_gen, v_gen)
    residual_gen = np.linalg.norm(w_gen - lam_gen * v_gen)
    print("Residual: ||A*v - λ*v|| =", residual_gen)
    print("Număr iterații:", it_gen)
except Exception as e:
    print("Eroare metoda puterii (matrice generată):", e)

fisier = "m_rar_sim_2025_256.txt"
n_fis, d_fis, vrar_fis = citire_matrice_A_var1(fisier, epsilon)

if is_symmetric(d_fis, vrar_fis, epsilon):
    print("Matricea citită este simetrică.")
else:
    print("Matricea citită NU este simetrică.")

try:
    lam_fis, v_fis, it_fis = metoda_puterii(d_fis, vrar_fis, epsilon)
    print("\nMetoda puterii pentru matricea citită din fișier:")
    print("Aproximarea valorii proprii:", lam_fis)
    print("Vectorul propriu aproximat:", v_fis)

    w_fis = inmultire_cu_scalar(d_fis, vrar_fis, v_fis)
    residual_fis = np.linalg.norm(w_fis - lam_fis * v_fis)
    print("Residual: ||A*v - λ*v|| =", residual_fis)
    print("Număr iterații:", it_fis)
except Exception as e:
    print("Eroare metoda puterii (matrice din fișier):", e)


p_dense = 256
n_dense = 128  # p > n

A_dense = np.random.randn(p_dense, n_dense)

b_dense = np.random.randn(p_dense)

svd_analysis(A_dense, b_dense, epsilon)
