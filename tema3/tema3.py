import os

def citire_matrice_A_var1(fisier_matrice_A, epsilon):
    if not os.path.exists(fisier_matrice_A):
        print(f"{fisier_matrice_A} nu există.")
        exit()

    n = 0
    d = []
    vrar = []

    try:
        with open(fisier_matrice_A, 'r') as infile:
            lines = infile.readlines()
            n = int(lines[0].strip())
            d = [0.0] * n
            vrar = [{} for _ in range(n)]  # Folosim dicționare pentru sumarea valorilor

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

        # Verificăm elementele diagonale
        for i in range(n):
            if abs(d[i]) < epsilon:
                print(f"Element diagonal nul la linia {i}.")
                exit()

        # Convertim dicționarele în liste de tupluri
        vrar_list = []
        for row in vrar:
            vrar_list.append([(j, val) for j, val in row.items()])

        return n, d, vrar_list

    except Exception as e:
        print(f"Eroare la citirea matricei: {e}")
        exit()

def citire_vector_b(fisier_vector_b):
    if not os.path.exists(fisier_vector_b):
        print(f"{fisier_vector_b} nu există.")
        exit()

    n = 0
    b = []

    try:
        with open(fisier_vector_b, 'r') as infile:
            lines = infile.readlines()
            n = int(lines[0].strip())
            if len(lines) != n + 1:
                print("Dimensiune vector incorectă.")
                exit()

            b = [float(line.strip()) for line in lines[1:]]
            return n, b

    except Exception as e:
        print(f"Eroare la citirea vectorului: {e}")
        exit()

# Parametrii
epsilon = 1e-10
max_iter = 10000

# Citire date
n, d, vrar = citire_matrice_A_var1("a_5.txt", epsilon)
n_b, b = citire_vector_b("b_5.txt")

if n != n_b:
    print("Dimensiuni incompatibile între matrice și vector.")
    exit()

# Inițializare
x = [0.0] * n
k = 0
delta_x = epsilon + 1

# Gauss-Seidel
while k < max_iter and delta_x >= epsilon:
    delta_x = 0.0
    for i in range(n):
        old_xi = x[i]
        sum_val = 0.0

        # Sumăm elementele nenule de pe linia i (non-diagonal)
        for (j, val) in vrar[i]:
            sum_val += val * x[j]

        # Calcul noua valoare
        new_xi = (b[i] - sum_val) / d[i]

        # Actualizăm delta_x
        current_diff = abs(new_xi - old_xi)
        if current_diff > delta_x:
            delta_x = current_diff

        # Actualizăm x[i]
        x[i] = new_xi

    k += 1

# Verificare convergență
if delta_x < epsilon:
    print(f"Soluție găsită în {k} iterații.")
else:
    print("Divergență.")

# Calcul normă reziduală
residual = []
for i in range(n):
    sum_ax = d[i] * x[i]
    for (j, val) in vrar[i]:
        sum_ax += val * x[j]
    residual.append(abs(sum_ax - b[i]))

norma = max(residual)
print(f"Norma ||Ax - b||_inf: {norma}")

print(x)