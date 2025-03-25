import os

def citire_matrice_A_var1(fisier_matrice_A, epsilon, verificare_zero = True):
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

def citire_matrice_A_var2(fisier_matrice_A, epsilon, verificare_zero = True):
    if not os.path.exists(fisier_matrice_A):
        print(f"Fișierul {fisier_matrice_A} nu există.")
        exit()

    n = 0
    valori = []
    ind_col = []
    inceput_linii = [0]
    diag = []

    try:
        with open(fisier_matrice_A, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
            n = int(lines[0])
            diag = [0.0] * n
            linii_temp = [{} for _ in range(n)]

            for line in lines[1:]:
                parts = line.split(',')
                if len(parts) != 3:
                    print(f"Linie invalidă: {line}")
                    exit()
                val = float(parts[0].strip())
                i = int(parts[1].strip())
                j = int(parts[2].strip())

                if i < 0 or i >= n or j < 0 or j >= n:
                    print(f"Indici invalizi (i={i}, j={j})")
                    exit()

                if i == j:
                    diag[i] += val
                else:
                    if j in linii_temp[i]:
                        linii_temp[i][j] += val
                    else:
                        linii_temp[i][j] = val

            for i in range(n):
                if abs(diag[i]) < epsilon and verificare_zero:
                    print(f"Element diagonal nul la linia {i}.")
                    exit()

            current_pos = 0
            for i in range(n):
                sorted_elements = sorted(linii_temp[i].items(), key=lambda x: x[0])
                for j, val in sorted_elements:
                    valori.append(val)
                    ind_col.append(j)
                    current_pos += 1
                inceput_linii.append(current_pos)

            return n, valori, ind_col, inceput_linii, diag

    except Exception as e:
        print(f"Eroare la citire: {e}")
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

def gauss_seidel_var1(d, vrar, b, epsilon=1e-10, max_iter=10000):
    n = len(d)
    x = [0.0] * n
    delta_x = epsilon + 1
    k = 0

    while k < max_iter and delta_x >= epsilon:
        delta_x = 0.0
        for i in range(n):
            old_xi = x[i]
            sum_val = 0.0

            for (j, val) in vrar[i]:
                sum_val += val * x[j]

            new_xi = (b[i] - sum_val) / d[i]
            delta_x = max(delta_x, abs(new_xi - old_xi))
            x[i] = new_xi
        k += 1

    if delta_x < epsilon:
        print(f"Metoda 1 - Convergență în {k} iterații.")
    else:
        print("Metoda 1 - Divergență.")
    return x

def gauss_seidel_var2(valori, ind_col, inceput_linii, diag, b, epsilon=1e-10, max_iter=10000):
    n = len(diag)
    x = [0.0] * n
    delta_x = epsilon + 1
    k = 0

    while k < max_iter and delta_x >= epsilon:
        delta_x = 0.0
        for i in range(n):
            old_xi = x[i]
            sum_non_diag = 0.0
            start = inceput_linii[i]
            end = inceput_linii[i+1]
            for idx in range(start, end):
                j = ind_col[idx]
                sum_non_diag += valori[idx] * x[j]
            new_xi = (b[i] - sum_non_diag) / diag[i]
            delta_x = max(delta_x, abs(new_xi - old_xi))
            x[i] = new_xi
        k += 1

    if delta_x < epsilon:
        print(f"Metoda 2 - Convergență în {k} iterații.")
    else:
        print("Metoda 2 - Divergență.")
    return x

def calcul_norma_var1(d, vrar, x, b):
    residual = []
    for i in range(len(d)):
        sum_ax = d[i] * x[i]
        for (j, val) in vrar[i]:
            sum_ax += val * x[j]
        residual.append(abs(sum_ax - b[i]))
    return max(residual)

def calcul_norma_var2(valori, ind_col, inceput_linii, diag, x, b):
    residual = []
    for i in range(len(diag)):
        sum_ax = diag[i] * x[i]
        start = inceput_linii[i]
        end = inceput_linii[i+1]
        for idx in range(start, end):
            j = ind_col[idx]
            sum_ax += valori[idx] * x[j]
        residual.append(abs(sum_ax - b[i]))
    return max(residual)


def aduna_matrici_var1(d_A, vrar_A, d_B, vrar_B, epsilon):
    n = len(d_A)
    if n != len(d_B):
        print("Dimensiuni diferite pentru metoda 1.")
        exit()

    # Adunare diagonale
    d_C = [d_A[i] + d_B[i] for i in range(n)]

    # Adunare elemente non-diagonale
    vrar_C = []
    for i in range(n):
        combined = {}
        # Procesăm elementele din prima matrice
        for (j, val) in vrar_A[i]:
            combined[j] = val
        # Adăugăm/însumăm elementele din a doua matrice
        for (j, val) in vrar_B[i]:
            if j in combined:
                combined[j] += val
            else:
                combined[j] = val
        # Eliminăm elemente sub epsilon
        vrar_C.append([(j, v) for j, v in combined.items() if abs(v) >= epsilon])

    return d_C, vrar_C


def aduna_matrici_var2(valori_A, ind_col_A, inceput_linii_A, diag_A,
                       valori_B, ind_col_B, inceput_linii_B, diag_B, epsilon):
    n = len(diag_A)
    if n != len(diag_B):
        print("Dimensiuni diferite pentru metoda 2.")
        exit()

    # Adunare diagonale
    diag_C = [diag_A[i] + diag_B[i] for i in range(n)]

    # Adunare elemente non-diagonale
    valori_C = []
    ind_col_C = []
    inceput_linii_C = [0]

    for i in range(n):
        combined = {}
        # Procesăm prima matrice
        start_A = inceput_linii_A[i]
        end_A = inceput_linii_A[i + 1]
        for idx in range(start_A, end_A):
            j = ind_col_A[idx]
            combined[j] = valori_A[idx]

        # Procesăm a doua matrice
        start_B = inceput_linii_B[i]
        end_B = inceput_linii_B[i + 1]
        for idx in range(start_B, end_B):
            j = ind_col_B[idx]
            if j in combined:
                combined[j] += valori_B[idx]
            else:
                combined[j] = valori_B[idx]

        # Sortare și filtrare
        sorted_items = sorted(combined.items())
        for j, val in sorted_items:
            if abs(val) >= epsilon:
                valori_C.append(val)
                ind_col_C.append(j)

        inceput_linii_C.append(len(valori_C))

    return valori_C, ind_col_C, inceput_linii_C, diag_C


def verifica_suma(fisier_a, fisier_b, fisier_aplusb, epsilon, metoda=1):
    # Citim matricele de intrare
    if metoda == 1:
        n_a, d_a, vrar_a = citire_matrice_A_var1(fisier_a, epsilon)
        n_b, d_b, vrar_b = citire_matrice_A_var1(fisier_b, epsilon)
        d_sum, vrar_sum = aduna_matrici_var1(d_a, vrar_a, d_b, vrar_b, epsilon)

        # Citim matricea așteptată
        n_exp, d_exp, vrar_exp = citire_matrice_A_var1(fisier_aplusb, epsilon, False)
    else:
        n_a, valori_a, ind_col_a, inceput_a, diag_a = citire_matrice_A_var2(fisier_a, epsilon)
        n_b, valori_b, ind_col_b, inceput_b, diag_b = citire_matrice_A_var2(fisier_b, epsilon)
        valori_sum, ind_col_sum, inceput_sum, diag_sum = aduna_matrici_var2(
            valori_a, ind_col_a, inceput_a, diag_a,
            valori_b, ind_col_b, inceput_b, diag_b, epsilon
        )

        # Citim matricea așteptată
        n_exp, valori_exp, ind_col_exp, inceput_exp, diag_exp = citire_matrice_A_var2(fisier_aplusb, epsilon, False)

    # Verificare dimensiuni
    if n_a != n_exp:
        print("Dimensiune invalidă a sumei.")
        return False

    # Verificare elemente
    if metoda == 1:
        # Verificare diagonală
        for i in range(n_a):
            if abs(d_sum[i] - d_exp[i]) >= epsilon:
                print(f"Eroare diagonală la linia {i}")
                return False

        # Verificare elemente non-diagonale
        for i in range(n_a):
            sum_dict = {j: val for (j, val) in vrar_sum[i]}
            exp_dict = {j: val for (j, val) in vrar_exp[i]}

            all_keys = set(sum_dict.keys()).union(exp_dict.keys())
            for j in all_keys:
                val_sum = sum_dict.get(j, 0.0)
                val_exp = exp_dict.get(j, 0.0)
                if abs(val_sum - val_exp) >= epsilon:
                    print(f"Eroare la ({i},{j}): {val_sum} vs {val_exp}")
                    return False
    else:
        # Verificare CSR
        if len(valori_sum) != len(valori_exp):
            print("Număr diferit de elemente non-nule.")
            return False

        for i in range(n_a):
            # Verificare diagonală
            if abs(diag_sum[i] - diag_exp[i]) >= epsilon:
                print(f"Eroare diagonală la linia {i}")
                return False

            # Verificare elemente pe linie
            start_sum = inceput_sum[i]
            end_sum = inceput_sum[i + 1]
            line_sum = dict(zip(ind_col_sum[start_sum:end_sum], valori_sum[start_sum:end_sum]))

            start_exp = inceput_exp[i]
            end_exp = inceput_exp[i + 1]
            line_exp = dict(zip(ind_col_exp[start_exp:end_exp], valori_exp[start_exp:end_exp]))

            all_keys = set(line_sum.keys()).union(line_exp.keys())
            for j in all_keys:
                val_sum = line_sum.get(j, 0.0)
                val_exp = line_exp.get(j, 0.0)
                if abs(val_sum - val_exp) >= epsilon:
                    print(f"Eroare la ({i},{j}): {val_sum} vs {val_exp}")
                    return False

    print("Suma matricelor este corectă!")
    return True


def aduna_matrici_mixte(d_A, vrar_A, valori_B, ind_col_B, inceput_linii_B, diag_B, epsilon):
    n = len(d_A)
    # Inițializăm rezultatul în format var1
    d_C = [d_A[i] + diag_B[i] for i in range(n)]
    vrar_C = [{} for _ in range(n)]

    # Adăugăm elementele din prima matrice (var1)
    for i in range(n):
        for (j, val) in vrar_A[i]:
            vrar_C[i][j] = val

    # Adăugăm elementele din a doua matrice (var2)
    for i in range(n):
        start = inceput_linii_B[i]
        end = inceput_linii_B[i + 1]
        for idx in range(start, end):
            j = ind_col_B[idx]
            val = valori_B[idx]
            if j in vrar_C[i]:
                vrar_C[i][j] += val
            else:
                vrar_C[i][j] = val

    # Convertim dicționarele în liste și eliminăm elementele sub epsilon
    vrar_rezultat = []
    for i in range(n):
        vrar_rezultat.append([(j, v) for j, v in vrar_C[i].items() if abs(v) >= epsilon])

    return d_C, vrar_rezultat


def verifica_suma_diferite(fisier_var1, fisier_var2, fisier_aplusb, epsilon):
    # Citim matricele
    n1, d1, vrar1 = citire_matrice_A_var1(fisier_var1, epsilon)
    n2, valori2, ind_col2, inceput2, diag2 = citire_matrice_A_var2(fisier_var2, epsilon)

    if n1 != n2:
        print("Matricele au dimensiuni diferite!")
        return False

    # Facem suma
    d_sum, vrar_sum = aduna_matrici_mixte(d1, vrar1, valori2, ind_col2, inceput2, diag2, epsilon)

    # Citim rezultatul așteptat (în format var1)
    n_exp, d_exp, vrar_exp = citire_matrice_A_var1(fisier_aplusb, epsilon, False)

    # Verificăm
    ok = True
    for i in range(n1):
        # Verificăm diagonala
        if abs(d_sum[i] - d_exp[i]) >= epsilon:
            print(f"Eroare diagonală la linia {i}: {d_sum[i]} vs {d_exp[i]}")
            ok = False

        # Verificăm elementele non-diagonale
        sum_dict = {j: val for (j, val) in vrar_sum[i]}
        exp_dict = {j: val for (j, val) in vrar_exp[i]}
        all_keys = set(sum_dict.keys()).union(exp_dict.keys())
        for j in all_keys:
            val_sum = sum_dict.get(j, 0.0)
            val_exp = exp_dict.get(j, 0.0)
            if abs(val_sum - val_exp) >= epsilon:
                print(f"Eroare la ({i},{j}): {val_sum} vs {val_exp}")
                ok = False


    if ok:
        print("Suma este corectă!")
    return ok



epsilon = 1e-10
max_iter = 10000

print("----- Metoda 1 -----")
n1, d1, vrar1 = citire_matrice_A_var1("a_2.txt", epsilon)
n_b1, b1 = citire_vector_b("b_2.txt")

if n1 != n_b1:
    print("Dimensiuni incompatibile între matrice și vector.")
    exit()

x1 = gauss_seidel_var1(d1, vrar1, b1, epsilon, max_iter)
norma1 = calcul_norma_var1(d1, vrar1, x1, b1)
print(f"Norma ||Ax - b||_inf: {norma1}")
print("Soluția:", x1)

print("\n----- Metoda 2 -----")
n2, valori2, ind_col2, inceput_linii2, diag2 = citire_matrice_A_var2("a_2.txt", epsilon)
n_b2, b2 = citire_vector_b("b_2.txt")

if n2 != n_b2:
    print("Dimensiuni incompatibile între matrice și vector.")
    exit()

x2 = gauss_seidel_var2(valori2, ind_col2, inceput_linii2, diag2, b2, epsilon, max_iter)
norma2 = calcul_norma_var2(valori2, ind_col2, inceput_linii2, diag2, x2, b2)
print(f"Norma ||Ax - b||_∞: {norma2}")
print("Soluția:", x2)


print("\n----- Bonus: Verificare sumă matrici -----")
print("Metoda 1:")
verifica_suma("a.txt", "b.txt", "aplusb.txt", epsilon, metoda=1)

print("\nMetoda 2:")
verifica_suma("a.txt", "b.txt", "aplusb.txt", epsilon, metoda=2)

print("\n----- Suma matrice var1 + var2 -----")
verifica_suma_diferite("a.txt", "b.txt", "aplusb.txt", epsilon)