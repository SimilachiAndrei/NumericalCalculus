import math

# Exemplu predefinit: P(x) = x³ -6x² +11x -6 (rădăcinile 1, 2, 3)
coeffs = [1.0, -6.0, 11.0, -6.0]
epsilon = 1e-6
kmax = 1000
num_start_points = 20

# 1. Calcul interval [-R, R]
a0 = coeffs[0]
A = max(abs(coeff) for coeff in coeffs[1:]) if len(coeffs) > 1 else 0.0
R = (abs(a0) + A) / abs(a0)
print(f"1. Intervalul calculat: [{-R:.2f}, {R:.2f}]\n")


# 2. Calcul derivate P' și P''
def compute_derivatives(c):
    n = len(c) - 1
    if n < 1: return [], []
    p = [c[i] * (n - i) for i in range(n)]  # P'
    m = len(p) - 1
    pp = [p[i] * (m - i) for i in range(m)] if m >= 1 else []  # P''
    return p, pp


coeffs_p, coeffs_pp = compute_derivatives(coeffs)
print(f"2. Derivata P': {[round(x, 2) for x in coeffs_p]}")
print(f"   Derivata P'': {[round(x, 2) for x in coeffs_pp]}\n")


# 3. Evaluare polinom cu Horner
def horner(c, x):
    result = 0.0
    for coeff in c:
        result = result * x + coeff
    return result


# 4. Metoda Halley
def halley(x0):
    x = x0
    for _ in range(kmax):
        P = horner(coeffs, x)
        P_prime = horner(coeffs_p, x)
        P_pp = horner(coeffs_pp, x) if coeffs_pp else 0.0

        denominator = 2 * (P_prime ** 2) - P * P_pp
        if abs(denominator) < epsilon:
            return None  # Evitare împărțire la zero

        delta = (2 * P * P_prime) / denominator
        x_new = x - delta

        if abs(delta) < epsilon:  # Convergență
            return x_new  # Returnăm valoarea nerotunjită pentru comparații
        x = x_new

    return None  # Divergență


# ------------------- BONUS: Metode N⁴ și N⁵ din articol -------------------
def method_N4(x0):
    x = x0
    for _ in range(kmax):
        fx = horner(coeffs, x)
        fp = horner(coeffs_p, x)
        if abs(fp) < epsilon:
            return None

        y = x - fx / fp
        fy = horner(coeffs, y)

        numerator = fx ** 2 + fy ** 2
        denominator = fp * (fx - fy)
        if abs(denominator) < epsilon:
            return None

        x_new = x - numerator / denominator
        if abs(x_new - x) < epsilon:
            return x_new
        x = x_new
    return None


def method_N5(x0):
    x = x0
    for _ in range(kmax):
        fx = horner(coeffs, x)
        fp = horner(coeffs_p, x)
        if abs(fp) < epsilon:
            return None

        y = x - fx / fp
        fy = horner(coeffs, y)

        # Calcul pas N⁴
        numerator_N4 = fx ** 2 + fy ** 2
        denominator_N4 = fp * (fx - fy)
        if abs(denominator_N4) < epsilon:
            return None
        z = x - numerator_N4 / denominator_N4

        # Pas suplimentar N⁵
        fz = horner(coeffs, z)
        x_new = z - fz / fp

        if abs(x_new - x) < epsilon:
            return x_new
        x = x_new
    return None


# 5. Generare puncte de start în interval
start_points = [-R + i * (2 * R) / (num_start_points - 1) for i in range(num_start_points)]
print(f"3. Puncte de start generate (n={num_start_points}):")
print("   ", [round(x, 2) for x in start_points], "\n")


# 6. Căutare rădăcini distincte cu toate metodele
def find_roots(method):
    roots = []
    for x0 in start_points:
        root = method(x0)
        if root is not None:
            # Verificare unicitate cu valori nerotunjite
            is_unique = all(abs(root - r) > epsilon for r in roots)
            if is_unique:
                roots.append(root)
    return roots


# Aplicăm toate metodele
halley_roots = find_roots(halley)
n4_roots = find_roots(method_N4)
n5_roots = find_roots(method_N5)

# Combinăm și eliminăm duplicate între metode
all_roots = []
for r in halley_roots + n4_roots + n5_roots:
    if all(abs(r - existing) > epsilon for existing in all_roots):
        all_roots.append(r)

# Rotunjire finală pentru afișare
final_roots = [round(root, int(-math.log10(epsilon)) + 1) for root in all_roots]

# 7. Afișare rezultate finale
print("\n5. Rezultate finale (Halley, N⁴, N⁵):")
print("   Rădăcini reale distincte:", sorted(final_roots))