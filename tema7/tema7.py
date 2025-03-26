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
            return round(x_new, int(-math.log10(epsilon)) + 1)
        x = x_new

    return None  # Divergență


# 5. Generare puncte de start în interval
start_points = [-R + i * (2 * R) / (num_start_points - 1) for i in range(num_start_points)]
print(f"3. Puncte de start generate (n={num_start_points}):")
print("   ", [round(x, 2) for x in start_points], "\n")

# 6. Căutare rădăcini distincte
roots = []
for x0 in start_points:
    root = halley(x0)
    if root is not None:
        # Verificare unicitate
        is_unique = all(abs(root - r) > epsilon for r in roots)
        if is_unique:
            roots.append(root)
            print(f"4. Punct start {x0:.2f} → Rădăcină: {root:.6f}")

# 7. Afișare rezultate finale
print("\n5. Rezultate finale:")
print("   Rădăcini reale distincte:", sorted(roots))