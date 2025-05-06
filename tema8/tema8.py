import numpy as np


# =============================================================================
# Utilitare generale (gradient descendent + backtracking)
# =============================================================================
def gradient_descent(F, grad_func, x0, eta, epsilon=1e-5, kmax=30000, backtracking=False, beta=0.8):
    x = x0.copy()
    history = []
    for k in range(kmax):
        g = grad_func(x)

        # Backtracking line search
        if backtracking:
            eta_current = 1.0
            p = 0
            while F(x - eta_current * g) > F(x) - (eta_current / 2) * np.linalg.norm(g) ** 2 and p < 8:
                eta_current *= beta
                p += 1

        eta_used = eta_current if backtracking else eta
        x_new = x - eta_used * g

        history.append({
            'iteration': k + 1,
            'x': x_new,
            'grad_norm': np.linalg.norm(g),
            'eta': eta_used
        })

        if np.linalg.norm(x_new - x) < epsilon:
            break
        x = x_new
    return x, k + 1, history


def approx_gradient(F, x, h=1e-5):
    n = len(x)
    grad = np.zeros(n)
    for i in range(n):
        x_plus2h = x.copy();
        x_plus2h[i] += 2 * h
        x_plush = x.copy();
        x_plush[i] += h
        x_minush = x.copy();
        x_minush[i] -= h
        x_minus2h = x.copy();
        x_minus2h[i] -= 2 * h
        grad[i] = (-F(x_plus2h) + 8 * F(x_plush) - 8 * F(x_minush) + F(x_minus2h)) / (12 * h)
    return grad


# =============================================================================
# Exemplul 1: F(x1, x2) = x1² + x2² - 2x1 - 4x2 - 1
# =============================================================================
print("\n" + "=" * 60 + "\nExemplul 1\n" + "=" * 60)

F1 = lambda x: x[0] ** 2 + x[1] ** 2 - 2 * x[0] - 4 * x[1] - 1
grad_analytic1 = lambda x: np.array([2 * x[0] - 2, 2 * x[1] - 4])
grad_approx1 = lambda x: approx_gradient(F1, x)

x0 = np.array([0.0, 0.0])

# Rulare cu gradient aproximativ (η=0.1)
x_min_ap1, iter_ap1, _ = gradient_descent(F1, grad_approx1, x0, eta=0.1)
print(f"Aproximativ: Minim la {x_min_ap1} în {iter_ap1} iterații (η=0.1)")

# Rulare cu backtracking (β=0.8)
x_min_bt1, iter_bt1, _ = gradient_descent(F1, grad_analytic1, x0, eta=1.0, backtracking=True)
print(f"Backtracking: Minim la {x_min_bt1} în {iter_bt1} iterații")

# =============================================================================
# Exemplul 2: F(x1, x2) = 3x1² - 12x1 + 2x2² + 16x2 - 10
# =============================================================================
print("\n" + "=" * 60 + "\nExemplul 2\n" + "=" * 60)

F2 = lambda x: 3 * x[0] ** 2 - 12 * x[0] + 2 * x[1] ** 2 + 16 * x[1] - 10
grad_analytic2 = lambda x: np.array([6 * x[0] - 12, 4 * x[1] + 16])
grad_approx2 = lambda x: approx_gradient(F2, x)

x0 = np.array([0.0, 0.0])

# Rulare cu gradient aproximativ (η=0.05)
x_min_ap2, iter_ap2, _ = gradient_descent(F2, grad_approx2, x0, eta=0.05)
print(f"Aproximativ: Minim la {x_min_ap2} în {iter_ap2} iterații (η=0.05)")

# Rulare cu backtracking (β=0.8)
x_min_bt2, iter_bt2, _ = gradient_descent(F2, grad_analytic2, x0, eta=1.0, backtracking=True)
print(f"Backtracking: Minim la {x_min_bt2} în {iter_bt2} iterații")

# =============================================================================
# Exemplul 3: F(x1, x2) = x1² - 4x1x2 + 5x2² - 4x2 + 3
# =============================================================================
print("\n" + "=" * 60 + "\nExemplul 3\n" + "=" * 60)

F3 = lambda x: x[0] ** 2 - 4 * x[0] * x[1] + 5 * x[1] ** 2 - 4 * x[1] + 3
grad_analytic3 = lambda x: np.array([2 * x[0] - 4 * x[1], -4 * x[0] + 10 * x[1] - 4])
grad_approx3 = lambda x: approx_gradient(F3, x)

x0 = np.array([0.0, 0.0])

# Rulare cu gradient aproximativ (η=0.1)
x_min_ap3, iter_ap3, _ = gradient_descent(F3, grad_approx3, x0, eta=0.1)
print(f"Aproximativ: Minim la {x_min_ap3} în {iter_ap3} iterații (η=0.1)")

# Rulare cu backtracking (β=0.8)
x_min_bt3, iter_bt3, _ = gradient_descent(F3, grad_analytic3, x0, eta=1.0, backtracking=True)
print(f"Backtracking: Minim la {x_min_bt3} în {iter_bt3} iterații")

# =============================================================================
# Exemplul 4: F(x1, x2) = x1²x2 - 2x1x2² + 3x1x2 + 4
# =============================================================================
print("\n" + "=" * 60 + "\nExemplul 4\n" + "=" * 60)

F4 = lambda x: x[0] ** 2 * x[1] - 2 * x[0] * x[1] ** 2 + 3 * x[0] * x[1] + 4
grad_analytic4 = lambda x: np.array([
    2 * x[0] * x[1] - 2 * x[1] ** 2 + 3 * x[1],
    x[0] ** 2 - 4 * x[0] * x[1] + 3 * x[0]
])
grad_approx4 = lambda x: approx_gradient(F4, x)

x0 = np.array([0.5, 1.0])

# Rulare cu gradient aproximativ (η=0.01)
x_min_ap4, iter_ap4, _ = gradient_descent(F4, grad_approx4, x0, eta=0.01)
print(f"Aproximativ: Minim la {x_min_ap4} în {iter_ap4} iterații (η=0.01)")

# Rulare cu backtracking (β=0.8)
x_min_bt4, iter_bt4, _ = gradient_descent(F4, grad_analytic4, x0, eta=1.0, backtracking=True)
print(f"Backtracking: Minim la {x_min_bt4} în {iter_bt4} iterații")