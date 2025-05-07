import numpy as np
import matplotlib.pyplot as plt


def f(x):
    # Example function (can be changed as needed)
    return np.sin(x)


def horner_eval(coeffs, x):
    """Standalone Horner's method implementation"""
    result = 0.0
    for c in reversed(coeffs):
        result = result * x + c
    return result


def generate_nodes(x0, xn, num_points):
    if num_points < 2:
        raise ValueError("Number of points must be at least 2.")
    internal = np.random.uniform(x0, xn, num_points - 2)
    nodes = np.concatenate([[x0], internal, [xn]])
    nodes.sort()
    return nodes


def least_squares_approximation(x_nodes, y_nodes, m, x_hat):
    n = len(x_nodes) - 1
    if m >= 6:
        raise ValueError("m must be less than 6.")

    # Build matrix B and rhs
    B = np.zeros((m + 1, m + 1))
    rhs = np.zeros(m + 1)
    for i in range(m + 1):
        for j in range(m + 1):
            B[i, j] = np.sum(x_nodes ** (i + j))
        rhs[i] = np.sum(y_nodes * (x_nodes ** i))

    # Solve the system
    a = np.linalg.solve(B, rhs)

    pm_xhat = horner_eval(a, x_hat)
    error_pm = np.abs(pm_xhat - f(x_hat))
    sum_abs_diff = np.sum(np.abs(horner_eval(a, x_nodes) - y_nodes))

    return pm_xhat, error_pm, sum_abs_diff, a  # Return coefficients for plotting


def trigonometric_interpolation(x_nodes, y_nodes, m_trig, x_hat):
    n = len(x_nodes) - 1
    if n != 2 * m_trig:
        raise ValueError("Number of nodes must be 2m+1 for trigonometric interpolation.")

    # Build matrix T
    T = np.zeros((2 * m_trig + 1, 2 * m_trig + 1))
    for i, x in enumerate(x_nodes):
        row = [1.0]
        for k in range(1, m_trig + 1):
            row.append(np.sin(k * x))
            row.append(np.cos(k * x))
        T[i, :] = row[:2 * m_trig + 1]

    # Solve the system and return coefficients
    coeffs = np.linalg.solve(T, y_nodes)

    # Evaluate Tn(x_hat)
    tn_xhat = coeffs[0]
    idx = 1
    for k in range(1, m_trig + 1):
        tn_xhat += coeffs[idx] * np.sin(k * x_hat) + coeffs[idx + 1] * np.cos(k * x_hat)
        idx += 2

    error_tn = np.abs(tn_xhat - f(x_hat))
    return tn_xhat, error_tn, coeffs  # Return coefficients for plotting


def main():
    # Least Squares Parameters
    x0_ls = 0.0
    xn_ls = 2 * np.pi
    num_points_ls = 10
    m = 3

    # Trigonometric Parameters
    x0_trig = 0.0
    xn_trig = 2 * np.pi - 1e-5
    m_trig = 2
    x_hat = 1.5

    # Generate nodes and perform approximations
    x_nodes_ls = generate_nodes(x0_ls, xn_ls, num_points_ls)
    y_nodes_ls = f(x_nodes_ls)
    pm_xhat, error_pm, sum_diff, coeffs_pm = least_squares_approximation(x_nodes_ls, y_nodes_ls, m, x_hat)

    num_points_trig = 2 * m_trig + 1
    x_nodes_trig = generate_nodes(x0_trig, xn_trig, num_points_trig)
    y_nodes_trig = f(x_nodes_trig)
    tn_xhat, error_tn, coeffs_trig = trigonometric_interpolation(x_nodes_trig, y_nodes_trig, m_trig, x_hat)

    # Print results
    print(f"Least Squares Approximation Pm({x_hat}) = {pm_xhat}")
    print(f"Error |Pm - f| = {error_pm}")
    print(f"Sum of absolute differences: {sum_diff}")
    print(f"\nTrigonometric Interpolation Tn({x_hat}) = {tn_xhat}")
    print(f"Error |Tn - f| = {error_tn}")

    # Bonus: Plotting
    x_plot = np.linspace(min(x_nodes_ls), max(x_nodes_ls), 400)
    y_true = f(x_plot)

    # Evaluate approximations
    y_pm = [horner_eval(coeffs_pm, x) for x in x_plot]

    y_tn = []
    for x in x_plot:
        val = coeffs_trig[0]
        idx = 1
        for k in range(1, m_trig + 1):
            val += coeffs_trig[idx] * np.sin(k * x) + coeffs_trig[idx + 1] * np.cos(k * x)
            idx += 2
        y_tn.append(val)

    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, y_true, label='True f(x)')
    plt.plot(x_plot, y_pm, '--', label=f'P{m}(x)')
    plt.plot(x_plot, y_tn, '-.', label=f'T{2 * m_trig}(x)')
    plt.scatter(x_nodes_ls, y_nodes_ls, c='red', s=30, label='LS Nodes')
    plt.scatter(x_nodes_trig, y_nodes_trig, c='green', s=30, label='Trig Nodes')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Function Approximation Comparison')
    plt.show()
    plt.savefig('ss.png')



if __name__ == "__main__":
    main()