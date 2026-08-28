def gradient_descent_quadratic(a: float, b: float, c: float, x0: float, lr: float, steps: int) -> float:
    """
    Returns the final scalar x after the requested iterations.
    """
    x_star = x0 
    for _ in range(steps):
        derivative = 2*a*x_star + b
        x_star = x_star - lr*derivative
    return x_star