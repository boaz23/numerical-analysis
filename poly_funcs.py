import numpy as np
import non_linear_equations

# noinspection PyTypeChecker,PyUnresolvedReferences,PyArgumentList
def find_roots(p: np.poly1d, start, end, maxerr=0.001):
    global base_poly_roots_finders
    coefs = p.coefficients
    d = p.order
    if d < len(base_poly_roots_finders):
        return base_poly_roots_finders[d](start, end, *coefs)
    dp_roots = find_roots(np.polyder(p), start, end, maxerr)
    return non_linear_equations.find_roots_from_der_roots(p, start, end, dp_roots, maxerr)

def _find_roots_poly_constant(start, end, constant):
    if constant == 0:
        raise Exception("Zero polynomial has infinite amount roots")
    return []
def _find_roots_poly_linear_line(start, end, slope, constant):
    if slope == 0:
        return _find_roots_poly_constant(start, end, constant)
    root = (-constant) / slope
    return _roots_list(start, end, root)
def _find_roots_poly_quadratic(start, end, a, b, c):
    if a == 0:
        return _find_roots_poly_linear_line(start, end, b ,c)
    return _find_roots_poly_quadratic_core(a, b, c, end, start)

def _find_roots_poly_quadratic_core(a, b, c, end, start):
    delta = (b ** 2) - 4 * a * c
    if delta < 0:
        return []
    elif delta == 0:
        root = -b / (2 * a)
        return _roots_list(start, end, root)
    else:
        return _find_roots_poly_quadratic_2_roots(a, b, delta, end, start)

def _find_roots_poly_quadratic_2_roots(a, b, delta, end, start):
    delta = np.sqrt(delta)
    root0 = (-b + delta) / (2 * a)
    root1 = (-b - delta) / (2 * a)
    root0, root1 = min(root0, root1), max(root0, root1)
    return _roots_list(end, start, root0, root1)


base_poly_roots_finders = [
    _find_roots_poly_constant,
    _find_roots_poly_linear_line,
    _find_roots_poly_quadratic
]


def _roots_list(end, start, *roots):
    return [root for root in roots if start <= root <= end]


def poly_to_string(p):
    coefficients = p.coefficients
    highest = coefficients[0]
    n = p.order

    s = ""
    if highest == -1:
        s += "-"
    elif highest == 1:
        pass
    else:
        s += str(highest)
    if n >= 10:
        s += f"x^{{{n}}}"
    elif n > 1:
        s += f"x^{n}"
    elif n == 1:
        s += "x"
    for k in range(1, n - 1):
        coef_string = coef_to_string(coefficients[k])
        if coef_string == "":
            continue
        s += f"{coef_string}x^{n-k}"

    if n > 1:
        coef_string = coef_to_string(coefficients[n - 1])
        if coef_string != "":
            s += f"{coef_string}x"
    if n > 0:
        coef = coefficients[n]
        s += f" {'-' if coef < 0 else '+'} {abs(coef)}"
    return s


def coef_to_string(coef):
    s = ""
    if coef != 0:
        s += f" {'-' if coef < 0 else '+'} "
        if abs(coef) != 1:
            s += str(abs(coef))
    return s

if __name__ == "__main__":
    #print(find_roots(np.poly1d([1, 0, -1, 0]), -10, 10))
    #print(find_roots(np.poly1d(np.random.random(11)), -10, 10))
    pass