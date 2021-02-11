import numpy as np
import math

def is_root(y, maxerr):
    return abs(y) < maxerr

def is_same_x(x1, x2, epsilon):
    return abs(x1 - x2) < epsilon

def find_roots_from_der_roots(f, a, b, d_roots, maxerr):
    roots = []

    prev_a = a
    f_prev_a = f(a)
    if is_root(f_prev_a, maxerr):
        roots.append(prev_a)

    for i in range(len(d_roots)):
        dr = d_roots[i]
        f_dr = f(dr)
        if is_root(f_dr, maxerr):
            roots.append(dr)
        elif (f_prev_a < 0) != (f_dr < 0):
            roots.append(regula_falsi(f, prev_a, dr, maxerr))
        else:
            pass
        prev_a = dr
        f_prev_a = f_dr

    f_b = f(b)
    if is_root(f_b, maxerr):
        roots.append(b)
    elif (f_prev_a < 0) != (f_b < 0):
        roots.append(regula_falsi(f, prev_a, b, maxerr))
    else:
        pass
    return roots


def regula_falsi(f, a, b, maxerr):
    c = b
    fc = fb = f(b)
    while not is_root(fc, maxerr):
        c = b - fb * ((b - a) / (fb - f(a)))
        fc = f(c)
        if (fc < 0) != (fb < 0):
            a = c
        else:
            b = c
            fb = fc
    return c

def find_roots_linear_spacing(f, a, b, maxerr, epsilon):
    precision = max(100, 1 / (maxerr * 10))
    n = int(math.ceil((b - a) * precision)) + 1
    #print(f"precision: {1 / precision}, n: {n}")

    n = max(11, n)
    xs = np.linspace(a, b, n)
    #print(f"a, b = {xs[0]}, {xs[n - 1]}")

    roots = []
    i = 0

    x0 = xs[i]
    f_x0 = f(x0)
    while is_root(f_x0, maxerr):
        roots.append((x0, f_x0))
        i += 1
        if i >= n:
            return roots
        x0 = xs[i]
        f_x0 = f(x0)
    i += 1
    is_x0_root = False
    while i < n:
        x1 = xs[i]
        f_x1 = f(x1)
        if is_root(f_x1, maxerr):
            roots.append((x1, f_x1))
            is_x0_root = True
        elif not is_x0_root and (f_x0 > 0) == (f_x1 < 0):
            root = regula_falsi(f, x0, x1, maxerr)
            roots.append((root, f(root)))
            #print(f"paired {x0} with {x1}, found root: {roots[-1]}")
            is_x0_root = False
        else:
            is_x0_root = False
        x0 = x1
        f_x0 = f_x1
        i += 1
    return roots

def unique_roots_filter(f, roots, maxerr):
    roots_amount = len(roots)
    roots.sort(key=lambda x: x[0])
    i = 0
    unique_roots = []
    while i < roots_amount:
        rprev_x, rprev_y = roots[i]
        min_x = rprev_x
        min_y = rprev_y

        i += 1
        while i < roots_amount:
            rx, ry = roots[i]
            if is_root(f((rprev_x + rx) / 2), maxerr):
                if min_y > ry:
                    min_x = rx
                    min_y = ry
            else:
                break
            rprev_x = rx
            i += 1
        unique_roots.append(min_x)
    return unique_roots

def find_roots_brute_force(f: callable, a: float, b: float, maxerr: float, epsilon: float):
    intervals_stack = [(a, b)]
    roots = []
    while len(intervals_stack) > 0:
        a, b = intervals_stack.pop()
        if a > b:
            continue
        if is_same_x(a, b, epsilon):
            fa = f(a)
            if is_root(fa, maxerr):
                roots.append((a, fa))
        else:
            mid = (a + b) / 2
            intervals_stack.append((mid, b))
            intervals_stack.append((a, mid))
    return roots