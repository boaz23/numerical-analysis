def is_root(y, maxerr):
    return abs(y) <= maxerr

def is_same_x(x1, x2, epsilon):
    return abs(x1 - x2) <= epsilon

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