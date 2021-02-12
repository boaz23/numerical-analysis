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
    pass