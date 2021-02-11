"""
In this assignment you should interpolate the given function.
"""
import functools

import numpy as np
import time
import random

poly_zero = np.poly1d([0])
poly_one = np.poly1d([1])
poly_x = np.poly1d([1, 0])
der_x_epsilon = 1e-8

def lagrange_li(xs: np.ndarray, n: int, i: int):
    global poly_x
    global poly_one
    x = poly_x
    def _li_item(i, j):
        return (x - xs[j]) / (xs[i] - xs[j])
    li_items = [_li_item(i, j) for j in range(n) if j != i]
    li = functools.reduce(np.polymul, li_items, poly_one)
    return li

def hermite_hi(xs: np.ndarray, n: int, i: int):
    global poly_x
    x = poly_x
    xi = xs[i]
    li = lagrange_li(xs, n, i)
    li_der = np.polyder(li)
    return (1 - (2 * (x - xi) * li_der(xi))) * (li ** 2)

def hermite_h_hat_i(xs: np.ndarray, n: int, i: int):
    global poly_x
    x = poly_x
    li = lagrange_li(xs, n, i)
    return (x - xs[i]) * (li ** 2)

def linear_slope(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return (y2 - y1) / (x2 - x1)

def estimate_y_ders_core(f: callable, epsilon: float, n: int, xs: np.ndarray, ys: np.ndarray):
    near_x_xs = np.array([x + epsilon for x in xs])
    near_x_ys = f(near_x_xs)
    return [linear_slope((xs[i], ys[i]), (near_x_xs[i], near_x_ys[i])) for i in range(n)]

def estimate_y_ders(f: callable, epsilon: float, n: int, xs: np.ndarray, ys: np.ndarray):
    y_ders = estimate_y_ders_core(f, epsilon, n - 1, xs[:-1], ys[:-1])
    y_ders = y_ders + estimate_y_ders_core(f, -epsilon, 1, np.array([xs[-1]]), np.array([ys[-1]]))
    return np.array(y_ders)

def weighted_ys_poly(xs: np.ndarray, ys: np.ndarray, n: int, f_weight: callable):
    global poly_zero
    lis = [f_weight(xs, n, i) * ys[i] for i in range(n)]
    p = functools.reduce(np.polyadd, lis, poly_zero)
    return p

def interpolation_lagrange_core(xs: np.ndarray, ys: np.ndarray, n: int):
    return weighted_ys_poly(xs, ys, n, lagrange_li)

def interpolation_lagrange_normal(f: callable, a: float, b: float, n: int):
    xs = np.linspace(a, b, n)
    ys = f(xs)
    return interpolation_lagrange_core(xs, ys, n)

def interpolate_hermite_from_points(xs: np.ndarray, ys: np.ndarray, y_ders: np.ndarray, n: int):
    fh = weighted_ys_poly(xs, ys, n, hermite_hi)
    f_der_h_hats = weighted_ys_poly(xs, y_ders, n, hermite_h_hat_i)
    return fh + f_der_h_hats

def interpolate_piecewise_hermite(epsilon):
    def interpolate(f: callable, a: float, b: float, n: int):
        n = n // 2
        xs = np.linspace(a, b, n)
        ys = f(xs)
        y_ders = estimate_y_ders(f, epsilon, n, xs, ys)
        polies = [interpolate_hermite_from_points(xs[i:i+2], ys[i:i+2], y_ders[i:i+2], 2) for i in range(n - 1)]
        intervals_count = n - 1
        interval_len = (b - a) / intervals_count
        def interpolation_func(x):
            if x < a or x > b:
                raise Exception("x={x} is not in the interpolation range")
            poly_index = int((x - a) / interval_len)
            poly_index = min(intervals_count - 1, poly_index)
            return polies[poly_index](x)
        return interpolation_func
    return interpolate

def interpolate_composite(epsilon):
    interpolate_hermite = interpolate_piecewise_hermite(epsilon)
    def interpolate(f: callable, a: float, b: float, n: int):
        if n < 1:
            raise Exception("Amount of sample points is 0 or less")
        elif n == 1:
            if a == b:
                c = a
            else:
                c = (a + b) / 2
            c = f(c)
            return lambda x: c
        elif n < 6:
            return interpolation_lagrange_normal(f, a, b, n)
        else:
            return interpolate_hermite(f, a, b, n)
    return interpolate

class Assignment1:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        starting to interpolate arbitrary functions.
        """
        global der_x_epsilon
        self._f_interpolate = interpolate_composite(der_x_epsilon)

    def interpolate(self, f: callable, a: float, b: float, n: int) -> callable:
        """
        Interpolate the function f in the closed range [a,b] using at most n 
        points. Your main objective is minimizing the interpolation error.
        Your secondary objective is minimizing the running time. 
        The assignment will be tested on variety of different functions with 
        large n values. 
        
        Interpolation error will be measured as the average absolute error at 
        2*n random points between a and b. See test_with_poly() below. 

        Note: It is forbidden to call f more than n times. 

        Note: This assignment can be solved trivially with running time O(n^2)
        or it can be solved with running time of O(n) with some preprocessing.
        **Accurate O(n) solutions will receive higher grades.** 
        
        Note: sometimes you can get very accurate solutions with only few points, 
        significantly less than n. 
        
        Parameters
        ----------
        f : callable. it is the given function
        a : float
            beginning of the interpolation range.
        b : float
            end of the interpolation range.
        n : int
            maximal number of points to use.

        Returns
        -------
        The interpolating function.
        """

        # replace this line with your solution to pass the second test
        return self._f_interpolate(f, a, b, n)


##########################################################################


import unittest
from functionUtils import *
from tqdm import tqdm


class TestAssignment1(unittest.TestCase):

    def test_with_poly(self):
        T = time.time()

        ass1 = Assignment1()
        mean_err = 0

        d = 30
        for i in tqdm(range(100)):
            a = np.random.randn(d)

            f = np.poly1d(a)

            ff = ass1.interpolate(f, -10, 10, 100)

            xs = np.random.random(200)
            err = 0
            for x in xs:
                yy = ff(x)
                y = f(x)
                err += abs(y - yy)

            err = err / 200
            mean_err += err
        mean_err = mean_err / 100

        T = time.time() - T
        print(T)
        print(mean_err)

    def test_with_poly_restrict(self):
        ass1 = Assignment1()
        a = np.random.randn(5)
        f = RESTRICT_INVOCATIONS(10)(np.poly1d(a))
        ff = ass1.interpolate(f, -10, 10, 10)
        xs = np.random.random(20)
        for x in xs:
            yy = ff(x)

    def test_with_poly_2(self):
        T = time.time()

        ass1 = Assignment1()
        mean_err = 0

        d = 30
        for i in tqdm(range(100)):
            a = np.random.randn(d)

            f = np.poly1d(a)

            ff = ass1.interpolate(f, -10, 10, 50)

            xs = (np.random.random(200))
            err = 0
            for x in xs:
                yy = ff(x)
                y = f(x)
                err += abs((y - yy) / y)

            err = err / 200
            mean_err += err
        mean_err = mean_err / 100

        T = time.time() - T
        print(f"time: {T}, mean relative err: {mean_err}")

if __name__ == "__main__":
    unittest.main()
