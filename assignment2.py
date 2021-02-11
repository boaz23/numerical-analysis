"""
In this assignment you should find the intersection points for two functions.
"""

import numpy as np
import time
import random
from collections.abc import Iterable
import non_linear_equations


class Assignment2:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """

        pass

    def intersections(self, f1: callable, f2: callable, a: float, b: float, maxerr=0.001) -> Iterable:
        """
        Find as many intersection points as you can. The assignment will be
        tested on functions that have at least two intersection points, one
        with a positive x and one with a negative x.
        
        This function may not work correctly if there is infinite number of
        intersection points. 


        Parameters
        ----------
        f1 : callable
            the first given function
        f2 : callable
            the second given function
        a : float
            beginning of the interpolation range.
        b : float
            end of the interpolation range.
        maxerr : float
            An upper bound on the difference between the
            function values at the approximate intersection points.


        Returns
        -------
        X : iterable of approximate intersection Xs such that for each x in X:
            |f1(x)-f2(x)|<=maxerr.

        """

        # replace this line with your solution
        f = lambda x: f1(x) - f2(x)
        return self.find_roots(
            a,
            b,
            f,
            maxerr,
            non_linear_equations.find_roots_brute_force,
            non_linear_equations.unique_roots_filter
        )

    def find_roots(self, a, b, f, maxerr, f_find_roots, f_post_find):
        epsilon = maxerr / 10
        roots = f_find_roots(f, a, b, maxerr, epsilon)
        if len(roots) == 0:
            return roots
        return f_post_find(f, roots, maxerr)


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm
import poly_funcs

class TestAssignment2(unittest.TestCase):

    def test_sqr(self):

        ass2 = Assignment2()

        f1 = np.poly1d([-1, 0, 1])
        f2 = np.poly1d([1, 0, -1])

        X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)

        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))

    def test_poly(self):

        ass2 = Assignment2()

        f1, f2 = randomIntersectingPolynomials(10)

        X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)

        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))

    def test_sin_x_square(self):
        ass2 = Assignment2()

        f1 = lambda x: np.sin(x ** 2)
        f2 = lambda x: 0
        T = time.time()
        X = ass2.intersections(f1, f2, -100, 100, maxerr=0.001)
        T = time.time() - T

        print(f"T: {T}, {len(X)}, f: sin(x^2), {X}")
        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))


    def test_poly_random_10(self):

        ass2 = Assignment2()

        f1, f2 = randomIntersectingPolynomials(10)

        T = time.time()
        X = ass2.intersections(f1, f2, -1, 3, maxerr=0.001)
        T = time.time() - T

        print(f"T: {T}, f: {poly_funcs.poly_to_string(f1 - f2)}, X: {X}")
        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))


    def test_poly_2(self):
        ass2 = Assignment2()

        f1, f2 = np.poly1d([-1, 0, 1]), lambda x: 0
        T = time.time()
        X = ass2.intersections(f1, f2, -1, 3, maxerr=0.001)
        T = time.time() - T

        print(f"T: {T}, f: {poly_funcs.poly_to_string(f1)}, X: {X}")
        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))

    def test_areabetween_example_given(self):
        ass2 = Assignment2()
        f1 = np.poly1d([1, -2, 0, 1])
        f2 = np.poly1d([1, 0])

        T = time.time()
        X = ass2.intersections(f1, f2, -1, 3, maxerr=0.001)
        T = time.time() - T

        print(f"T: {T}, f: {poly_funcs.poly_to_string(f1 - f2)}, X: {X}")
        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))


if __name__ == "__main__":
    unittest.main()
    #TestAssignment2().test_poly2()
