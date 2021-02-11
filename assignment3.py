"""
In this assignment you should find the area enclosed between the two given functions.
The rightmost and the leftmost x values for the integration are the rightmost and 
the leftmost intersection points of the two functions. 

The functions for the numeric answers are specified in MOODLE. 


This assignment is more complicated than Assignment1 and Assignment2 because: 
    1. You should work with float32 precision only (in all calculations) and minimize the floating point errors. 
    2. You have the freedom to choose how to calculate the area between the two functions. 
    3. The functions may intersect multiple times. Here is an example: 
        https://www.wolframalpha.com/input/?i=area+between+the+curves+y%3D1-2x%5E2%2Bx%5E3+and+y%3Dx
    4. Some of the functions are hard to integrate accurately. 
       You should explain why in one of the theoretical questions in MOODLE. 

"""

import numpy as np
import functools
import time
import random
import assignment2
import math

def integrate_simpson_composite(f: callable, a: float, b: float, n: int) -> np.float32:
    if n % 2 == 0:
        n -= 1
    a = np.float32(a)
    b = np.float32(b)
    xs = np.linspace(a, b, n, dtype=np.float32)

    def build_point_info(i):
        y = np.float32(f(xs[i]))
        if i % 2 == 1:
            return np.float32(4 * y)
        return np.float32(2 * y)
    ys = [np.float32(f(a))]
    ys = ys + [build_point_info(i) for i in range(1, n - 1)]
    ys = ys + [np.float32(f(b))]
    ys.sort()
    def sum_ys(acc, y):
        return np.float32(acc + y)
    s = functools.reduce(sum_ys, ys, np.float32(0))
    h = np.float32(np.float32(b - a) / np.float32(n - 1))
    return np.float32(np.float32(h / 3) * s)

class Assignment3:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """

        pass

    def integrate(self, f: callable, a: float, b: float, n: int) -> np.float32:
        """
        Integrate the function f in the closed range [a,b] using at most n 
        points. Your main objective is minimizing the integration error. 
        Your secondary objective is minimizing the running time. The assignment
        will be tested on variety of different functions. 
        
        Integration error will be measured compared to the actual value of the 
        definite integral. 
        
        Note: It is forbidden to call f more than n times. 
        
        Parameters
        ----------
        f : callable. it is the given function
        a : float
            beginning of the integration range.
        b : float
            end of the integration range.
        n : int
            maximal number of points to use.

        Returns
        -------
        np.float32
            The definite integral of f between a and b
        """

        # replace this line with your solution
        return integrate_simpson_composite(f, a, b, n)

    def areabetween(self, f1: callable, f2: callable) -> np.float32:
        """
        Finds the area enclosed between two functions. This method finds 
        all intersection points between the two functions to work correctly. 
        
        Example: https://www.wolframalpha.com/input/?i=area+between+the+curves+y%3D1-2x%5E2%2Bx%5E3+and+y%3Dx

        Note, there is no such thing as negative area. 
        
        In order to find the enclosed area the given functions must intersect 
        in at least two points. If the functions do not intersect or intersect 
        in less than two points this function returns NaN.  
        This function may not work correctly if there is infinite number of 
        intersection points. 
        

        Parameters
        ----------
        f1,f2 : callable. These are the given functions

        Returns
        -------
        np.float32
            The area between function and the X axis

        """

        # replace this line with your solution
        intersection_xs = assignment2.Assignment2().intersections(f1, f2, 1, 100)
        intersections_amount = len(intersection_xs)
        if intersections_amount < 2:
            return np.float32(np.nan)

        f = lambda x: f1(x) - f2(x)
        s = np.float32(0)
        for i in range(intersections_amount - 1):
            x0 = intersection_xs[i]
            x2 = intersection_xs[i + 1]
            n = math.ceil((x2 - x0) / 2)
            n = min(5, 5 * n)
            s = np.float32(s + abs(self.integrate(f, x0, x2, n)))
        return s


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment3(unittest.TestCase):

    def test_integrate_float32(self):
        ass3 = Assignment3()
        f1 = np.poly1d([-1, 0, 1])
        r = ass3.integrate(f1, -1, 1, 10)

        self.assertEquals(r.dtype, np.float32)

    def test_integrate_hard_case(self):
        ass3 = Assignment3()
        f1 = strong_oscilations()
        r = ass3.integrate(f1, 0.09, 10, 20)
        true_result = -7.78662 * 10 ** 33
        self.assertGreaterEqual(0.001, abs((r - true_result) / true_result))

    def test_integrate_float32_2(self):
        ass3 = Assignment3()
        f1 = np.poly1d([1, 0, -1])
        r = ass3.integrate(f1, -1, 1, 10)
        self.assertEqual(r.dtype, np.float32)
        self.assertGreaterEqual(5e-8, self.relative_error(-4 / 3, r))

    def test_integrate_float32_3(self):
        ass3 = Assignment3()
        f1 = np.poly1d([1, 0, 0])
        r = ass3.integrate(f1, -4, 4, 10)
        self.assertEqual(r.dtype, np.float32)
        self.assertGreaterEqual(5e-8, self.relative_error(128 / 3, r))

    def test_integrate_poly_5_1(self):
        ass3 = Assignment3()
        f1 = np.poly1d([0.63, 0.7, 0.21, 0.86, 0.12, 0.48])
        r = ass3.integrate(f1, -1, 4, 10)
        self.assertEqual(r.dtype, np.float32)

    def test_integrate_float32_sin_1(self):
        ass3 = Assignment3()
        f1 = np.sin
        r = ass3.integrate(f1, -np.pi, np.pi, 10)
        self.assertEqual(r.dtype, np.float32)
        self.assertEqual(0, r)

    def test_areabetween_sin_1(self):
        ass3 = Assignment3()
        f1 = np.sin
        r = ass3.areabetween(f1, lambda x: 0)
        self.assertEqual(r.dtype, np.float32)

    def test_areabetween_example_given(self):
        ass3 = Assignment3()
        f1 = np.poly1d([1, -2, 0, 1])
        f2 = np.poly1d([1, 0])
        r = ass3.areabetween(f1, f2)
        self.assertEqual(r.dtype, np.float32)

    def relative_error(self, expected, actual):
        return abs((expected - actual) / expected)


if __name__ == "__main__":
    unittest.main()
