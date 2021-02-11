"""
In this assignment you should fit a model function of your choice to data 
that you sample from a given function. 

The sampled data is very noisy so you should minimize the mean least squares 
between the model you fit and the data points you sample.  

During the testing of this assignment running time will be constrained. You
receive the maximal running time as an argument for the fitting method. You 
must make sure that the fitting function returns at most 5 seconds after the 
allowed running time elapses. If you take an iterative approach and know that 
your iterations may take more than 1-2 seconds break out of any optimization 
loops you have ahead of time.

Note: You are NOT allowed to use any numeric optimization libraries and tools 
for solving this assignment. 

"""

import numpy as np
import time
import torch
import random
import functools
import poly_funcs


def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt(((x1 - x2) ** 2) + ((y1 - y2) ** 2))

def converted_to_np_float32(l):
    return [np.float32(item) for item in l]

class Assignment4A:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """
        self.Ms = [self.build_M(d) for d in range(13)]

    def fit(self, f: callable, a: float, b: float, d:int, maxtime: float) -> callable:
        """
        Build a function that accurately fits the noisy data points sampled from
        some closed shape. 
        
        Parameters
        ----------
        f : callable. 
            A function which returns an approximate (noisy) Y value given X. 
        a: float
            Start of the fitting range
        b: float
            End of the fitting range
        d: int 
            The expected degree of a polynomial matching f
        maxtime : float
            This function returns after at most maxtime seconds. 

        Returns
        -------
        a function:float->float that fits f between a and b
        """
        start_time = time.time()
        n = 1000
        M = self.Ms[d] if d < len(self.Ms) else self.build_M(d)

        fit_func = self.fit_core(f, a, b, n, d, M)
        time_took = time.time() - start_time

        time_left = maxtime - time_took - 0.01
        # the estimated amount of times we can calculate 5000 points in the time left while giving some buffer.
        n = (time_left / (time_took * 2 + 0.005)) * 5000
        n = int(min((b - a) * 5000, n))
        if n >= 2000:
            time_left = time.time() - start_time
            fit_func = self.fit_core(f, a, b, n, d, M)
        return fit_func

    def fit_core(self, f: callable, a: float, b: float, n: int, d: int, M):
        #x_samples = np.concatenate([[a], (np.random.random(n - 2) * (b - a)) + a, [b]])
        #x_samples.sort()
        x_samples = np.linspace(a, b, n)
        y_samples = f(x_samples)

        dis = [0] * n
        def sum_d(acc, i):
            d = acc + euclidean_distance(x_samples[i], y_samples[i], x_samples[i - 1], y_samples[i - 1])
            dis[i] = d
            return d
        d_total = functools.reduce(sum_d, range(1, n), 0)

        t_dis = torch.Tensor([dis[i] / d_total for i in range(n)])
        T = torch.stack([t_dis ** k for k in range(d, -1, -1)]).T
        P = torch.stack([torch.Tensor(x_samples), torch.Tensor(y_samples)]).T
        C = M.inverse().mm((T.T.mm(T)).inverse()).mm(T.T).mm(P)
        bezier_curve = M.mm(C).T
        # print(f"T: {T.size()}")
        # print(f"P: {P.size()}")
        # print(f"C: {C.size()}")
        # print(f"Bezier: {bezier_curve.size()}")
        #bezier_curve = (T.T.mm(T)).inverse().mm(T.T).mm(P).T
        bezier_x_poly = np.poly1d(converted_to_np_float32(bezier_curve[0]))
        bezier_y_poly = np.poly1d(converted_to_np_float32(bezier_curve[1]))

        def find_t(x):
            if x == a:
                t = 0
                #print(f"matched x={x} with t={t}")
            elif x == b:
                t = 1
                #print(f"matched x={x} with t={t}")
            else:
                t = find_t_in_interval(x)
            return t
        def find_t_in_interval(x):
            ts = [root for root in (bezier_x_poly - x).roots if root.imag == 0 and 0.35 < root.real < 1.2]
            if len(ts) == 0:
                raise Exception(f"No t_dis matching x={x}")
            elif len(ts) == 1:
                t = ts[0]
                #print(f"matched x={x} with t={t}")
            else:
                estimated_t = (x - a) / (b - a)
                t = min(ts, key=lambda x: abs(x - estimated_t))
                #print(f"{estimated_t}, matched x={x} with {ts}, took {t}")
            return t
        def fit_func(x):
            return bezier_y_poly(find_t(x))

        return fit_func

    def build_M(self, d: int):
        # took code from stackoverflow on how to efficiently calculate a pascal's triangle row
        #see https://stackoverflow.com/questions/15580291/how-to-efficiently-calculate-a-row-in-pascals-triangle
        m = d + 1
        M = torch.zeros([m, m], dtype=torch.float32)

        d_choose_i = 1
        for control_point_index in range(m):
            one_minus_t_power = d - control_point_index
            power_choose_t = 1 if (d + control_point_index) % 2 == 0 else -1
            for choose_1_amount in range(one_minus_t_power + 1):
                M[choose_1_amount][control_point_index] = d_choose_i * power_choose_t
                power_choose_t *= -(one_minus_t_power - choose_1_amount) / (choose_1_amount + 1)
            d_choose_i *= (d - control_point_index) / (control_point_index + 1)
        return M

##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment4(unittest.TestCase):

    def test_return(self):
        f = NOISY(0.01)(poly(1,1,1))
        ass4 = Assignment4A()
        T = time.time()
        shape = ass4.fit(f=f, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        self.assertLessEqual(T, 5)

    def test_delay(self):
        f = DELAYED(7)(NOISY(0.01)(poly(1,1,1)))

        ass4 = Assignment4A()
        T = time.time()
        shape = ass4.fit(f=f, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        self.assertGreaterEqual(T, 5)

    def test_err(self):
        f = poly(1,1,1)
        nf = NOISY(1)(f)
        ass4 = Assignment4A()
        T = time.time()
        ff = ass4.fit(f=nf, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        mse=0
        for x in np.linspace(0,1,1000):
            self.assertNotEquals(f(x), nf(x))
            mse+= (f(x)-ff(x))**2
        mse = mse/1000
        print(mse)

    def test_err_constant(self):
        f = poly(5)
        nf = NOISY(1)(f)
        self.mse_poly(f, nf, 0, 1)

    def test_err_2(self):
        f = poly(1, 1, 1)
        nf = NOISY(1)(f)
        self.mse_poly(f, nf, -6, 6)

    def test_err_linear_line(self):
        f = poly(5, 0.3)
        nf = NOISY(1)(f)
        self.mse_poly(f, nf, 0, 1)

    def test_err_cubic(self):
        f = poly(*np.random.random(4))
        nf = NOISY(1)(f)
        self.mse_poly(f, nf, 0, 1)

    def test_err_poly_4(self):
        f = poly(*np.random.random(5))
        nf = NOISY(1)(f)
        self.mse_poly(f, nf, 0, 1)

    def test_err_poly_15(self):
        f = poly(*np.random.random(16))
        nf = NOISY(1)(f)
        self.mse_poly(f, nf, 0, 1)

    def mse_poly(self, f, nf, a, b, maxtime=5):
        ass4 = Assignment4A()
        T = time.time()
        ff = ass4.fit(f=nf, a=a, b=b, d=max(10, f.order), maxtime=maxtime)
        T = time.time() - T
        mse=0
        for x in np.linspace(0,1,1000):
            self.assertNotEqual(f(x), nf(x))
            mse+= (f(x)-ff(x))**2
        mse = mse/1000
        print(f"[{a}, {b}], f: {poly_funcs.poly_to_string(f)}")
        print(f"time: {T}")
        print(f"mse: {mse}")
        print("")


if __name__ == "__main__":
    unittest.main()
