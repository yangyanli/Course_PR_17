"""
A set of misc. methods in pure python to perform generic operations over datasets,
vectors, strings, etc. I have added a lot of stuff in it, and a little is used as of now.
You are free to take and use anything you like.
"""
from __future__ import division
import sys
import random
from numpy.linalg import inv, det
import math

sys.dont_write_bytecode = True


# addition of list items using recursion for both flat and not-flat lists
def add_rec(x):
    y = 0
    for a in x:
        if type(a) == type([]):
            y = y + add_rec(a)
        else:
            y = y + a
    return y


# list flattening
def flat_x(x, y):
    for a in x:
        if type(a) == type([]):
            flat_x(a, y)
        else:
            y.append(a)
    return y


# fibbonaci sum
def fib_x(n):
    if n == 0 or n == 1:
        return 1
    else:
        return fib_x(n - 1) + fib_x(n - 2)


# count down using recursion
def count_x(n):
    if n == 0:
        print
        'blast'
    else:
        print
        n
        n -= 1
        count_x(n)


# factorial rec
def fact_x(n):
    if n == 0:
        return 1
    else:
        return n * fact_x(n - 1)


# digit sum using recursion
def digit_sum(n):
    if n < 10:
        return n
    else:
        return n % 10 + digit_sum(n // 10)


# sort regular using insertionsort
def sort_x(x):
    for a in range(1, len(x)):
        current = x[a]
        position = a

        while position > 0 and x[position - 1] > current:
            x[position] = x[position - 1]
            position = position - 1

        x[position] = current

    return x


# reverse sorting using insertionsort
def rev_sort(x):
    for a in range(1, len(x)):
        current = x[a]
        position = a

        while position > 0 and x[position - 1] < current:
            x[position] = x[position - 1]
            position = position - 1

        x[position] = current

    return x


# missing number in array
def missing_array(x):
    sx = 0
    min_ = 0
    max_ = 0
    for a in range(0, len(x)):
        sx = sx + x[a]

        if x[a] > max_:
            max_ = x[a]
        if x[a] < min_:
            min_ = x[a]
    rsx = (max_ + min_) * (max_ - min_ + 1) / 2
    return rsx - sx


# find 2 missing numbers in an array
def missing_array_2(x):
    sx = 0
    min_ = 1
    max_ = 1

    for a in x:
        sx += a
        if a > max_:
            max_ = a
        if a < min_:
            min_ = a

    rsx = (max_ + min_) * (max_ - min_ + 1) / 2
    z = (rsx - sx) / 2
    rsx1 = (z + 0) * (z - 0 + 1) / 2
    rsx2 = (max_ + (z + 1)) * (max_ - (z + 1) + 1) / 2
    sx1 = 0
    sx2 = 0
    for a in x:
        if a < z:
            sx1 += a
        if a > z:
            sx2 += a

    first_ = rsx1 - sx1
    second_ = rsx2 - sx2

    return first_, second_


# max subset in an array
def max_sub(x):
    c = 0
    ci = 0
    b = 0
    bi = 0
    si = 0

    for a in range(0, len(x)):
        if c + x[a] > 0:
            c = c + x[a]
        else:
            c = 0
            ci = a + 1

        if c > b:
            si = ci
            bi = a + a
            b = c

    return b, x[si:bi]


# cumulative sum array
def cum_sum(x):
    y = range(0, len(x))
    for a in range(0, len(x)):
        if a == 0:
            y[a] = x[a]
        else:
            current = x[a]
            y[a] = current + y[a - 1]

    return y


# finds max min in an array
def max_min(x):
    min_ = 1
    max_ = 1
    for a in x:
        if a > max_:
            max_ = a
        if a < min_:
            min_ = a
    return max_, min_


# downhill simplex algorithm for optimal function minimization
def fmin(F, xStart, side=0.1, tol=0.000006):
    import numpy
    n = len(xStart)  # Number of variables
    x = numpy.zeros((n + 1, n))
    f = numpy.zeros(n + 1)
    # Generate starting simplex
    x[0] = xStart
    for i in range(1, n + 1):
        x[i] = xStart
        x[i, i - 1] = xStart[i - 1] + side
        # Compute values of F at the vertices of the simplex
    for i in range(n + 1): f[i] = F(x[i])
    # Main loop
    for k in range(500):
        # Find highest and lowest vertices
        iLo = numpy.argmin(f)
        iHi = numpy.argmax(f)
        # Compute the move vector d
        d = (-(n + 1) * x[iHi] + numpy.sum(x, axis=0)) / n
        # Check for convergence
        if math.sqrt(numpy.dot(d, d) / n) < tol: return x[iLo]
        # Try reflection
        xNew = x[iHi] + 2.0 * d
        fNew = F(xNew)
        if fNew <= f[iLo]:  # Accept reflection
            x[iHi] = xNew
            f[iHi] = fNew
            # Try expanding the reflection
            xNew = x[iHi] + d
            fNew = F(xNew)
            if fNew <= f[iLo]:  # Accept expansion
                x[iHi] = xNew
                f[iHi] = fNew
        else:
            # Try reflection again
            if fNew <= f[iHi]:  # Accept reflection
                x[iHi] = xNew
                f[iHi] = fNew
            else:
                # Try contraction
                xNew = x[iHi] + 0.5 * d
                fNew = F(xNew)
                if fNew <= f[iHi]:  # Accept contraction
                    x[iHi] = xNew
                    f[iHi] = fNew
                else:
                    # Use shrinkage
                    for i in range(len(x)):
                        if i != iLo:
                            x[i] = (x[i] - x[iLo]) * 0.5
                            f[i] = F(x[i])
    return x[iLo]


# set of functions reserved for normalization procedures
class Functions(object):
    def __init__(self):
        return

    # flatten an array
    def flatten(self, x, out_):
        for a in x:
            if type(a) == type([]):
                out_ = self.flatten(a, out_)
            else:
                out_.append(a)
        return out_

    # mean of an array
    def mean_(self, x):
        return float(sum(x)) / len(x)

    # standard deviation of an array
    def std_(self, x):
        m_x = self.mean_(x)
        return math.sqrt(sum([math.pow((a - m_x), 2) for a in x]) / len(x))

    # mean of a n x m array with axis defined (1 being row wise mean, 0 being column wise,
    # and no axis definition means for the flattened array/matrix
    def mean_nm(self, x, axis=False):
        if axis == 0:
            return map(self.mean_, zip(*x))
        elif axis == 1:
            return map(self.mean_, x)
        elif not axis:
            return self.mean_(self.flatten(x, []))
        else:
            return False

    # standard deviation of a n x m array with axis defined (1 being row wise, 0 being column wise,
    # and no axis definition means for the flattened array/matrix
    def std_nm(self, x, axis=False):
        if axis == 0:
            return map(self.std_, zip(*x))
        elif axis == 1:
            return map(self.std_, x)
        elif not axis:
            return self.std_(self.flatten(x, []))
        else:
            return False

    def normalize_1d(self, x):
        k = []
        x_mean = self.mean_(x)
        for a in x:
            k.append(a - x_mean)
        l = []
        x_std = self.std_(x)
        for a in k:
            l.append(float(a) / x_std)

        return l

    def normalize_(self, x):
        # data -= data.mean(axis=0)
        k = []
        x_mean = self.mean_nm(x, axis=0)
        for a in range(0, len(x)):
            l = []
            for b in range(0, len(x[a])):
                l.append(float(x[a][b]) - float(x_mean[b]))
            k.append(l)

        # data /= data.std(axis=0)
        l_ = []
        x_std = self.std_nm(k, axis=0)
        for a in range(0, len(k)):
            g = []
            for b in range(0, len(k[a])):
                g.append(float(float(k[a][b]) / float(x_std[b])))
            l_.append(g)

        return l_

    # inv
    def inv_(self, x):
        return map(list, inv(x))

    # det
    def det_(self, x):
        return det(x)

    # matrix multiplication (not a generic one)
    def mult_mat(self, a, b):
        try:
            zip_b = zip(*b)
            return [[sum(ele_a * ele_b for ele_a, ele_b in zip(row_a, col_b)) for col_b in zip_b] for row_a in a]
        except:
            j_ = []
            for n_ in a:
                i = []
                for m_, c in zip(n_, b):
                    k = m_ * c
                    i.append(k)
                j_.append(sum(i))
            return j_

    # argmin implementation
    def arg_min(self, x):
        min_ = x[0]
        min_index = 0
        for i in range(0, len(x)):
            if x[i] < min_:
                min_ = x[i]
                min_index = i
        return min_index

    # argmax implementation
    def arg_max(self, x):
        max_ = x[0]
        max_index = 0
        for i in range(0, len(x)):
            if x[i] > max_:
                max_ = x[i]
                max_index = i
        return max_index

    # verifies that each item in the array has equal lengths
    def verify_dimensions(self, x):

        random_pick = len(x[random.randint(0, len(x) - 1)])

        if sum(map(len, x)) / len(x) != random_pick:
            return False
        else:
            return True

    # def scalar product with a vector
    def scalar_product(self, vec, q):
        for i, a in enumerate(vec):
            vec[i] = vec[i] * q
        return vec

    # multiplies a column vector x with matrix y
    def prod(self, x, y):
        result = []
        for a in y:
            l = []
            for i, b in enumerate(x):
                l.append(a[i] * b)
            result.append(sum(l))
        return result

    # dot product of two vectors
    def dot(self, x, y):
        return sum(a * b for a, b in zip(x, y))

    # multiplies a vector with a matrix (another case specific version)
    def prod_2(self, ui, cvg):
        r = []
        for i, a in enumerate(ui):
            r.append(self.dot(ui, cvg[i]))
        return r

    # scalar produvt with a vector
    def sclr_prod_(self, scalar, vector):
        result = []
        for i, a in enumerate(vector):
            result.append(vector[i] * scalar)
        return result