"""
Test pychebfun

"""

import sys, os

sys.path.insert(0,os.path.abspath('..'))

from pychebfun import *
import numpy as np
import scipy as sp
from scipy.integrate import quad
from matplotlib import pyplot as plt

from scipy.special import j0, jn_zeros, jn

def test_explicit_cos():
    """ Test whether we can create a function
    explicitly """
    print "Constructing explicitly 'cos(20*x)' "
    f = Chebfun('cos(20*x)')
    xs = np.random.uniform(-1,1,1000)
    print "Testing at random points "
    np.testing.assert_almost_equal(cos(20*xs),f(xs))

def test_implied_cos():
    """ See whether we can make implied functions """
    print "Testing implicit construction"
    f = cos(20*x)
    xs = np.random.uniform(-1,1,1000)
    print "testing at random points"
    np.testing.assert_almost_equal(cos(20*xs),f(xs))

def test_lambda():
    """ Test construction with a callable """
    print "Testing creation with anonymous function"
    f = Chebfun(lambda x: cos(20*x))
    xs = np.random.uniform(-1,1,1000)
    print "Testing at random points "
    np.testing.assert_almost_equal(cos(20*xs),f(xs))

def test_cos_quad():
    """ Test quadrature """
    print "Testing quad for cos(20*x) "
    f = cos(20*x)
    cheb_quad = f.quad()
    # traditional_quad,err = quad(lambda x: cos(20*x), -1,1)
    analytical_quad = sin(20.)/10.
    print "Cheb quad: ", cheb_quad
    print "analytical: ", analytical_quad
    np.testing.assert_almost_equal(cheb_quad,analytical_quad)

def test_bessel_zeros():
    """ Test that we can compute the zeros of the bessel function """
    f = Chebfun(j0,(0,100))
    roots = f.introots()
    real_roots = jn_zeros(0,len(roots))
    np.testing.assert_allclose(roots,real_roots)


def test_composition():
    """ Test chebfun composition """
    xx = Chebfun('x')
    f = lambda x: 1./(1+25*x**2)
    cheb = Chebfun(f)
    xs = np.random.uniform(-1,1,1000)
    np.testing.assert_allclose(cheb(xs),f(xs))

def test_deriv():
    """ Test the derivative functionality """
    # form bessel j1
    f = Chebfun(lambda x: jn(1,x),(0,100))
    anald = lambda x: 0.5*(jn(0,x) - jn(2,x))
    xs = np.random.uniform(0,100,1000)
    np.testing.assert_allclose( f.deriv()(xs), anald(xs) )

def test_integ():
    """ Test the indefinite integral function """
    f = Chebfun(np.tanh)
    analf = lambda x: np.log(np.cosh(x))
    xs = np.random.uniform(-1,1,1000)
    np.testing.assert_allclose(f.integ()(xs), analf(xs))
