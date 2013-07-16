#!/usr/bin/env python
from __future__ import division

"""
Chebfun is a work in progress clone of the Matlab Chebfun project"""

__author__ = "Alex Alemi"
__version__ = "0.2"

#My imports
import warnings
import sys
import math
import copy
import operator
from bisect import bisect
import collections

import numpy as np
from numpy import frompyfunc, piecewise
# import scipy as sp
import matplotlib.pyplot as plt

import tools
import logging
logger = logging.getLogger('pychebfun')
logger.setLevel(logging.INFO)

#----------------------
# SETTINGS
#----------------------

NAF_CUTOFF = 128
DEFAULT_TOL = tools.DEFAULT_TOL


def opr(func):
    """ reverse the arguments to a function, decorator
    Used to help create all of the __r versions of operators"""
    def rfunc(*args):
        return func(*reversed(args))
    return rfunc

NP_OVERLOAD = set(("arccos", "arccosh", "arcsin", "arcsinh", "arctan", "arctanh", "cos",
     "sin", "tan", "cosh", "sinh", "tanh", "exp", "exp2", "expm1", "log", "log2", "log1p",
     "sqrt", "ceil", "trunc", "fabs", "floor", ))


class Cheb(object):
    """ A simple cheb object, which represents a function defined on a
    domain with a chebyshev polynomial to within machine precision """

    def __init__(self,obj,domain=None,N=None,rtol=None):
        """ Initilize the cheb

                obj can be one of
                    * a python callable
                    * a numpy ufunc
                    * a string (using the numpy namespace)
                    * a ndarray to use as the cheb coeffs
                    * an existing tools.ChebyshevPolynomial poly object

                domain is a tuple (low,high) of the bounds of the function

                if N is specified, use that number of points

                rtol is the relative tolerance in the coefficients,
                should be approximately the accuracy of the resulting cheb
        """
        self.mapper = lambda x: x
        self.imapper = lambda x: x
        self.domain = (-1,1)

        if domain is not None:
            #if we were passed a domain
            a,b = domain
            self.domain = (a,b)
            #mapper maps from (a,b) to (-1,1)
            self.mapper = tools.gen_mapper(a,b)
            #imapper maps from (-1,1) to (a,b)
            self.imapper = tools.gen_imapper(a,b)

        #by default use numpy float tolerance
        self.rtol = rtol or DEFAULT_TOL

        self._constructed = False

        #Here I have a somewhat inelegant casing out
        #    to allow initilization overloading
        if isinstance(obj, self.__class__):
            #if we have a chebfun, just copy it
            self._from_cheb(other_cheb=obj)
        elif isinstance(obj,tools.ChebyshevPolynomial):
            #we have a chebyshev poly
            self._from_poly(poly=obj)
        elif isinstance(obj,np.ndarray):
            #use the ndarray as our coefficients
            self._from_array(array=obj)
        elif isinstance(obj,str):
            #we have a string, eval it in the numpy namespace
            self._from_string(string=obj)
        elif isinstance(obj,(np.ufunc,np.vectorize)):
            # we're good to go
            self._from_ufunc(ufunc=obj)
        elif callable(obj):
            #try to vectorize a general callable
            self._from_func(func=obj)
        else:
            raise TypeError, "I don't understand your func: {}".format(obj)

        if N is not None:
            #if the user passed in an N, assume that's what he wants
            #we need the function on the interval (-1,1)
            func = lambda x: self.func(self.imapper(x))
            coeffs = tools.fit(func, N)
            self.poly = tools.ChebyshevPolynomial(coeffs,self.domain)
            self._constructed = True

    def _from_cheb(self, other_cheb):
        """ Create a new guy from a cheb """
        logger.debug("INIT: Copying chebfun")
        self.poly = other_cheb.cheb
        self.domain = other_cheb.domain
        self.rtol = other_cheb.rtol
        self.func = other_cheb.func
        self._constructed = True

    def _from_poly(self, poly):
        """ Create a new guy from a poly """
        logger.debug("INIT: chebyshev polynomial")
        self.poly = poly
        self.func = self.poly
        self.domain = tuple(self.poly.domain)
        self._constructed = True

    def _from_array(self,array):
        """ Create a new guy from an array """
        logger.debug("INIT: ndarray")
        self.poly = tools.ChebyshevPolynomial(array,domain=self.domain)
        self.func = self.poly
        self._constructed = True

    def _from_string(self,string):
        """ Create a new guy from a string that we evaluate in the numpy namespace """
        logger.debug("INIT: string")
        self.func = eval("lambda x: {}".format(string),np.__dict__)

    def _from_ufunc(self,ufunc):
        """ Create a new guy from a ufunc """
        logger.debug("INIT: safe non vectorize func")
        self.func = ufunc

    def _from_func(self, func):
        """ Try to build a new guy from a function """
        a,b = self.domain
        xs = (b-a)*np.random.rand(2) + a
        # First see if we can treat it as vectorized anyway
        try:
            logger.debug("INIT: func = %r", func)
            z = func(xs)
            self.func = func
            logger.debug("INIT: sneaky vectorize")
        except Exception:
            logger.debug("INIT: explicit vectorize")
            # frompyfunc was about twice as fast as vectorize
            self.func = frompyfunc(func,1,1)


    @property
    def poly(self):
        """ Try to make construction lazy """
        if not self._constructed: # or not hasattr(self,'_poly'):
            logger.debug("lazy construct")
            a,b = self.domain
            poly = tools.construct(self.func,a,b,rtol=self.rtol)
            self.poly = poly
            self._constructed = True
        return self._poly

    @poly.setter
    def poly(self, x):
        self._constructed = True
        self._poly = x

    @poly.deleter
    def poly(self):
        del self._poly
        self._constructed = False

    @property
    def naf(self):
        return self.__len__() > NAF_CUTOFF

    def trim(self):
        coeffs = tools.trim_arr(self.poly.coef)
        self.poly = tools.ChebyshevPolynomial(coeffs,domain=self.domain)

    def deriv(self,m=1):
        """ Take a derivative, m is the order """
        newcheb = self.poly.deriv(m)
        newguy = self.__class__(newcheb,rtol=self.rtol)
        newguy.trim()
        return newguy

    def integ(self,m=1,k=[],lbnd=None):
        """ Take an integral, m is the number,
            k is an array of constants,
            lbnd is a lower bound
        """
        a,b = self.domain
        if lbnd is None and a<=0<=b:
            lbnd = 0
        else:
            lbnd = a
        newcheb = self.poly.integ(m,k=k,lbnd=lbnd)
        newguy = self.__class__(newcheb,rtol=self.rtol)
        return newguy

    def quad(self):
        """ Try to take the integral """
        goodcoeffs = self.poly.coef[0::2]
        weights = np.fromfunction(lambda x: 2./(1-(2*x)**2), goodcoeffs.shape)
        result =  sum(weights*goodcoeffs)
        #need to multiply by domain
        a,b = self.domain
        return result *0.5 * (b-a)

    def norm(self):
        """ Get the norm of our cheb """
        return math.sqrt((self.__pow__(2)).quad())

    def dot(self,other):
        """ return the dot product of the two functions """
        return (self.__mul__(other)).quad()

    def roots(self):
        """ Get all of the roots,
            note that a lot of these are outside the domain
        """
        return self.poly.roots()

    @property
    def range(self):
        """ try to determine the range for the function """
        a,b = self.domain
        fa = float(self.poly(a))
        fb = float(self.poly(b))
        #use the cheb derivative to find extemums
        extremums = self.poly(self.deriv().introots())
        if extremums:
            exmax = float(extremums.max())
            exmin = float(extremums.min())
            return min(exmin,fa,fb),max(exmax,fa,fb)
        return min(fa,fb), max(fa,fb)

    @property
    def coef(self):
        """ get the coefficients """
        return self.poly.coef

    @property
    def scl(self):
        """ get the *scale* of the cheb, i.e. its largest coefficient """
        return np.abs(self.coef).max()

    def introots(self):
        """ Return all of the real roots in the domain """
        a,b = self.domain
        roots = self.roots()
        realroots = np.real(roots[np.imag(roots)==0])
        return realroots[(a<=realroots)*(realroots<=b)]

    def __len__(self):
        """ Len gives the number of terms """
        return len(self.poly)

    def grid(self,N=1000):
        """ Return the cheb evaluated on a discrete lattice
            returns both the grid, and the evaluations
        """
        a,b = self.domain
        xs = np.linspace(a,b,N)
        return (xs, self.__call__(xs))

    def plot(self,N=1000,*args,**kwargs):
        """ Plot the cheb on its domain """
        a,b = self.domain
        xs,ys = self.grid(N)
        pl = plt.plot(xs,ys,*args,**kwargs)
        plt.xlim(self.domain)
        return pl

    def errplot(self,N=1000,*args,**kwargs):
        """ plot the absolute errors on a log plot """
        a,b = self.domain
        xs, ys = self.grid(N)
        diff = abs(self.func(xs) - ys)
        try:
            pl =  plt.semilogy(xs, diff,*args,**kwargs)
        except ValueError:
            pl = None
            print "Nice! Doesn't look like we have any discernable errors, at all"
        return pl

    def coefplot(self,*args,**kwargs):
        """ plot the absolute values of the coefficients on a semilog plot """
        return plt.semilogy(abs(self.poly.coef),*args,**kwargs)

    def __repr__(self):
        out = "<{}(N={},domain={},rtol={},naf={})>".format(self.__class__.__name__,
                    self.__len__(),self.domain,self.rtol,self.naf)
        return out

    def _repr_html_(self):
        """ Have the plot be the representation in ipython notebook """
        self.plot()

    def __getattr__(self,attr):
        """ Allow numpy overloading, called if something else is broken
            A bit hackish, but avoids poluting the namespace for the Chebfun """
        if attr in NP_OVERLOAD:
            ufunc = np.__getattribute__(attr)
            def func():
                return self._new_func( lambda x: ufunc( self._eval(x) ) )
            func.__name__ = attr
            func.__doc__  = "wraps numpy ufunc: {}".format(attr)
            return func
        return self.__getattribute__(attr)

    def _new_func(self,newfunc,rtol=None,domain=None):
        """ Replace the function with another function """
        # FIXME: I'd like to do this, but it causes an infinite loop
        # self.rtol = rtol or self.rtol
        # self.domain = domain or self.domain
        # a,b = self.domain
        # self.mapper = tools.gen_mapper(a,b)
        # self.imapper = tools.gen_imapper(a,b)
        # logger.debug("func type: %r", type(newfunc))
        # self.func = newfunc
        # self._constructed = False
        # return self
        logger.debug("NEWFUNC")
        rtol = rtol or self.rtol
        domain = domain or self.domain
        newguy = self.__class__(newfunc,rtol=rtol,domain=domain)
        return newguy

    def _new_domain(self,other):
        """ Compose two domains """
        a,b = self.domain
        othera, otherb = getattr(other,"domain",(-np.inf,np.inf))
        return (max(a,othera), min(b,otherb))

    def _compose(self,other,op):
        """ Given some other thing and an operator, create a new cheb
        by evaluating the two, i.e. function composition """
        logger.debug("_compose, other=%r, op=%r", other, op)
        new_rtol = max(self.rtol, getattr(other,"rtol",0))
        new_domain = self._new_domain(other)

        try:
            newpoly = op(self.poly,other.poly)
            logger.debug("short cut")
            return self.__class__(newpoly,rtol=new_rtol,domain=new_domain)
        except Exception:
            pass
        try:
            newpoly = op(self.poly, other)
            logger.debug("other short cut")
            return self.__class__(newpoly,rtol=new_rtol,domain=new_domain)
        except Exception:
            if callable(other):
                newfunc = lambda x: op(self._eval(x), other(x))
            else:
                newfunc = lambda x: op(self._eval(x) , other)
            return self._new_func(newfunc,rtol=new_rtol,domain=new_domain)

    def _eval(self,arg):
        """ A fancy call """
        if isinstance(arg,Cheb):
            # we have another cheb here
            mya,myb = self.domain
            othera, otherb = arg.range
            assert mya <= othera and myb >= otherb, "Domain must include range of other function"
            return self._new_func(lambda x: self._eval(arg._eval(x)), arg.domain,rtol=min(arg.rtol,self.rtol))
        #Check that we are still in the domain
        a,b = self.domain
        if np.any( (arg < a) + (arg > b) ):
            warnings.warn("Evaluating outside the domain", tools.DomainWarning)
        return self.poly(arg)
    __call__ = _eval

    def __add__(self,other):
        return self._compose(other, operator.add)

    def __radd__(self,other):
        return self._compose(other, opr(operator.add))

    def __sub__(self,other):
        return self._compose(other, operator.sub)

    def __rsub__(self,other):
        return self._compose(other, opr(operator.sub))

    def __mul__(self,other):
        return self._compose(other, operator.mul)

    def __rmul__(self,other):
        return self._compose(other, opr(operator.mul))

    def __div__(self,other):
        return self._compose(other, operator.div)

    def __rdiv__(self,other):
        return self._compose(other, opr(operator.div))

    def __pow__(self,pow):
        return self._compose(pow, operator.pow)

    def __abs__(self):
        newguy = self._new_func(lambda x: np.abs(self._eval(x)))
        return newguy

    def __neg__(self):
        return self.__mul__(-1)

    def __pos__(self):
        return self



class Chebfun(Cheb):
    """ A container for a full chebfun, with piecewise bits """
    def __init__(self, funcs = None, edges = None, imps = None, chebs = None, rtol=None):
        """ Initialize """
        # assert len(chebs) == len(edges)+1, "Number of funcs and interior edges don't match"
        # assert edges == sorted(edges), "Edges must be in order, least to greatest"

        self.edges = edges or (-1,1)

        (a,b) = (self.edges[0], self.edges[-1])
        self.domain = (a,b)
        self.mapper = tools.gen_mapper(a,b)
        self.imapper = tools.gen_imapper(a,b)
        #by default use numpy float tolerance
        self.rtol = rtol or DEFAULT_TOL

        if not isinstance(funcs, collections.Sequence):
            self.funcs = [funcs]*(len(self.edges)-1)
        else:
            self.funcs = funcs

        self.chebs = chebs or [ Cheb(func,domain=edgepair,rtol=self.rtol) for (func,edgepair) in zip( self.funcs, zip( self.edges, self.edges[1:] ) ) ]

        if not imps:
            # by default make the values at the breaks as the average on either side
            nfuncs = len(self.funcs)
            pairs = zip(xrange(nfuncs),xrange(1,nfuncs))
            self.imps = [self.funcs[0](self.edges[0])] + [ 0.5*(self.funcs[i](x) + self.funcs[j](x)) for ((i,j),x) in zip(pairs,self.edges) ] + [ self.funcs[-1](self.edges[-1]) ]
        else:
            self.imps = imps

        #need a self.func
        self._eval = self.__call__

    @property
    def nfuncs(self):
        """ number of functions """
        return len(self.funcs)

    def __call__(self,xs):
        """ Evaluate on some points, using np.piecewise """
        #local access
        edges, imps, chebfuns = self.edges, self.imps, self.chebs

        if len(edges) == 2:
            return chebfuns[0](xs)

        fullfuncs = list( tools.roundrobin( chebfuns, imps ))
        middleconds = ( ( (a<xs) + (xs > b) ) for (a,b) in tools.pairwise( edges[1:-1] ) )
        impulseconds = ( (xs==z) for z in edges[1:-1] )
        conds = [(xs<edges[1])] + list(tools.roundrobin(impulseconds,middleconds)) + [(xs>edges[-2])]
        return piecewise(xs, conds, fullfuncs)

    def _new_chebfuns(self,chebfuns,edges=None,imps=None,rtol=None):
        """ Replace the chebfuns """
        newguy = copy.copy(self)
        newguy.chebfuns = chebfuns

        if edges:
            newguy.edges = edges
        if imps:
            newguy.imps = imps
        if rtol:
            newguy.rtol = rtol

        return newguy


    def __add__(self,other):
        """ Add  """
        try:
            newchebs = [ cheb.__add__(other) for cheb in self.chebs ]
        except NameError:
            newchebs = [ cheb._new_func(lambda x: cheb._eval(x) + other._eval(x),
                            rtol=min(self.rtol,other.rtol)) for cheb in self.chebs ]
        newguy = self._new_chebfuns(newchebs)
        return newguy

    def __len__(self):
        """ get the number of chebfuns """
        return len(self.funcs)

    def __repr__(self):
        out = "<{}(nfuncs={},edges={},rtol={},naf={})>".format(self.__class__.__name__,
                    self.__len__(), self.edges,self.rtol,self.naf)
        return out




# a convenience chebfun
x = Cheb("x")

def xdom(domain=(-1,1),*args,**kwargs):
    """ Convenience function to get an identity on whatever interval you want"""
    return Cheb("x",domain=domain,*args,**kwargs)

def deriv(x,*args,**kwargs):
    return x.deriv(*args,**kwargs)

def integ(x,*args,**kwargs):
    return x.integ(*args,**kwargs)

__all__ = ["Cheb","x","xdom","deriv","integ"] # + mathfuncs.keys()


""" TODO:
    Piecewise chebfuns not working

    they need an appropriate _compose method:

    think about ChebOps

    Note: for speed, I could try to work out the _compose method some more to catch
    cases, i.e. if we can just do the coefficient addition, do it. """
