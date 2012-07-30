#!/usr/bin/env python

"""
Chebfun is a work in progress clone of the Matlab Chebfun project"""

__author__ = "Alex Alemi"
__version__ = "0.1"

#My imports
import math, warnings, itertools, copy, operator, sys
from bisect import bisect, insort

import numpy as np
import scipy as sp
import pylab as py
import numpy.polynomial.chebyshev as cheb
from scipy.fftpack import idct, dct, ifft, fft
Chebyshev = cheb.Chebyshev

#the maximum number of points to try is 2**MAXPOW
MAXPOW = 15

def opr(func):
    """ reverse the arguments to a function, decorator
    Used to help create all of the __r versions of operators"""
    def rfunc(*args):
        return func(*reversed(args))
    return rfunc


#A simple convergence warning
class ConvergenceWarning(Warning): pass
class DomainWarning(Warning): pass
warnings.simplefilter("always")

class Chebfun(object):
    """ A simple chebfun object, which represents a function defined on a
    domain with a chebyshev polynomial to within machine precision
    """

    def __init__(self,func,domain=None,N=None,rtol=None):
        """ Initilize the chebfun

                func can be one of
                    * a python callable
                    * a numpy ufunc
                    * a string (using the numpy namespace)
                    * a ndarray to use as the chebfun coeffs
                    * an existing Chebyshev poly object

                domain is a tuple (low,high) of the bounds of the function

                if N is specified, use that number of points

                rtol is the relative tolerance in the coefficients,
                should be approximately the accuracy of the resulting chebfun
        """
        self.mapper = lambda x: x
        self.imapper = lambda x: x
        self.domain = (-1,1)
        #tells if we've had problems
        self.naf = False

        if domain is not None:
            #if we were passed a domain
            a,b = domain
            self.domain = (a,b)
            #mapper maps from (a,b) to (-1,1)
            self.mapper = lambda x: (2*x-(a+b))/(b-a)
            #imapper maps from (-1,1) to (a,b)
            self.imapper = lambda x: 0.5*(a+b) + 0.5*(b-a)*x

        #by default use numpy float tolerance
        self.rtol = rtol or 3.*np.finfo(np.float).eps

        need_construct = True

        #Here I have a somewhat inelegant casing out
        #    to allow initilization overloading
        if isinstance(func, self.__class__):
            #if we have a chebfun, just copy it
            self.cheb = func.cheb
            self.domain = func.domain
            self.rtol = func.rtol
            need_construct = False
        elif isinstance(func,Chebyshev):
            #we have a chebyshev poly
            self.cheb = func
            self.func = self.cheb
            self.domain = tuple(self.cheb.domain)
            need_construct = False
        elif isinstance(func,np.ndarray):
            #use the ndarray as our coefficients
            arr = func
            self.cheb = Chebyshev(arr,domain=self.domain)
            self.func = self.cheb
            need_construct = False
        elif isinstance(func,str):
            #we have a string, eval it in the numpy namespace
            self.func = eval("lambda x: {}".format(func),np.__dict__)
        elif isinstance(func,(np.ufunc,np.vectorize)):
            # we're good to go
            self.func = func
        elif callable(func):
            #try to vectorize a general callable
            self.func = np.vectorize(func)
        else:
            raise TypeError, "I don't understand your func: {}".format(func)

        if N is not None:
            #if the user passed in an N, assume that's what he wants
            #we need the function on the interval (-1,1)
            func = lambda x: self.func(self.imapper(x))
            coeffs = self._fit(func, N)
            self.cheb = Chebyshev(coeffs,self.domain)
            need_construct = False

        if need_construct:
            self.construct()

        #this allows me to switch between keeping the funcs and using the chebfun
        #to evaluate the deeper things
        self._eval = self.__call__

    def _fit_builtin(self,func,N):
        """ Return the chebyshev coefficients using the builtin chebfit """
        pts = cheb.chebpts2(N)
        y = func(pts)
        coeffs = cheb.chebfit(pts,y,N)
        return coeffs

    def _fit_idct(self,func,N):
        """ Return the chebyshev coefficients using the idct """
        pts = -cheb.chebpts1(N)
        y = func(pts)
        coeffs = idct(y,type=3)/N
        coeffs[0] /= 2.
        coeffs[-1] /= 2.
        return coeffs

    def _fit_fft(self,func,N):
        """ Get the chebyshev coefficients using fft
            inspired by: http://www.scientificpython.net/1/post/2012/4/the-fast-chebyshev-transform.html
            *Doesn't seem to work right now*
        """
        pts = cheb.chebpts2(N)
        y = func(pts)
        A = y[:,np.newaxis]
        m = np.size(y,0)
        k = m-1-np.arange(m-1)
        V = np.vstack((A[0:m-1,:],A[k,:]))
        F = ifft(V,n=None,axis=0)
        B = np.vstack((F[0,:],2*F[1:m-1,:],F[m-1,:]))

        if A.dtype != 'complex':
            return np.real(B)
        return B

    #set the default fit function
    _fit = _fit_idct

    def construct(self):
        """ Construct the chebyshev polynomial

            Starts with N=4 points and evaluates the function on a set of
            chebyshev points, determining the chebyshev coefficients

            At that point, check to see if the last two coefficients are small
            compared to the largest

            If not, increment N, if yes, trim as many coefs as possible
        """
        #map to the interval (-1,1)
        func = lambda x: self.func(self.imapper(x))
        power = 2
        done = False
        while not done:
            N = 2**power

            coeffs = self._fit(func,N)

            if all(np.abs(coeffs[-2:]) <= np.max(np.abs(coeffs))*self.rtol):
                done = True

            power += 1
            if power > MAXPOW and not done:
                warnings.warn("we've hit the maximum power",ConvergenceWarning)
                self.naf = True
                done = True

        coeffs = self._trim_arr(coeffs)
        self.cheb = Chebyshev(coeffs,self.domain)

    def _trim_arr(self,arr,rtol=None):
        """ trim an array by rtol """
        if rtol is None:
            rtol = self.rtol

        return cheb.chebtrim(arr,max(abs(arr))*rtol)

    def trim(self):
        coeffs = self._trim_arr(self.cheb.coef)
        self.cheb = Chebyshev(coeffs,domain=self.domain)

    def _new_func(self,func,rtol=None,domain=None):
        """ Replace the function with another function """
        rtol = rtol or self.rtol
        domain = domain or self.domain
        newguy = self.__class__(func,rtol=rtol,domain=domain)
        return newguy

    def _wrap_call(self,func):
        """ We were called by a wrapped function """
        newguy = self._new_func(lambda x: func(self._eval(x)))
        return newguy

    def _new_domain(self,other):
        """ Compose two domains """
        a,b = self.domain
        othera, otherb = getattr(other,"domain",(-np.inf,np.inf))
        return (max(a,othera), min(b,otherb))

    def _compose(self,other,op):
        """ Given some other thing and an operator, create a new chebfun
        by evaluating the two, i.e. function composition """
        if callable(other):
            newfunc = lambda x: op(self._eval(x), other(x))
        else:
            newfunc = lambda x: op(self._eval(x) , other)

        new_rtol = max(self.rtol, getattr(other,"rtol",0))
        new_domain = self._new_domain(other)

        return self._new_func(newfunc,rtol=new_rtol,domain=new_domain)

    def __call__(self,arg):
        """ make it behave like a function """
        if isinstance(arg,Chebfun):
            # we have another chebfun here
            mya,myb = self.domain
            othera, otherb = arg.range
            assert mya <= othera and myb >= otherb, "Domain must include range of other function"

            return self._new_func(lambda x: self._eval(arg._eval(x)), arg.domain,rtol=min(arg.rtol,self.rtol))
        a,b = self.domain
        if np.any( (arg < a) + (arg > b) ):
            warnings.warn("Evaluating outside the domain", DomainWarning)
        return self.cheb(arg)

    def __add__(self,other):
        """ Add  """
        return self._compose(other, operator.add)

    def __radd__(self,other):
        """ Reversed Add """
        return self._compose(other, opr(operator.add))

    def __sub__(self,other):
        """ Subtract  """
        return self._compose(other, operator.sub)

    def __rsub__(self,other):
        """ Reversed Subtract """
        return self._compose(other, opr(operator.sub))

    def __mul__(self,other):
        """ Multiply """
        return self._compose(other, operator.mul)

    def __rmul__(self,other):
        """ Reverse Multiply """
        return self._compose(other, opr(operator.mul))

    def __div__(self,other):
        """ Division: makes a new chebfun """
        return self._compose(other, operator.div)

    def __rdiv__(self,other):
        """ Reversed divide """
        return self._compose(other, opr(operator.div))

    def __pow__(self,pow):
        """ Raise to a power """
        return self._compose(pow, operator.pow)

    def __abs__(self):
        """ absolute value """
        newguy = self._new_func(lambda x: np.abs(self._eval(x)))
        return newguy

    def __neg__(self):
        """ negative """
        return self.__mul__(-1)

    def __pos__(self):
        """ pos returns self """
        return self

    def deriv(self,m=1):
        """ Take a derivative, m is the order """
        newcheb = self.cheb.deriv(m)
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
        newcheb = self.cheb.integ(m,k=k,lbnd=lbnd)
        newguy = self.__class__(newcheb,rtol=self.rtol)
        return newguy

    def quad(self):
        """ Try to take the integral """
        goodcoeffs = self.cheb.coef[0::2]
        weights = np.fromfunction(lambda x: 2./(1-(2*x)**2), goodcoeffs.shape)
        result =  sum(weights*goodcoeffs)
        #need to multiply by domain
        a,b = self.domain
        return result *0.5 * (b-a)

    def norm(self):
        """ Get the norm of our chebfun """
        return math.sqrt((self.__pow__(2)).quad())

    def dot(self,other):
        """ return the dot product of the two functions """
        return (self.__mul__(other)).quad()

    def roots(self):
        """ Get all of the roots,
            note that a lot of these are outside the domain
        """
        return self.cheb.roots()

    @property
    def range(self):
        """ try to determine the range for the function """
        a,b = self.domain
        fa = float(self._eval(a))
        fb = float(self._eval(b))
        #use the chebfun derivative to find extemums
        extremums = self._eval(self.deriv().introots())
        if extremums:
            exmax = float(extremums.max())
            exmin = float(extremums.min())
            return min(exmin,fa,fb),max(exmax,fa,fb)
        return min(fa,fb), max(fa,fb)

    @property
    def coef(self):
        """ get the coefficients """
        return self.cheb.coef

    @property
    def scl(self):
        """ get the *scale* of the chebfun, i.e. its largest coefficient """
        return np.abs(self.coef).max()

    def introots(self):
        """ Return all of the real roots in the domain """
        a,b = self.domain
        roots = self.roots()
        realroots = np.real(roots[np.imag(roots)==0])
        return realroots[(a<=realroots)*(realroots<=b)]

    def __len__(self):
        """ Len gives the number of terms """
        return len(self.cheb)

    def grid(self,N=1000):
        """ Return the chebfun evaluated on a discrete lattice
            returns both the grid, and the evaluations
        """
        a,b = self.domain
        xs = np.linspace(a,b,N)
        return (xs, self.__call__(xs))

    def plot(self,N=1000,*args,**kwargs):
        """ Plot the chebfun on its domain """
        a,b = self.domain
        xs,ys = self.grid(N)
        pl = py.plot(xs,ys,*args,**kwargs)
        py.xlim(self.domain)
        return pl

    def errplot(self,N=1000,*args,**kwargs):
        """ plot the absolute errors on a log plot """
        a,b = self.domain
        xs, ys = self.grid(N)
        diff = abs(self.func(xs) - ys)
        try:
            pl =  py.semilogy(xs, diff,*args,**kwargs)
        except ValueError:
            pl = None
            print "Nice! Doesn't look like we have any discernable errors, at all"
        return pl

    def coefplot(self,*args,**kwargs):
        """ plot the absolute values of the coefficients on a semilog plot """
        return py.semilogy(abs(self.cheb.coef),*args,**kwargs)

    def __repr__(self):
        out = "<{}(N={},domain={},rtol={},naf={})>".format(self.__class__.__name__,
                    self.__len__(),self.domain,self.rtol,self.naf)
        return out

    def _repr_html_(self):
        """ Have the plot be the representation in ipython notebook """
        self.plot()



class PiecewiseChebfun(Chebfun):
    """ A container for a piecewise chebfun """
    def __init__(self, funcs, edges, imps = None, domain = None, rtol=None):
        """ Initialize """
        assert len(funcs) == len(edges)+1, "Number of funcs and interior edges don't match"
        assert edges == sorted(edges), "Edges must be in order, least to greatest"

        self.funcs = funcs
        self.edges = edges

        self.mapper = lambda x: x
        self.imapper = lambda x: x
        self.domain = (-1,1)
        #tells if we've had problems
        self.naf = False

        if domain is not None:
            #if we were passed a domain
            a,b = domain
            self.domain = (a,b)
            #mapper maps from (a,b) to (-1,1)
            self.mapper = lambda x: (2*x-(a+b))/(b-a)
            #imapper maps from (-1,1) to (a,b)
            self.imapper = lambda x: 0.5*(a+b) + 0.5*(b-a)*x

        #by default use numpy float tolerance
        self.rtol = rtol or 3.*np.finfo(np.float).eps

        if imps is None:
            # by default make the values at the breaks the average on either side
            nfuncs = len(self.funcs)
            pairs = zip(xrange(nfuncs),xrange(1,nfuncs))
            self.imps = [ 0.5*(self.funcs[i](x) + self.funcs[j](x)) for ((i,j),x) in zip(pairs,edges) ]
        else:
            self.imps = imps

        a,b = self.domain
        fulledges = [a] + edges + [b]
        #get the individual domains
        edgepairs = zip(fulledges,fulledges[1:])

        #initial go at chebfuns
        self.chebfuns = [Chebfun(self.funcs[i],domain=edgepair,rtol=self.rtol) for (i,edgepair) in enumerate(edgepairs)]

        #need a self.func
        self._eval = self.__call__

    @property
    def nfuncs(self):
        """ number of functions """
        return len(self.funcs)

    def __call__(self,xs):
        """ Evaluate on some points """
        #local access
        edges, imps, chebfuns = self.edges, self.imps, self.chebfuns
        @np.vectorize
        def call_on_x(x):
            if x in edges:
                return imps[edges.index(x)]
            else:
                pk = bisect(edges,x)
                return chebfuns[pk](x)

        return call_on_x(xs)

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

    def _wrap_call(self,func):
        """ We were called by a wrapped func """
        newchebs = [ cheb._new_func(lambda x: func(cheb._eval(x))) for cheb in self.chebfuns ]
        newguy = self._new_chebfuns(newchebs)
        return newguy

    def __add__(self,other):
        """ Add  """
        try:
            newchebs = [ cheb.__add__(other) for cheb in self.chebfuns ]
        except NameError:
            newchebs = [ cheb._new_func(lambda x: cheb._eval(x) + other._eval(x),
                            rtol=min(self.rtol,other.rtol)) for cheb in self.chebfuns ]
        newguy = self._new_chebfuns(newchebs)
        return newguy

    def __len__(self):
        """ get the number of chebfuns """
        return len(self.funcs)

    def __repr__(self):
        out = "<{}(nfuncs={},domain={},rtol={},naf={})>".format(self.__class__.__name__,
                    self.__len__(), self.domain,self.rtol,self.naf)
        return out



############## BEGIN NUMPY OVERLOADING ##############

#Let's overload all the numpy ufuncs
#  so that if they are called on a chebfun, make a new chebfun
def wrap(func):
    def chebfunc(arg):
        __doc__  = func.__doc__
        if isinstance(arg,Chebfun):
            #return arg.__class__(lambda x: func(arg.func(x)), arg.domain, rtol=arg.rtol)
            return arg._wrap_call(func)
        else:
            return func(arg)
    return chebfunc

this_module = sys.modules[__name__]
#list of numpy functions to overload
toimport = ["sin","cos","tan","sinh","cosh","tanh",
                "arcsin","arccos","arctan",
                "arcsinh","arccosh","arctanh",
                "exp","exp2","log","log2","log10","expm1","log1p",
                "sqrt","square","reciprocal","sign","absolute","conj"]
#make wrapped version of the functions
mathfuncs = { k:wrap(v) for k,v in np.__dict__.iteritems() if k in toimport }

#add the funcs to the current name space
for k,v in mathfuncs.iteritems():
    setattr(this_module,k,v)

#########  END NUMPY OVERLOADING ###############


# a convenience chebfun
x = Chebfun("x")

def xdom(domain=(-1,1),*args,**kwargs):
    """ Convenience function to get an identity on whatever interval you want"""
    return Chebfun("x",domain=domain,*args,**kwargs)

def deriv(x,*args,**kwargs):
    return x.deriv(*args,**kwargs)

def integ(x,*args,**kwargs):
    return x.integ(*args,**kwargs)

__all__ = ["Chebfun","wrap","x","xdom","deriv","integ"] + mathfuncs.keys()


""" TODO:
    Piecewise chebfuns not working

    they need an appropriate _compose method:

    think about ChebOps

    Note: for speed, I could try to work out the _compose method some more to catch
    cases, i.e. if we can just do the coefficient addition, do it. """
