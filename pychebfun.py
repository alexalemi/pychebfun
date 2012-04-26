#!/usr/bin/env python

"""
Chebfun is a work in progress clone of the Matlab Chebfun project"""

__author__ = "Alex Alemi"
__version__ = "0.1"

import numpy as np
import pylab as py

import numpy.polynomial.chebyshev as cheb

from scipy.fftpack import idct, dct

import sys

Chebyshev = cheb.Chebyshev

#MAXPOW = 16
MAXPOW = 15

def trim(arr):
	""" trim an array """
	eps = np.finfo(arr.dtype).eps
	return cheb.chebtrim(arr,max(abs(arr))*2*eps)


#Let's overload all the numpy ufuncs
def wrap(func):
	def chebfunc(arg):
		__doc__  = func.__doc__
		if hasattr(arg,"cheb"):
			return arg.__class__(lambda x: func(arg.func(x)), arg.domain)
		else:
			return func(arg)
	return chebfunc

this_module = sys.modules[__name__]

toimport = ["sin","cos","tan","sinh","cosh","tanh",
				"arcsin","arccos","arctan",
				"arcsinh","arccosh","arctanh",
				"exp","exp2","log","log2","log10","expm1","log1p",
				"sqrt","square","reciprocal","sign","absolute","conj"]
mathfuncs = { k:wrap(v) for k,v in np.__dict__.iteritems() if k in toimport }

for k,v in mathfuncs.iteritems():
	setattr(this_module,k,v)

class Chebfun(object):
	""" First attempt at a simple chebfun """

	def __init__(self,func,edges=None):
		self.mapper = lambda x: x
		self.imapper = lambda x: x
		self.domain = (-1,1)

		if edges is not None:
			a,b = edges
			self.domain = (a,b)
			self.mapper = lambda x: 2.*(x-a)/(b-a)-1.
			self.imapper = lambda x: 0.5*(b-a)*(x+1)+a

		need_construct = True

		if isinstance(func,Chebyshev):
			#we have a chebyshev poly
			self.cheb = func
			self.func = self.cheb
			self.domain = tuple(self.cheb.domain)
			need_construct = False
		elif isinstance(func,np.ndarray):
			arr = func
			self.cheb = Chebyshev(arr,domain=self.domain)
			self.func = self.cheb
			need_construct = False
		elif isinstance(func,str):
			#we have a string
			self.func = eval("lambda x: {}".format(func),np.__dict__)
		elif isinstance(func,(np.ufunc,np.vectorize)):
			self.func = func
		elif callable(func):
			self.func = np.vectorize(func)
		else:
			raise TypeError, "I don't understand your func: {}".format(func)

		if need_construct:
			self.construct()

	def construct(self):
		""" Construct the chebyshev polynomial """
		#map to the interval (-1,1)
		func = lambda x: self.func(self.imapper(x)) 
		power = 2
		eps = 2*np.finfo(cheb.chebpts2(2).dtype).eps
		done = False
		while not done:
			N = 2**power
			
			#pts = cheb.chebpts2(N)
			pts = -cheb.chebpts1(N)
			
			y = func(pts)
			
			#coeffs = cheb.chebfit(pts,y,N)
			coeffs = idct(y,type=3)/N
			coeffs[0] /= 2.
			coeffs[-1] /= 2.

			if all(np.abs(coeffs[-2:]) <= np.max(np.abs(coeffs))*2*eps):
				done = True
			
			power += 1
			if power >= MAXPOW:
				print "Warning, we've hit the maximum power"
				done = True

		coeffs = trim(coeffs)
		self.cheb = Chebyshev(coeffs,self.domain)

	def trim(self):
		coeffs = trim(self.cheb.coef)
		self.cheb = Chebyshev(coeffs,domain=self.domain)

	def __call__(self,arg):
		""" make it behave like a function """
		if isinstance(arg,Chebfun):
			# we have another chebfun here
			mya,myb = self.domain
			othera, otherb = arg.range
			assert mya <= othera and myb >= otherb, "Domain must include range of other function"

			return self.__class__(lambda x: self.func(arg.func(x)), arg.domain)

		return self.cheb(arg)

	def __add__(self,other):
		""" Add two chebfuns """
		try:
			newcheb = self.cheb.__add__(other.cheb)
		except AttributeError:
			newcheb = self.cheb.__add__(other)
		newguy =  self.__class__(newcheb)
		newguy.trim()

		return newguy

	def __radd__(self,other):
		""" Add two chebfuns """
		try:
			newcheb = self.cheb.__radd__(other.cheb)
		except AttributeError:
			newcheb = self.cheb.__radd__(other)
		newguy =  self.__class__(newcheb)
		newguy.trim()

		return newguy

	def __sub__(self,other):
		""" Add two chebfuns """
		try:
			newcheb = self.cheb.__sub__(other.cheb)
		except AttributeError:
			newcheb = self.cheb.__sub__(other)
		newguy =  self.__class__(newcheb)
		newguy.trim()

		return newguy

	def __rsub__(self,other):
		""" Add two chebfuns """
		try:
			newcheb = self.cheb.__rsub__(other.cheb)
		except AttributeError:
			newcheb = self.cheb.__rsub__(other)
		newguy =  self.__class__(newcheb)
		newguy.trim()

		return newguy

	def __mul__(self,other):
		""" Multiply two of them """
		try:
			newcheb = self.cheb.__mul__(other.cheb)
		except AttributeError:
			newcheb = self.cheb.__mul__(other)
		newguy = self.__class__(newcheb)
		newguy.trim()

		return newguy

	def __rmul__(self,other):
		""" Multiply two of them """
		try:
			newcheb = self.cheb.__rmul__(other.cheb)
		except AttributeError:
			newcheb = self.cheb.__rmul__(other)
		newguy = self.__class__(newcheb)
		newguy.trim()

		return newguy

	def __div__(self,other):
		""" Multiply two of them """
		try:
			newguy = self.__class__(lambda x: self.func(x)/other.func(x))
		except AttributeError:
			newguy = self.__class__(lambda x: self.func(x)/other)
		return newguy

	def __rdiv__(self,other):
		""" Multiply two of them """
		try:
			newguy = self.__class__(lambda x: other.func(x)/self.func(x))
		except AttributeError:
			newguy = self.__class__(lambda x: other/self.func(x))
		return newguy

	def __pow__(self,pow):
		newcheb = self.cheb.__pow__(pow)
		newguy = self.__class__(newcheb)
		newguy.trim()
		return newguy

	def __abs__(self):
		newguy = self.__class__(lambda x: np.abs(self.func(x)))
		return newguy

	def __neg__(self):
		return self.__mul__(-1)

	def __pos__(self):
		return self

	def deriv(self,m=1):
		""" Take a derivative, m is the order """
		newcheb = self.cheb.deriv(m)
		newguy = self.__class__(newcheb)
		newguy.trim()
		return newguy

	def integ(self,m=1,k=[],lbnd=None):
		""" Take an integral, m is the number, k is an array of constants, lbnd is a lower bound """
		a,b = self.domain
		if lbnd is None and a<=0<=b:
			lbnd = 0
		else:
			lbnd = a
		newcheb = self.cheb.integ(m,k=k,lbnd=lbnd)
		newguy = self.__class__(newcheb)
		return newguy

	def quad(self):
		""" Try to take the integral """
		goodcoeffs = self.cheb.coef[0::2]
		weights = np.fromfunction(lambda x: 2./(1-(2*x)**2), goodcoeffs.shape)
		return sum(weights*goodcoeffs)

	def roots(self):
		return self.cheb.roots()

	@property 
	def range(self):
		a,b = self.domain
		fa = float(self.func(a))
		fb = float(self.func(b))
		extremums = self.func(self.deriv().introots())
		exmax = float(extremums.max())
		exmin = float(extremums.min())
		return min(exmin,fa,fb),max(exmax,fa,fb)
	
	@property 
	def coef(self):
		return self.cheb.coef

	def introots(self):
		a,b = self.domain
		roots = self.roots()
		realroots = np.real(roots[np.imag(roots)==0])
		return realroots[(a<=realroots)*(realroots<=b)]

	def __len__(self):
		return len(self.cheb)

	def plot(self,N=1000):
		a,b = self.domain
		xs = np.linspace(a,b,N)
		pl = py.plot(xs,self.cheb(xs))
		py.xlim(self.domain)
		return pl

	def errplot(self,N=1000):
		a,b = self.domain
		xs = np.linspace(a,b,N)
		diff = abs(self.__call__(xs) - self.func(xs))
		return py.semilogy(xs, diff)

	def coefplot(self):
		return py.semilogy(abs(self.cheb.coef))

	def __repr__(self):
		out = "<{}(N={})>".format(self.__class__.__name__,self.__len__())
		return out

	def _repr_html_(self):
		""" Have the plot be the representation in ipython notebook """
		self.plot()




x = Chebfun("x")
xp = Chebfun("x",(0,1))

def deriv(x,*args,**kwargs):
	return x.deriv(*args,**kwargs)

def integ(x,*args,**kwargs):
	return x.integ(*args,**kwargs)

__all__ = ["Chebfun","wrap","trim","x","xp","deriv","integ"] + mathfuncs.keys()
