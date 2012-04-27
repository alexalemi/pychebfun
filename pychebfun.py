#!/usr/bin/env python

"""
Chebfun is a work in progress clone of the Matlab Chebfun project"""

__author__ = "Alex Alemi"
__version__ = "0.1"

import numpy as np
import pylab as py

import numpy.polynomial.chebyshev as cheb

from scipy.fftpack import idct, dct, ifft, fft

import sys

Chebyshev = cheb.Chebyshev

#the maximum number of points to try is 2**MAXPOW
MAXPOW = 16

#Let's overload all the numpy ufuncs
#  so that if they are called on a chebfun, make a new chebfun
def wrap(func):
	def chebfunc(arg):
		__doc__  = func.__doc__
		if isinstance(arg,Chebfun):
			return arg.__class__(lambda x: func(arg.func(x)), arg.domain)
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


class Chebfun(object):
	""" A simple chebfun object, which represents a function defined on a
	domain with a chebyshev polynomial to within machine precision 
	"""

	def __init__(self,func,edges=None,N=None,rtol=None):
		""" Initilize the chebfun
	 
				func can be one of
					* a pyton callable
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

		if edges is not None:
			#if we were passed some edges
			a,b = edges
			self.domain = (a,b)
			#mapper maps from (a,b) to (-1,1)
			self.mapper = lambda x: 2.*(x-a)/(b-a)-1.
			#imapper maps from (-1,1) to (a,b)
			self.imapper = lambda x: 0.5*(b-a)*(x+1)+a

		if rtol is None:
			#by default use numpy float tolerance
			self.rtol = 3.*np.finfo(np.float).eps
		else:
			self.rtol = rtol

		need_construct = True

		if isinstance(func,Chebyshev):
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
			if power >= MAXPOW:
				print "Warning, we've hit the maximum power"
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
		""" Add  """
		try:
			newcheb = self.cheb.__add__(other.cheb)
		except AttributeError:
			newcheb = self.cheb.__add__(other)
		newguy =  self.__class__(newcheb)
		newguy.trim()

		return newguy

	def __radd__(self,other):
		""" Reversed Add """
		try:
			newcheb = self.cheb.__radd__(other.cheb)
		except AttributeError:
			newcheb = self.cheb.__radd__(other)
		newguy =  self.__class__(newcheb)
		newguy.trim()

		return newguy

	def __sub__(self,other):
		""" Subtract  """
		try:
			newcheb = self.cheb.__sub__(other.cheb)
		except AttributeError:
			newcheb = self.cheb.__sub__(other)
		newguy =  self.__class__(newcheb)
		newguy.trim()

		return newguy

	def __rsub__(self,other):
		""" Reversed Subtract """
		try:
			newcheb = self.cheb.__rsub__(other.cheb)
		except AttributeError:
			newcheb = self.cheb.__rsub__(other)
		newguy =  self.__class__(newcheb)
		newguy.trim()

		return newguy

	def __mul__(self,other):
		""" Multiply """
		try:
			newcheb = self.cheb.__mul__(other.cheb)
		except AttributeError:
			newcheb = self.cheb.__mul__(other)
		newguy = self.__class__(newcheb)
		newguy.trim()

		return newguy

	def __rmul__(self,other):
		""" Reverse Multiply """
		try:
			newcheb = self.cheb.__rmul__(other.cheb)
		except AttributeError:
			newcheb = self.cheb.__rmul__(other)
		newguy = self.__class__(newcheb)
		newguy.trim()

		return newguy

	def __div__(self,other):
		""" Division: makes a new chebfun """
		try:
			newguy = self.__class__(lambda x: self.func(x)/other.func(x),self.domain)
		except AttributeError:
			newguy = self.__class__(lambda x: self.func(x)/other,self.domain)
		return newguy

	def __rdiv__(self,other):
		""" Reversed divide """
		try:
			newguy = self.__class__(lambda x: other.func(x)/self.func(x),self.domain)
		except AttributeError:
			newguy = self.__class__(lambda x: other/self.func(x),self.domain)
		return newguy

	def __pow__(self,pow):
		""" Raise to a power """
		newcheb = self.cheb.__pow__(pow)
		newguy = self.__class__(newcheb)
		newguy.trim()
		return newguy

	def __abs__(self):
		""" absolute value """
		newguy = self.__class__(lambda x: np.abs(self.func(x)),self.domain)
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
		newguy = self.__class__(newcheb)
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
		newguy = self.__class__(newcheb)
		return newguy

	def quad(self):
		""" Try to take the integral """
		goodcoeffs = self.cheb.coef[0::2]
		weights = np.fromfunction(lambda x: 2./(1-(2*x)**2), goodcoeffs.shape)
		result =  sum(weights*goodcoeffs)
		#need to multiply by domain
		a,b = self.domain
		return result *0.5 * (b-a)

	def roots(self):
		""" Get all of the roots,
			note that a lot of these are outside the domain
		"""
		return self.cheb.roots()

	@property 
	def range(self):
		""" try to determine the range for the function """
		a,b = self.domain
		fa = float(self.func(a))
		fb = float(self.func(b))
		#use the chebfun derivative to find extemums
		extremums = self.func(self.deriv().introots())
		exmax = float(extremums.max())
		exmin = float(extremums.min())
		return min(exmin,fa,fb),max(exmax,fa,fb)
	
	@property 
	def coef(self):
		""" get the coefficients """
		return self.cheb.coef

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
		return py.semilogy(xs, diff,*args,**kwargs)

	def coefplot(self,*args,**kwargs):
		""" plot the absolute values of the coefficients on a semilog plot """
		return py.semilogy(abs(self.cheb.coef),*args,**kwargs)

	def __repr__(self):
		out = "<{}(N={},domain={})>".format(self.__class__.__name__,
					self.__len__(),self.domain)
		return out

	def _repr_html_(self):
		""" Have the plot be the representation in ipython notebook """
		self.plot()



#a couple convienence chebfuns
x = Chebfun("x")
xp = Chebfun("x",(0,1))

def deriv(x,*args,**kwargs):
	return x.deriv(*args,**kwargs)

def integ(x,*args,**kwargs):
	return x.integ(*args,**kwargs)

__all__ = ["Chebfun","wrap","x","xp","deriv","integ"] + mathfuncs.keys()
