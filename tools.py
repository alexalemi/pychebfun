
import warnings

#A simple convergence warning
class ConvergenceWarning(Warning): pass
class DomainWarning(Warning): pass
warnings.simplefilter("always")

import numpy as np
from numpy import max, abs, where
from collections import namedtuple
import numpy.polynomial.chebyshev as npcheb

from scipy.fftpack import dct

EPS = np.finfo(float).eps
MAXPOW = 15
NAF_CUTOFF = 128
DEFAULT_TOL = 3.*EPS

chebpts = npcheb.chebpts2

def vals_to_coeffs(f):
    """ Given the samples of a function on
    a chebyshev grid, return the coefficients """
    N = f.size-1
    out = dct(f[::-1],type=1)/N
    out[0] /= 2.
    out[-1] /= 2.
    return out

def coeffs_to_vals(coeffs):
    """ Given the coefficients, give the 
    values of a function """
    foo = coeffs.copy()
    foo[0] *= 2
    foo[-1] *= 2
    vals = dct(foo,type=1)/2
    return vals[::-1]

def fit(func, N):
    """ Return the values of a function on a chebyshev grid """


def gen_mapper(a,b):
    """ Returns a function that mapps from (a,b) to (-1,1) """
    return lambda x: (2.*x - (a+b))/(b-a)

def gen_imapper(a,b):
    """ Returns a function that maps from (-1,1) to (a,b) """
    return lambda x: 0.5*(a+b) + 0.5*(b-a)*x

def trim_arr(arr,scl,rtol=DEFAULT_TOL):
    """ trim an array by rtol """
    ind, = where(abs(arr) > scl*rtol)
    if len(ind) == 0:
        return arr[:1] * 0
    return arr[:ind[-1]+1].copy()

chebinfo = namedtuple('chebinfo','pts vals coeffs N scl domain func')

def construct(func,a=-1,b=1,rtol=DEFAULT_TOL):
    """ Construct the chebyshev polynomial

        Starts with N=4 points and evaluates the function on a set of
        chebyshev points, determining the chebyshev coefficients

        At that point, check to see if the last two coefficients are small
        compared to the largest

        If not, increment N, if yes, trim as many coefs as possible
    """
    # logger.debug("Inside Construct")
    #map to the interval (-1,1)
    imapper = gen_imapper(a,b)
    mapped_func = lambda x: func(imapper(x))
    power = 2
    done = False
    while not done:
        N = 2**power
        
        pts = chebpts(N)
        vals = mapped_func(pts)
        coeffs = vals_to_coeffs(vals)
        scl = max(abs(coeffs))

        if all(abs(coeffs[-2:]) <= scl*rtol):
            done = True

        power += 1
        if power > MAXPOW and not done:
            warnings.warn("we've hit the maximum power",ConvergenceWarning)
            done = True

    coeffs = trim_arr(coeffs, scl, rtol)
    N = len(coeffs)
    vals = coeffs_to_vals(coeffs)
    pts = chebpts(N)
    return chebinfo(pts=pts, vals=vals, coeffs=coeffs, 
            scl=scl, domain=(a,b), func=func, N=N)

def interp(vals,xs,pts=None,domain=(-1,1)):
    """ interpolate the function represented by the modal points f, on the points xs """
    N = vals.size
    if pts is None:
        pts = chebpts(N)
    mapper = gen_mapper(*domain)
    mapxs = mapper(xs)
    weights = np.r_[ 1./2, np.ones(N-2), 1./2 ] * (-1)**np.arange(N)
    mid = ( weights / ( mapxs[:,None] - pts[None,:] ))
    top =  (mid * vals ).sum(-1)
    bottom =  mid.sum(-1)
    return top/bottom

########################################################
## Chebfun utility commands, translated from the matlab
########################################################

def default_vertical_scale(f,a,b):
    """ Generate a guess at the vertical scale """
    xx = np.array([-0.85441,-0.333331,0.0012,0.2212129,0.766652])
    xx = b*(xx+1)/2. + a*(1-xx)/2.
    vs = max(abs(f(xx)))
    return vs

def detectedge(f,a,b,hs=None,vs=None,der=None,checkblowup=True):
    """ Try to detect an edge """
    edge = None
    nder = 4
    N = 15
    der = der or (lambda x: 1.)
    vs = vs or default_vertical_scale(f,a,b)
    hs = hs or (b-a+0.0)

    if (b-a)**nder < 10*DEFAULT_TOL:
        return

    (na,nb,maxd) = maxder(f,a,b,nder=nder,N=50,der=der)
    maxd1 = maxd

    ends = (na[nder-1],nb[nder-1])

    while (maxd[nder-1] != inf) and (~isnan(maxd[nder-1])) and (diff(ends) > DEFAULT_TOL*hs):
        maxd1 = maxd[:nder]
        (na,nb,maxd) = maxder( f, ends[0], ends[1], nder, N, der)
        c1 = (4.5 - arange(nder))*maxd1
        c2 = (10. * vs/(hs**arange(nder)))
        nder_cands, = ( (maxd > c1) & (maxd > c2) ).nonzero()
        if not nder_cands.size:
            return
        nder = nder_cands[0]+1
        if (nder == 0) and ( diff(ends) < 1e-3*hs ):
            return findjump(f,ends[0],ends[2], hs, vs, der)

        ends = ( na[nder-1], nb[nder-1] )

        if checkblowup and abs(f( (ends[0]+ends[1])/2. )) > 100*vs:
            nedge = findblowup(f, ends[0], ends[1], vs, hs)
            if not nedge:
                checkblowup = False
            else:
                edge = nedge
                return edge

    edge = mean(ends)
    return edge

def maxder(f,a,b,nder,N=15,der=None):
    """ Compute the maximum values of the derivative,
        and the interval they are in """
    maxd = zeros(nder)
    na = a*ones(nder)
    nb = b*ones(nder)
    der = der or (lambda x: 1.)

    x = linspace(a,b,N)
    dx = (b-a)/(N-1.)
    dy = f(x)

    for j in xrange(nder):
        dy = diff(dy)
        x = (x[:-1] + x[1:])/2
        fprimeest = np.abs(dy/der(x))
        ind = fprimeest.argmax()
        biggest = fprimeest[ind]
        maxd[j] = biggest
        if ind > 1:
            na[j] = x[ind-1]
        if ind < len(x)-2:
            nb[j] = x[ind+1]


    if dx**nder <= 2*EPS:
        maxd = inf + maxd
    else:
        maxd = maxd/dx**(arange(nder)+1)

    return (na,nb,maxd)


def findjump(f,a,b,hs=None,vs=None,der=None):
    """ Detects a blowup in first the derivative and uses
    bisection to locate the edge. """
    edge = None
    der = der or (lambda x: 1.)
    (ya,yb) = f(a),f(b)

    def max_abs_derivative(ya,yb,a,b):
        return abs(ya-yb)/((b-a)*der((b+a)/2))

    #estimate the max abs of derivative
    maxd = max_abs_derivative(ya,yb,a,b)

    #if the derivative is very small, probably a false edge
    if maxd < 1e-5 * vs/hs:
        return

    cont = 0;  #how many times the derivative has stopped growing
    e1 = (b+a)/2.  #estimate edge location
    e0 = e1+1 # force loop

    #main loop
    # note that maxd = inf whenver dx < EPS
    while (cont < 2 or isinf(maxd)) and (abs(e0 - e1) > EPS):
        # find c at center of the interval [a,b]
        c = (a+b)/2.
        yc = f(c);
        dy1 = max_abs_derivative(yc,ya,a,c)
        dy2 = max_abs_derivative(yb,yc,b,c)
        maxd1 = maxd
        if dy1 > dy2:
            (b,yb) = (c,yc)
            maxd = dy1/(b-a)
        else:
            (a,ya) = (c,yc)
            maxd = dy2/(b-a)
        (e0,e1) = (e1,(a+b)/2.)
        if maxd < maxd1*(1.5):
            cont += 1

    if (e0-e1) <= 2*EPS:
        yright = f(b+EPS)
        if abs(yright-yb) > EPS*100*vs:
            edge = b
        else:
            edge = a

    return edge

def findblowup(f,a,b,vs,hs):
    """ Detects a blowup at values of the function """
    ya = abs(f(a))
    yb = abs(f(b))

    y = array([ya,yb])
    x = array([a,b])

    while b-a > 1e7*hs:
        x = linspace(a,b,50)
        yy = abs(f(x[1:-1]))
        y = r_[ya, yy, yb]
        ind = abs(y).argmax()
        # maxy = y[ind]
        if ind == 0:
            b = x[2]
            yb = y[2]
        elif ind == 49:
            a = x[47]
            ya = y[47]
        else:
            a = x[ind-1]
            yb = y[ind-1]
            b = x[ind+1]
            ya = y[ind+1]

    while (b-a) > 50*EPS:
        x = linspace(a,b,10)
        yy = abs( f(x[1:-1]) )
        y = r_[ya, yy, yb]
        ind = abs(y).argmax()
        # maxy = y[ind]
        if ind == 0:
            b = x[2]
            yb = y[2]
        elif ind == 9:
            a = x[7]
            ya = y[7]
        else:
            a = x[ind-1]
            yb = y[ind-1]
            b = x[ind+1]
            ya = y[ind+1]

    while (b-a) >= 4*EPS:
        x = linspace(a,b,4)
        yy = abs( f(x[1:-1]) )
        y = r_[ya, yy, yb]
        if y[1] > y[2]:
            b = x[2]
            yb = y[2]
        else:
            a = x[1]
            ya = y[1]


    ind = y.argmax()
    ymax = y[ind]
    edge = x[ind]

    if ymax < 1e5*vs:
        return
    return edge


