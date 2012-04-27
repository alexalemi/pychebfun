PyChebfun
=========

A python clone of the Matlab [Chebfun](http://www2.maths.ox.ac.uk/chebfun/) package.  Note that I am in no way associated with those guys, but I think they have a really neat package and wanted to see if I could create something similar in python.

Purpose
-------

To try to replicate some of the functionality of Chebfun in python using the tools that already exist in numpy/scipy/pylab

Basic Usage
-----------

The package is setup to be be imported in full

    from pychebfun import *

this gives you the main `Chebfun` class, as well as provides some useful overloading of numpy functions, as well as expose `x` as a simple chebfun.

You can construct simple chebfuns by creating a function of an existing chebfun, e.g.

    1./(1+x**2)

will create a chebfun on the default interval of `(-1,1)`.

The Chebfun constructor also excepts any python function, or strings

    f = Chebfun(lambda x: x**2, (-5,5))
    g = Chebfun("sin(x)+sin(x**2)",(0,10))

The optional second argument gives the domain the function should be defined on.

These chebfuns can then be manipulated, used in other functions, differentiated, integrated, plotted, the roots found, etc.  Note that I've also implemented an ipython display hook that will default to displaying a plot of the function

    f.deriv()   #take a derivative
    deriv(f)    #another way
    f.deriv(2)  #second derivative

    f.integ()   #integral

    f**2        #square it
    f.plot()    #plot the function
    f.errplot() #see the error of the interplate

    f.introots()    #see the roots in the interval
    f.domain    #domain of the function
    f.cheb      #the chebyshev polynomial object
    f.coef      #array of coefficients

    len(f)      #number of interpolating points
    f(rand(10)) #chebfuns behave like ufuncs

![an ipython notebook screenshot](https://github.com/alexalemi/pychebfun/raw/master/docs/ipython-notebook-screenshot.png "ipython notebook screenshot")

Features
--------

Currently, the chebfuns *should* work on any finite domain, and fit to near machine precision

Integrate, differentiate, find roots, manipulate, and in general have fun with functions


Installation
------------

    sudo python setup.py install

Tutorial
--------

A set of tutorial ipython notebooks are included in the `docs/` folder, to run in their full glory

    cd docs/
    ipython notebook --pylab=inline

Note that you need a relatively new version of ipython


ToDo
----

* Implement piecewise-chebfuns
* try to handle infinite domains
* try to make chepops




