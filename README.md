PyChebfun
=========

A python clone of the Matlab [Chebfun](http://www2.maths.ox.ac.uk/chebfun/) package.  Note that I am in no way associated with those guys, but I think they have a really neat package and wanted to see if I could create something similar in python.

Purpose
-------

To try to replicate some of the functionality of Chebfun in python using the tools that already exist in numpy/scipy/pylab

Usage
-----

The package is setup to be be imported in full

    from pychebfun import *

this gives you the main `Chebfun` class, as well as provides some useful overloading of numpy functions, as well as expose `x` as a simple chebfun.

You can construct simple chebfuns by creating a function of an existing chebfun, e.g.

    1./(1+x**2)

will create a chebfun on the default interval of `(-1,1)`.

The Chebfun constructor also excepts any python function, or strings

    Chebfun(lambda x: x**2, (-5,5))
    Chebfun("sin(x)+sin(x**2)",(0,10))

The optional second argument gives the domain the function should be defined on.


