"""
pychebfun is a python implementation of the chebfun package from MATLAB
"""

__author__ = "Alex Alemi"
__version__ = "0.2"

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import tools

class Fun(object):
    """ Represents a nonpiecewise chebfun """

class Chebfun(object):
    """ Represents a full piecewise chebfun """
    def __init__(self, inp):

        self.funs = []  #list of funs
        self.nfuns = 0  #number of funs
        self.scl = 0    #largest coeff across the series
        self.ends = []  #n+1 ends
        self.imps = []  #values of function at points
