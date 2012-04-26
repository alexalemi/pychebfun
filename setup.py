#! /usr/bin/env python

import sys
import os
from distutils.core import setup

import chebfun

setup(name="chebfun",
        version=chebfun.__version__,
        description="Simple Chebfun clone",
        long_description=chebfun.__doc__,
        author=chebfun.__author__,
        author_email="alexalemi@gmail.com",
        py_modules=['chebfun'])
