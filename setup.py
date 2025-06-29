#! /usr/bin/env python

import sys
import os
from distutils.core import setup

import pychebfun

setup(
    name="pychebfun",
    version=pychebfun.__version__,
    description="Simple Chebfun clone",
    long_description=pychebfun.__doc__,
    author=pychebfun.__author__,
    author_email="alexalemi@gmail.com",
    py_modules=["pychebfun"],
    package_data={"": ["LICENSE"]},
    license="ISCL",
)
