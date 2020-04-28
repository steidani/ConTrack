#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='ConTrack',
    version='0.1.0',
    description='Contour Tracking package for analysing circulation anomalies in Weather and Climate Data.',
    long_description=readme,
    author='Daniel Steinfeld',
    author_email='daniel.steinfeld@alumni.ethz-.ch',
    url='https://github.com/steidani/ConTrack',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
    keywords=['data', 'science', 'meteorology', 'climate', 'atmospheric blocking', 'troughs and ridges']
)
