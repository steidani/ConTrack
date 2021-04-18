#!/usr/bin/env python

from setuptools import setup, find_packages


with open("README.rst", "r") as f:
    readme = f.read()

setup(
    name="ConTrack",
    version="0.3.1",
    description="Contour Tracking of circulation anomalies in Weather and Climate Data.",
    long_description=readme,
    long_description_content_type="text/x-rst",
    author="Daniel Steinfeld",
    author_email="daniel.steinfeld@alumni.ethz.ch",
    url="https://github.com/steidani/ConTrack",
    project_urls={
        "Bug Tracker": "https://github.com/steidani/ConTrack/issues",
    },
    packages=find_packages(exclude=("tests", "tests.*", "docs", "docs.*", "examples", "examples.*" )),
    python_requires=">=3.6",
    install_requires=open("requirements.txt").read().split(),
    classifiers=[
        "Programming Language :: Python :: 3",
	"License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
	"Intended Audience :: Science/Research",
	"Topic :: Scientific/Engineering :: Atmospheric Science",
    ],
    keywords=["data", "science", "meteorology", "climate", "atmospheric blocking", "troughs and ridges"]
)
