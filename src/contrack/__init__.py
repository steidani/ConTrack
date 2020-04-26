"""

Description
-----------

 The contrack package provides classes and function to read, write, plot and
 analyze circulation anomalies in weather and climate data.


Content
-------

 The following classes are available:

 contrack:      To create a contrack object with functions to detect and track circulation anomalies


Examples
--------

>>> filename = 'era5_clim_z500.nc'
>>> blocking = contrack()
>>> blocking.read(filename)

"""
from .contrack import contrack  # noqa
