.. image:: docs/logo_contrack.png
   :width: 30 px
   :align: center


###########################
ConTrack - Contour Tracking
###########################
==================================================================================
Spatial and temporal tracking of circulation anomalies in weather and climate data
==================================================================================

ConTrack is a Python package intended to simpify the process of tracking and analyzing synoptic weather features (individual systems or long-term climatology) in weather and climate datasets. This feature-based tool is mostly used to track and characterize the entire life cycle of atmospheric blocking, but can be also used to identify upper-level troughs/cyclones and ridges/anticyclones (storm track). It is built on top of `xarray`_ and `scipy`_.

Based on the atmospheric blocking index (FORTRAN) by `Schwierz et al. (2004) <https://doi.org/10.1029/2003GL019341>`_ developed at the `Institute for Atmospheric and Climate Science, ETH Zurich <https://iac.ethz.ch/group/atmospheric-dynamics.html>`_.

See also:  

- `Scherrer et al. (2005) <https://doi.org/10.1002/joc.1250>`_: 
- `Croci-Maspoli et al. (2007) <https://doi.org/10.1175/JCLI4029.1>`_
- `Pfahl et al. (2015) <https://www.nature.com/articles/ngeo2487>`_
- `Woollings et al. (2018) <https://link.springer.com/article/10.1007/s40641-018-0108-z#appendices>`_
- `Steinfeld and Pfahl (2019) <https://doi.org/10.1007/s00382-019-04919-6>`_
- `Steinfeld et al. (2020) <https://doi.org/10.5194/wcd-2020-5>`_
- and used in many more atmospheric blocking studies...

The ERA-Interim global blocking climatology used in Steinfeld and Pfahl (2019) is publicly available via an ETH Zurich-based web server [`http://eraiclim.ethz.ch/ <http://eraiclim.ethz.ch/>`_ , see `Sprenger et al. (2017) <https://doi.org/10.1175/BAMS-D-15-00299.1>`_].  

..
  References
.. _xarray: https://xarray.pydata.org/en/stable/
.. _scipy: https://www.scipy.org/

==========
What's New
==========

v0.1.0 (20.04.2020): 
--------------------

- Extend functionality: Calculate anomalies from daily (long-term) climatology.
- ``pip install contrack`` is currently not working -> cartopy dependency error

future plans: 
--------------------
- life cycle characteristics: temporal evolution of intensity, spatial extent, center of mass and age from genesis to lysis.
- calculate anomalies based on pre-defined climatology.

============
Installation
============

Using pip
---------

Ideally install it in a virtual environment.

.. code:: bash

    pip install contrack

Copy from Github repository
---------------------------

Copy/clone locally the latest version from ConTrack:

.. code-block:: bash

    git clone git@github.com:steidani/ConTrack.git /path/to/local/contrack
    cd path/to/local/contrack

==========
Tutorial
==========

Example: Calculate blocking climatology 
---------------------------------------

.. code-block:: python 
   
   # import contrack module 
   from contrack import contrack

   # initiate blocking instance
   block = contrack()
   
   # read ERA5 Z500 (geopotential at 500 hPa, daily with 1° spatial resolution)
   # downloaded from https://cds.climate.copernicus.eu
   block.read('data/era5_1981-2010_z_500.nc')

   # select only winter months January, February and December
   block.ds = block.ds.sel(time=block.ds.time.dt.month.isin([1, 2, 12]))

   # calculate geopotential height
   block.calculate_gph_from_gp(gp_name='z',
                               gp_unit='m**2 s**-2',
                               gph_name='z_height')

   # calculate Z500 anomaly with respect to 31-day running mean (long-term) climatology, 
   block.calc_anom('z_height', window=31)

   # Finally, track blocking anticyclones (>=150gmp, 50% overlap twosided, 5 days persistence)
   block.run_contrack(variable='anom', 
                      threshold=150,
		      gorl='gt'
                      overlap=0.5,
                      persistence=5,
		      twosided=True)

   # plotting blocking frequency (in %) for winter over Northern Hemisphere
   import matplotlib.pyplot as plt
   import cartopy.crs as ccrs

   fig, ax = plt.subplots(figsize=(7, 5), subplot_kw={'projection': ccrs.NorthPolarStereo()})
   (xr.where(block['flag']>1,1,0).sum(dim='time')/block.ntime*100).plot(levels=np.arange(2,18,2), cmap='Oranges', extend = 'max', transform=ccrs.PlateCarree())
   (xr.where(block['flag']>1,1,0).sum(dim='time')/block.ntime*100).plot.contour(colors='grey', linewidths=0.8, levels=np.arange(2,18,2), transform=ccrs.PlateCarree())
   ax.set_extent([-180, 180, 30, 90], crs=ccrs.PlateCarree()); ax.coastlines();
   plt.show()

.. image:: docs/era5_blockingfreq_DJF.png
   :width: 20 px
   :align: center
