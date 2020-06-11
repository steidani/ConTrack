
###########################
ConTrack - Contour Tracking
###########################
=============================================================
Tracking of circulation anomalies in weather and climate data
=============================================================

Based on the atmospheric blocking index by `Schwierz et al. (2004) <https://doi.org/10.1029/2003GL019341>`_ developed at the `Institute for Atmospheric and Climate Science, ETH Zurich <https://iac.ethz.ch/group/atmospheric-dynamics.html>`_

See also:  

- `Croci-Maspoli et al. (2007) <https://doi.org/10.1175/JCLI4029.1>`_
- `Pfahl et al. (2015) <https://www.nature.com/articles/ngeo2487>`_
- `Steinfeld and Pfahl (2019) <https://doi.org/10.1007/s00382-019-04919-6>`_
- `Steinfeld et al. (2020) <https://doi.org/10.5194/wcd-2020-5>`_

The PV-Anomaly blocking climatology used in Steinfeld and Pfahl (2019) is publicly available via an ETH Zurich-based web server [`http://eraiclim.ethz.ch/ <http://eraiclim.ethz.ch/>`_ , see Sprenger et al. (2017)].  

==========
What's New
==========

v0.1.0 (20.04.2020): 
--------------------

- Extend functionality: Calculating anomalies from daily (long-term) climatology.

Install the development environment
-----------------------------------

Copy locally the latest version from ConTrack:

.. code-block:: bash

    git clone git@github.com:steidani/ConTrack.git /path/to/local/contrack
    cd path/to/local/contrack

==========
Tutorial
==========

Example: Calculate blocking climatology 
--------------------

.. code-block:: python 
   
   # import contrack module 
   from contrack import contrack

   # initiate blocking instance
   block = contrack()
   
   # read ERA5 z500 (geopotential at 500 hPa)
   block.read('data/era5_1981-2010_z_500.nc')

   # calculate geopotential height
   block.calculate_gph_from_gp(gp_name='z',
                               gp_unit='m**2 s**-2',
                               gph_name='z_height')

   # calculate z500 anomaly
   block.calc_anom('z_height', window=31)

   # Finally, calculate blocking
   block.run_contrack('anom', 
                      threshold=150,
                      overlap=0.5,
                      persistence=5)

   # plotting blocking frequency for winter over Northern Hemisphere

   import matplotlib.pyplot as plt
   import cartopy.crs as ccrs

   fig, ax = plt.subplots(figsize=(7, 5), subplot_kw={'projection': ccrs.NorthPolarStereo()})
   (xr.where(block['flag']>1,1,0).sum(dim='time')/block.ntime*100).plot(levels=np.arange(2,21,2), cmap='Oranges', extend = 'max', transform=ccrs.PlateCarree())
   (xr.where(block['flag']>1,1,0).sum(dim='time')/block.ntime*100).plot.contour(colors='grey', linewidths=0.8, levels=np.arange(2,21,2), transform=ccrs.PlateCarree())
   ax.set_extent([-180, 180, 30, 90], crs=ccrs.PlateCarree()); ax.coastlines();
   plt.show()

.. image:: docs/era5_blockingfreq_DJF.png
  :width: 400
  :alt: Mean blocking frequency for Winter 1981 - 2010 [color shading, %]

  Mean blocking frequency for Winter 1981 - 2010 [color shading, %]
