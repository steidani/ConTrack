#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 13:51:22 2020

@author: steidani
"""


# =======
# import packages
#import sys
#sys.path.append('../src/contrack')

# import contrack
from contrack import contrack

# data analysis
import numpy as np
import xarray as xr
import datetime
from numpy.core import datetime64

# logs
import logging

# parallel computing
import dask
from dask.diagnostics import ProgressBar
import time

# plotting
try:
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
except:
    print('not installed')

#%%
# =============================================================================
# blocking class testing
# =============================================================================

from contrack import contrack

# initiate blocking instance
block = contrack()
# read ERA5
#block.read('data/era5_2016_z_500.nc')
#block.read('data/VAPVA2016_lowres')
block.read('data/cesm/VAPV_1990.nc')
block.read('data/cesm/Vanom1990')


# Step 1: clean data

# daily mean
block.ds = block.ds.resample(time='1D', keep_attrs=True).mean(keep_attrs=True)
# select only wanted month
#block.ds = block.ds.sel(time=block.ds.time.dt.month.isin([1, 2, 12]))
# remove leap day
#block.ds = block.ds.sel(time=~((block.ds.time.dt.month == 2) & (block.ds.time.dt.day == 29)))
# chunk
#block.ds = block.ds.chunk({'time': 365, 'longitude': 10})
#block.z = block.z.chunk({'time': 365, 'longitude': 10})

# set up
block.set_up(force=True)

# calculate geopotential height
block.calculate_gph_from_gp()

#block.z_height.chunks
#block.ds = block.ds.chunk({'time': None})
# block.ds = block.ds.drop_vars('z_height')

# calculate clim
#clim = block.calc_clim('z')
clim = xr.open_dataset('data/cesm/VAPV_clim.nc')
clim = clim.rename({'time': 'month'})
clim_mean = clim.z_mean


# calculate z500 anomaly
block.calc_anom('VAPV', window=31, smooth=8)
block.calc_anom('z_height', window=31, clim='data/era5_1981_2010_z_clim.nc')
block.calc_anom('VAPV', window=1, smooth=8, groupby='month', clim=clim.VAPV)

# block.ds.to_netcdf('data/anom_1981_2010.nc')

block = contrack()
block.read('data/anom_1981_2010.nc')

# calculate blocking
block.run_contrack(variable='anom', 
                  threshold=150,
                  gorl='>=',
                  overlap=0.5,
                  persistence=5,
                  twosided=True)

#block = contrack()
#block.read('data/cesm/BLOCKS1990.nc')

# life cycle analysis
test = block.run_lifecycle(flag='flag', variable='anom')
test.to_csv('data/test.csv', index=False)


# =======
# PLOTTING Block Track
f, ax = plt.subplots(1, 1, figsize=(7,5), subplot_kw=dict(projection=ccrs.NorthPolarStereo()))

# =======
# add coastlines, 
ax.coastlines()

# ======= 
# add gridlines and gridticks
#ax.gridlines(color='black', alpha=0.2, linestyle='--', ylocs=np.arange(-90, 91, 30), xlocs=np.arange(-180, 181, 60))
ax.set_extent([-180, 180, 30, 90], crs=ccrs.PlateCarree())
      

#need to split each blocking track due to longitude wrapping (jumping at basemap edge) 
for bid in np.unique(np.asarray(test['Flag'])): #select blocking id in year yy and seas ii        
    lons = np.asarray(test['Longitude'].iloc[np.where(test['Flag']==bid)])
    lats = np.asarray(test['Latitude'].iloc[np.where(test['Flag']==bid)])
    
    # cosmetic: sometimes there is a gap near dateline where split: 
    lons[lons >= 355] = 359.9
    lons[lons <= 3] = 0.1
    segment = np.vstack((lons,lats))  
    
    #move longitude into the map region and split if longitude jumps by more than "threshold"
    lon0 = 0 #center of map
    bleft = lon0-0.                                                                            
    bright = lon0+360
    segment[0,segment[0]> bright] -= 360                                                                 
    segment[0,segment[0]< bleft]  += 360
    threshold = 180  # CHANGE HERE                                                                                    
    isplit = np.nonzero(np.abs(np.diff(segment[0])) > threshold)[0]                                                                                         
    subsegs = np.split(segment,isplit+1,axis=+1)

    
    #plot the tracks
    for seg in subsegs:                                                                                  
        x,y = seg[0],seg[1]                                                                          
        ax.plot(x ,y,c = 'm',linewidth=1, transform=ccrs.PlateCarree())  
    #plot the starting points
    ax.scatter(lons[0],lats[0],s=11,c='m', zorder=10, edgecolor='black', transform=ccrs.PlateCarree())  
plt.savefig('data/fig/cesm_blocking_track.png', dpi=300)


    
# plot frequency
fig, ax = plt.subplots(figsize=(7, 5), subplot_kw={'projection': ccrs.NorthPolarStereo()})
h2 = (xr.where(block['flag']>1,1,0).sum(dim='time')/block.ntime*100).plot(levels=np.arange(2,18,2), cmap='Oranges', extend = 'max', transform=ccrs.PlateCarree())
(xr.where(block['flag']>1,1,0).sum(dim='time')/block.ntime*100).plot.contour(colors='grey', linewidths=0.8, levels=np.arange(2,18,2), transform=ccrs.PlateCarree())
ax.set_extent([-180, 180, 30, 90], crs=ccrs.PlateCarree()); ax.coastlines();
#ax.set_title('DJF 1981 - 2010')
#fig_cbar = h2.colorbar
#fig_cbar.ax.set_ylabel("blocking frequency [%]")
plt.savefig('data/fig/era5_blockingfreq_DJF.png', dpi=300)