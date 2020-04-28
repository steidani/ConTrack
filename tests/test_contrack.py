#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 13:51:22 2020

@author: steidani
"""


# =======
# import packages
import sys
sys.path.append('../src/contrack')

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
    logger.warning("Matplotlib and/or Caropy is not installed in your python environment. Xarray Dataset plotting functions will not work.")

#%%
# =======
# read data

# era
ds_era = xr.open_dataset('../data/era5_2019_z_t_500.nc')
ds_era = xr.open_dataset('../data/era5_1981-2010_z_500.nc')
ds_era = xr.open_dataset('../data/era5_1981-2010_z_500.nc', chunks={'longitude': 1})
ds_era_1 =  xr.open_dataset('~/Downloads/35878605-0b92-49dc-aa63-3073c8511c45.nc')

print('ds size in GB {:0.2f}\n'.format(ds_era.nbytes / 1e9))

list(ds_era.variables)
list(ds_era.data_vars)

# plot
ax = plt.axes(projection=ccrs.PlateCarree())
ds_era['z'][58].plot.contour(colors="Green")
ds_era_1['z_NON_CDM_climatology_mean'][58].plot.contour(colors='Red')
ax.set_global(); ax.coastlines();


#%%
# =======
# dask
if ds_era['z'].chunks:
    ds_era['z'] = ds_era['z'].chunk({'longitude': -1})

# Step 1: remove leap day
ds_era = ds_era.sel(time=~((ds_era.time.dt.month == 2) & (ds_era.time.dt.day == 29)))

ds_era = ds_era.chunk({'time': 365, 'longitude': 30})

with ProgressBar():
    ds_era = ds_era.groupby('time.dayofyear').mean('time')

ds_era = ds_era.chunk({'dayofyear': None, 'longitude': 30})

# step 4: 31-day running mean
roll1 = ds_era.roll(dayofyear=150).rolling(dayofyear=31, center=True).mean()
roll2 = ds_era.rolling(dayofyear=31, center=True).mean()
sol2 = xr.concat([roll1, roll2], dim='r').mean('r')


with ProgressBar():
    results = sol2.compute()    
    #ds_era.rolling(dayofyear=31).construct('window').mean('window')
    

sol2['z'].sel(longitude=10, latitude=48).plot()

#%%
# =======
# create daily long-term climatology

# Step 1: remove leap day
ds_era = ds_era.sel(time=~((ds_era.time.dt.month == 2) & (ds_era.time.dt.day == 29)))

# Step 2: daily mean

ds_era = ds_era.resample(time='1D').mean()

# Step 4: calc geopotential height
g = 9.80665
ds_era['z'] = ds_era['z'] / g

# Step 3: daily long-term climatology over period 

ds_era['z_mean'] = ds_era.groupby('time.dayofyear').mean('time')
ds_era['z_std'] = ds_era.z.groupby('time.dayofyear').std('time')

# drop z
ds_era = ds_era.drop_vars(['z','time'])

# step 4: 31-day running mean
roll1 = ds_era.roll(dayofyear=150).rolling(dayofyear=31, center=True).mean()
roll2 = ds_era.rolling(dayofyear=31, center=True).mean()
sol2 = xr.concat([roll1, roll2], dim='r').mean('r')







# rename variable
#sol2 = sol2.rename({'z': 'z_clim'})
attrs={'units': 'm',
       'long_name': 'Geopotential Height',
       'standard_name': 'geopotential height',
}
sol2.z_mean.attrs = attrs
sol2.z_std.attrs = attrs
# store as netcdf
sol2.to_netcdf('data/era5_1981_2010_z_clim.nc')

# test load
sol2 = xr.open_dataset('data/era5_1981_2010_z_clim.nc')

# plot test
sol2['z_mean'].sel(longitude=10, latitude=48).plot()
ds_era['z'].sel(longitude=10, latitude=48).plot()
clim.sel(longitude=10, latitude=48).plot()



# step 5: calcualte anom
ds_era['anom'] = (ds_era['z'].groupby('time.dayofyear') - sol2['z_mean']*g) 
ds_era['anom'] = ds_era['anom'].groupby('time.dayofyear') / (sol2['z_std']*g)

# smooth
ds_era['anom'] = ds_era['anom'].rolling(time=3, center=True).mean().fillna(
            ds_era['anom'][-3:].mean(dim='time')    
            )


for ii in range(170,180):
    ax = plt.axes(projection=ccrs.PlateCarree())
    ds_era['anom'][ii].plot(levels=np.arange(-3000,3010,100))
    #ds_era['z_mean'][ii].plot.contour(colors="green")
    sol2['z_mean'][ii].plot.contour(colors='red')
    #sol2['z_std'][ii].plot()
    #ds_era['flag'][ii].plot.contour(colors='black')
    #ax.set_extent([-30,30,30,70]); ax.coastlines();
    ax.set_global(); ax.coastlines();
    plt.show()
    
    
# step 6: calculate threshold 

ds_era['threshold'] = ds_era['anom'].groupby('time.season').quantile([0.90], dim='time')


ds_era['flag'] = xr.where((ds_era['anom'].groupby('time.season') - ds_era['threshold'][:,:,:,0]) > 0, 1, 0)
ds_era['flag'] = xr.where(ds_era['anom'] >= 1.500, 1, 0)


# plot
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_global(); ax.coastlines();
#ds_era['threshold'][0].plot()
ds_era['flag'][300].plot.contour(colors='red')
ds_era['z'][300].plot.contour()
#ds_era['anom'].std(dim='dayofyear').plot()

# sum
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_global(); ax.coastlines();
(ds_era['flag'].sum(dim='time')/3.65).plot()


#%%
# =======
# get resolution

var = ds_era['time'].to_index()
dime = np.unique((var[1:] - var[:-1]).astype('timedelta64[D]'))

var = ds_era['latitude'].data
dlat = abs(np.unique((var[1:] - var[:-1])))

var = ds_era['longitude'].data
dlon = np.unique((var[1:] - var[:-1]))
   
#%%
# test running mean
a = ds_era['z'].rolling(time=60, center=True).mean()

# test std of anom
ax = plt.axes(projection=ccrs.PlateCarree())
block['anom'].std(dim='time').plot(levels=np.arange(50, 160,10))
ax.set_global(); ax.coastlines();

std_anom = block['anom'].std(dim='time')
std_anom = std_anom.where(std_anom > 100, 100)

std_anom.plot()
ax.set_global(); ax.coastlines();

ds_era['test'] = xr.where((block['anom'] / std_anom) > 1, 1, 0)


#%%
# scaling anomalies with coriolis param
"""Also
note that following Sausen et al. [1995] and Barriopedro et al. [2010], and unlike Dole and Gordon [1983,
equation (1)] and Dunn-Sigouin and Son [2013], we do not scale the anomalies with the Coriolis parameter. We have found that when the mean states are different, this scaling produces a bias favoring cases
with equatorward shifted blocks and may result in missing some of the anomalies that truly affect the
zonal flow."""

#%%
# =======
# test label
from scipy import ndimage

a = np.array([[[1,1,0,0,0,0],[0,0,1,0,0,0],[0,0,1,0,1,0],[0,0,0,0,0,0]],[[1,0,1,1,1,1],[1,0,0,0,0,0],[0,0,0,1,0,1],[0,0,0,0,1,0]]])
structure = np.array([[[0, 0, 0], [0,1,0], [0,0,0]],[[1, 1, 1], [1,1,1], [1,1,1]],[[0, 0, 0], [0,1,0], [0,0,0]]])

ndimage.label(a, structure)

#%%
# =======


# STEP 1: Smooth 2d running mean
ds_era['anom'] = block['anom'].rolling(time=2, center=True).mean().fillna(
            block['anom'][-2:].mean(dim='time')    
            )

ax = plt.axes(projection=ccrs.PlateCarree())
ds_era['anom'].std(dim='time').plot(levels=np.arange(100, 160,10))
ax.set_global(); ax.coastlines();

# STEP 2: identify contours
def _apply_threshold(field, th, cond):
    if cond == "ge":
        return np.where(field >= th, 1, 0)
    if cond == "le":
        return np.where(field <= th, 1, 0)   
test = _apply_threshold(field=block['std_anom'].data, th=-1.5, cond="le")

ds_era['test'] = xr.where((ds_era['anom'] - ds_era['anom'].quantile([0.90], dim='time')[0]) > 0, 1, 0)
ds_era['test'] = xr.where(ds_era['anom'] >= 150, 1, 0)

#plt.contourf(block['anom'].data[11], cmap="RdBu"), plt.colorbar()

for ii in range(1,30):
    ax = plt.axes(projection=ccrs.PlateCarree())
    ds_era['anom'][ii].plot.contourf()
    ds_era['z'][ii].plot.contour(colors='gray')
    ds_era['test'][ii].plot.contour(colors='green')
    ax.set_global(); ax.coastlines();
    plt.show()
    
    
ax = plt.axes(projection=ccrs.PlateCarree())
(ds_era['test'].sum(dim='time')/3.65).plot()
block['z_height'].mean(dim='time').plot.contour()
ax.set_global(); ax.coastlines();

# STEP 3: individual contours
# A) using ndimage
from scipy import ndimage

# loop over each time step
structure = np.ones((3,) * 2)
arr = []
for tt in range(0,ds_era['test'].data.shape[0]):
    print(tt)
    labeled_array, num_features = ndimage.label(ds_era['test'].data[tt])
    # periodic boundry
    for y in range(labeled_array.shape[0]):
        if labeled_array[y, 0] > 0 and labeled_array[y, -1] > 0:
            labeled_array[labeled_array == labeled_array[y, -1]] = labeled_array[y, 0]
    arr.append(labeled_array)
arr = np.asarray(arr).squeeze()  

# do it over all timesteps
# structure = np.ones((3,) * 3) # diagonal along all axis
structure = np.array([[[0, 0, 0], [0,1,0], [0,0,0]],[[1, 1, 1], [1,1,1], [1,1,1]],[[0, 0, 0], [0,1,0], [0,0,0]]]) # overlap along time axis, diagonal along lat lon
arr, num_features = ndimage.label(ds_era['test'].data, structure=structure)
# periodic boundry: allow features to cross date border
for tt in range(arr.shape[0]):
    for y in range(arr.shape[1]):
        if arr[tt, y, 0] > 0 and arr[tt, y, -1] > 0:
            arr[tt][arr[tt] == arr[tt, y, -1]] = arr[tt, y, 0]


len(np.unique(arr))
for tt in range(0,11):
    plt.contourf(arr[tt], cmap='jet'), plt.colorbar()
    plt.show()

# remove values below 20°
arr[:,70:110,:] = 0

# STEP 4: overlapping: 50% between successive days
import time
lat = ds_era['latitude'].data
weight_lat = np.cos(lat*np.pi/180)
overlap = 0.5 # in %
#sinlat = (np.sin(np.deg2rad(45))/np.sin(np.deg2rad(lat)))

# OLD SLOW
# loop over time
start_time = time.time()
for tt in range(300,350):# range(1,arr.shape[0]-1):
    print(tt)
    # loop over individual contours
    for label in np.trim_zeros(np.unique(arr[tt])):
        
        areacon  = 0.
        areaover_forward  = 0.
        areaover_backward = 0.
        fraction_backward = 0.
        fraction_forward = 0.
        
        for ii in range(arr.shape[2]):
            for jj in range(arr.shape[1]):
                
                if (arr[tt,jj,ii] == label):
                    areacon = areacon + (111 * dlat * (111 * dlon * weight_lat[jj]))
                if (arr[tt,jj,ii] == label) and (arr[tt+1,jj,ii] >= 1):   
                    areaover_forward = areaover_forward + (111 * dlat * (111 * dlon * weight_lat[jj]))
                if (arr[tt,jj,ii] == label) and (arr[tt-1,jj,ii] >= 1):    
                    areaover_backward = areaover_backward + (111 * dlat * (111 * dlon * weight_lat[jj]))
        
        fraction_backward = 1 / areacon * areaover_backward
        fraction_forward = 1 / areacon * areaover_forward 
        
        for ii in range(arr.shape[2]):
            for jj in range(arr.shape[1]):
                if (((fraction_backward < overlap) or (fraction_forward < overlap)) and (arr[tt,jj,ii] == label)):            
                    arr[tt,jj,ii] = 0.
print("--- %s seconds ---" % (time.time() - start_time))

# NEW FAST
weight_grid = np.ones((181, 360)) * np.array((111 * dlat * 111 * dlon * weight_lat)).astype(np.float32)[:, None]

start_time = time.time()
for tt in range(1,arr.shape[0]-1):
    print(tt)   
    # loop over individual contours
    slices = ndimage.find_objects(arr[tt])
    label = 0
    for slice_ in slices:
        label = label+1
        #if slice_ is None:
            # no feature with this flag
        #    continue
        areacon = np.sum(weight_grid[slice_][arr[tt][slice_] == label])
        areaover_forward = np.sum(weight_grid[slice_][(arr[tt][slice_] == label) & (arr[tt+1][slice_] >= 1)])
        areaover_backward = np.sum(weight_grid[slice_][(arr[tt][slice_] == label) & (arr[tt-1][slice_] >= 1)])

        fraction_backward = (1 / areacon) * areaover_backward
        fraction_forward = (1 / areacon) * areaover_forward 
        
        #if (fraction_backward < overlap and fraction_backward != 0) or (fraction_forward < overlap): # and any(i > 0 for i in arr[tt-1][slice_].flatten()):
        #    arr[tt][slice_][(arr[tt][slice_] == label)] = 0.
     
        # middle
        if fraction_backward != 0 and fraction_forward != 0:
            if (fraction_backward < overlap) or (fraction_forward < overlap):
                arr[tt][slice_][(arr[tt][slice_] == label)] = 0.
        # decay
        if fraction_backward != 0 and fraction_forward == 0:
            if (fraction_backward < overlap):
                arr[tt][slice_][(arr[tt][slice_] == label)] = 0.
        # onset
        if fraction_backward == 0 and fraction_forward != 0:        
            if (fraction_forward < overlap):
                arr[tt][slice_][(arr[tt][slice_] == label)] = 0.
        
        
        #elif (fraction_forward < overlap):
        #    arr[tt][slice_][(arr[tt][slice_] == label)] = 0.
print("--- %s seconds ---" % (time.time() - start_time))            

#%%
# =======
# Identify individual contour areas along time

# do it over all timesteps
# structure = np.ones((3,) * 3) # diagonal along all axis
structure = np.array([[[0, 0, 0], [0,1,0], [0,0,0]],[[1, 1, 1], [1,1,1], [1,1,1]],[[0, 0, 0], [0,1,0], [0,0,0]]]) # overlap along time axis, diagonal along lat lon
arr2, num_features = ndimage.label(arr, structure = structure)
# periodic boundry: allow features to cross date border
for tt in range(arr2.shape[0]):
    for y in range(arr2.shape[1]):
        if arr2[tt, y, 0] > 0 and arr2[tt, y, -1] > 0:
            arr2[tt][arr2[tt] == arr2[tt, y, -1]] = arr2[tt, y, 0]

len(np.unique(arr2))
for tt in range(0,30):
    plt.contourf(arr2[tt], cmap='jet', levels=np.arange(1,50,1)), plt.colorbar()
    plt.show()



# =======
# termporal perisistence

persistance = 5 # days

# loop trough objects
label = 0
for slice_ in ndimage.find_objects(arr2):
    label = label+1
    print(str(label) + ": " + str(slice_))
    if slice_ is None:
        #no feature with this flag
        continue
    if (slice_[0].stop - slice_[0].start) < persistance:
        arr2[slice_][(arr2[slice_] == label)] = 0.

#%%
# =======
# calc frequency
arr2[arr2>=1] = 1
freq = np.sum(arr2, axis=0)/arr.shape[0]
plt.contourf(freq*100), plt.colorbar()

ax = plt.axes(projection=ccrs.PlateCarree())
(ds_era['test'].sum(dim='time')/3.65).plot()
block['z_height'].mean(dim='time').plot.contour()
ax.set_global(); ax.coastlines();


ds_era = ds_era.assign(arr2=lambda ds_era: ds_era.test * 0 + arr2)

ax = plt.axes(projection=ccrs.PlateCarree())
(ds_era['arr2'].sum(dim='time')/365).plot()
ax.set_global(); ax.coastlines();

#%%
# =======
# C) 90th quantile anaomaly
quantile = anom.quantile([0.90], dim='time')

ax = plt.axes(projection=ccrs.PlateCarree())
anom_roll[25].plot(ax=ax, transform=ccrs.PlateCarree())
(anom_roll[25] - quantile[0]).plot.contour(levels=[0], ax=ax, transform=ccrs.PlateCarree())
ax.set_global(); ax.coastlines();

#%%
# =============================================================================
# blocking class testing
# =============================================================================

from contrack import contrack

# initiate blocking instance
block = contrack()
# read ERA5
block.read('../data/era5_2019_z_t_500.nc')

# calculate geopotential height
block.calculate_gph_from_gp()

# calculate clim
clim = block.calc_clim('z')

# calculate z500 anomaly
block.calc_anom('z_height', window=61)

# plot z500 anomaly on 2 Sep 2019 (Hurricane Dorian)
ax = plt.axes(projection=ccrs.PlateCarree())
block['anom'].sel(time='2019-09-5').plot(ax=ax, transform=ccrs.PlateCarree())
ax.set_extent([-90, -60, 20, 50], crs=ccrs.PlateCarree())
ax.coastlines()

# plot z500 anomaly on 29 Jan 2019 (US Cold Spell)
for ii in range(20,35):
    ax = plt.axes(projection=ccrs.PlateCarree())
    block['anom'].isel(time=ii).plot(ax=ax, transform=ccrs.PlateCarree())
    ax.set_extent([-180, -60, 20, 90], crs=ccrs.PlateCarree())
    ax.coastlines()
    plt.show()


block.read_xarray(a)
block = blocking(a)
block.__len__()
block.__getattr__('z')
block.variables
block.dimensions
block.set_up()

print(block._time_name, block._longitude_name, block._latitude_name, block._variable_name)
block.dataset
block['z']
block.ntime
block.grid
block._get_name_time()
block._get_name_latitude()
block._get_name_longitude()
block._get_name_variable()
block._dtime
block._dlat
block._dlon

# calc geopot height
block.calculate_gph_from_gp()
block['z_height'].isel(time=1).plot()
block['z'].isel(time=1).plot()

# calc anom
block.calc_anom(std_dev=True)
block['anom'].isel(time=60).plot()
block['std_anom'].isel(time=60).plot()
block['z_height'].isel(time=60).plot()

ax = plt.axes(projection=ccrs.PlateCarree())
#block['z_height'].sel(time='2019-01-29').plot(ax=ax, transform=ccrs.PlateCarree())
block['std_anom'].sel(time='2019-09-2')[0].plot(ax=ax, transform=ccrs.PlateCarree())
block['std_anom'].sel(time='2019-09-2')[0].plot.contour(ax=ax, levels=[-4, -2], cmap='Greys', transform=ccrs.PlateCarree())
ax.set_extent([-90, -60, 20, 50], crs=ccrs.PlateCarree())
ax.coastlines()

# calc mean
var_mean = block.calc_mean(variable="z")
var_mean.plot()
block1 = blocking(filename='data/era5_2019_z500.nc')
block2 = blocking(ds=ds_era)

blocking.set_raise_amt(1.05)
print(blocking.apply_raise)
print(block.raise_amt)
isinstance(block, blocking) # True
my_date = datetime.date(2016, 7, 11)
print(blocking.is_workday(my_date))

#%%

def calc_anom(self, 
                  variable="",
                  window=60,
                  std_dev=False
    ):
        """
        Creates a new variable with name "anom" from variable.
        Anomalies are computed for each grid point and time step as the departure from the "window"-day running mean.

        Parameters
        ----------
            variable : string, optional
                Input variable. The default is z_height.
            window : float, optional
                runnwing mean window. The default is 60 days.
            std_dev : bool, optional
                if True calculate also the standardized anomaly

        Returns
        -------
            array: float
                Anomalie field

        """
        if variable not in self._ds.variables:
            logger.warning(
                "\n'{}' not found.\n"
                "Available fields: {}".format(
                variable, ", ".join(self.variables))
            )
            return None
        

        # step 1: running mean and fill nans at start/end with mean over last window-day timesteps
        clim = self._ds[variable].rolling(time=window, center=True).mean().fillna(
            self._ds[variable][-window:].mean(dim='time')    
            )
        
        if not std_dev:            
            # step 3: calculate and create new variable anomaly     
            self._ds['anom'] = xr.Variable(
                self._ds.variables[variable].dims,
                self._ds.variables[variable].data - clim.data,
                attrs={
                    'units': self._ds[variable].attrs['units'],
                    'long_name': self._ds[variable].attrs['long_name'] + ' Anomaly',
                    'standard_name': self._ds[variable].attrs['long_name'] + ' anomaly',
                    'history': 'Calculated from {} with running mean={} days'.format(variable, window)}
            )
            logger.info('Calculating Anomaly... DONE')
        
        if std_dev:
            # step 4: calculate and create new variable standardized anomaly
            clim_std = self._ds[variable].rolling(time=window, center=True).std()
            fill_nans = self._ds[variable][-window:].std(dim='time')
            clim_std = clim_std.fillna(fill_nans)
            
            self._ds['std_anom'] = xr.Variable(
                self._ds.variables[variable].dims,
                xr.apply_ufunc(
                    lambda x, m, s: (x - m) / s,
                    self._ds.variables[variable].data,
                    clim.data,
                    clim_std.data,
                ),
                attrs={
                    'units': '', # standarized variable without unit
                    'long_name': self._ds[variable].attrs['long_name'] + ' Standardized Anomaly',
                    'standard_name': self._ds[variable].attrs['long_name'] + ' standardized anomaly',
                    'history': 'Calculated from {} with running mean={} days'.format(variable, window)}
            )
            logger.info('Calculating Standardized Anomaly... DONE')