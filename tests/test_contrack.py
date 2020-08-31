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
    logger.warning("Matplotlib and/or Caropy is not installed in your python environment. Xarray Dataset plotting functions will not work.")

#%%
# =======
# read data

# era
ds_era = xr.open_dataset('data/era5_2019_z_t_500.nc')
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

# ----- regrid clim

clim = sol2.z_mean.reindex(latitude=np.arange(90,-90.5,-0.5), longitude=np.arange(0,359.5,0.5), method='nearest')

ax = plt.axes(projection=ccrs.PlateCarree())
clim[1].plot(levels=np.arange(5500,5800,1))
#sol2.z_mean[1].plot(levels=np.arange(5500,5800,1))
ax.set_extent([-10,10,30,40]); ax.coastlines();

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

ds_era['anom'] = block['anom']

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
structure = np.array([[[0, 0, 0], [0,0,0], [0,0,0]],[[1, 1, 1], [1,1,1], [1,1,1]],[[0, 0, 0], [0,0,0], [0,0,0]]]) # overlap along time axis, diagonal along lat lon
arr, num_features = ndimage.label(ds_era['test'].data, structure=structure)
# periodic boundry: allow features to cross date border
for tt in range(arr.shape[0]):
    for y in range(arr.shape[1]):
        if arr[tt, y, 0] > 0 and arr[tt, y, -1] > 0:
            arr[tt][arr[tt] == arr[tt, y, -1]] = arr[tt, y, 0]


len(np.unique(arr))
for tt in range(0,11):
    plt.contourf(arr[tt], cmap='jet', levels=np.arange(1,30,1)), plt.colorbar()
    plt.show()

# remove values below 20°
arr[:,70:110,:] = 0

# STEP 4: overlapping: 50% between successive days
import time
lat = block['latitude'].data
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
    #print(tt)   
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
        #if fraction_backward != 0 and fraction_forward != 0:
        #    if (fraction_backward < overlap) or (fraction_forward < overlap):
        #        arr[tt][slice_][(arr[tt][slice_] == label)] = 0.
        # decay
        #if fraction_backward != 0 and fraction_forward == 0:
        #    if (fraction_backward < overlap):
        #        arr[tt][slice_][(arr[tt][slice_] == label)] = 0.
        # onset
        #if fraction_backward == 0 and fraction_forward != 0:        
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
block.read('data/era5_2016_z_500.nc')
# block.read('data/VAPVA2016_lowres')

# clean data
# Step 1: remove leap day
block.ds = block.ds.sel(time=~((block.ds.time.dt.month == 2) & (block.ds.time.dt.day == 29)))

# Step 2: daily mean
block.ds = block.ds.resample(time='1D', keep_attrs=True).mean(keep_attrs=True)

# select only wanted month
block.ds = block.ds.sel(time=block.ds.time.dt.month.isin([1, 2, 12]))

block.ds = block.ds.chunk({'time': 365, 'longitude': 10})

block.z = block.z.chunk({'time': 365, 'longitude': 10})

# calculate geopotential height
block.calculate_gph_from_gp()

block.z_height.chunks
block.ds = block.ds.chunk({'time': None})

# block.ds = block.ds.drop_vars('z_height')

# calculate clim
#clim = block.calc_clim('z')
clim = xr.open_dataset('data/era5_1981_2010_z_clim.nc')
clim_mean = clim.z_mean


# calculate z500 anomaly
block.calc_anom('z_height', window=31, smooth=2)
block.calc_anom('z_height', window=31, clim='data/era5_1981_2010_z_clim.nc')
block.calc_anom('z_height', window=31, clim=clim_mean)
# block.ds.to_netcdf('data/anom_1981_2010.nc')


# calculate blocking
block.run_contrack(variable='anom', 
                  threshold=150,
                  gorl='>=',
                  overlap=0.5,
                  persistence=5,
                  twosided=False)

test = block.run_lifecycle(flag='flag', variable='anom')

block.flag.to_netcdf('data/test.nc')
test = xr.open_dataset('data/test.nc')

# plot z500 anomaly on 2 Sep 2019 (Hurricane Dorian)
ax = plt.axes(projection=ccrs.PlateCarree())
block['anom'].sel(time='2016-10-08').plot(ax=ax, transform=ccrs.PlateCarree())
block['flag'].sel(time='2016-10-08').plot.contour(ax=ax, transform=ccrs.PlateCarree())
ax.set_extent([-120, 60, 30, 90], crs=ccrs.PlateCarree())
ax.coastlines()

# plot z500 anomaly on 29 Jan 2019 (US Cold Spell)
start_date = datetime.date(2016, 10, 1)
end_date = datetime.date(2016, 10, 5)

ii = start_date
while ii <= end_date:
    ax = plt.axes(projection=ccrs.PlateCarree())
    block['anom'].sel(time=ii).plot(ax=ax, transform=ccrs.PlateCarree())
    block['flag'].sel(time=ii).plot.contour(ax=ax,transform=ccrs.PlateCarree())
    ax.coastlines()
    plt.show()
    ii = ii+datetime.timedelta(1)
    
# plot frequency
fig, ax = plt.subplots(figsize=(7, 5), subplot_kw={'projection': ccrs.NorthPolarStereo()})
h2 = (xr.where(block['flag']>1,1,0).sum(dim='time')/block.ntime*100).plot(levels=np.arange(2,18,2), cmap='Oranges', extend = 'max', transform=ccrs.PlateCarree())
(xr.where(block['flag']>1,1,0).sum(dim='time')/block.ntime*100).plot.contour(colors='grey', linewidths=0.8, levels=np.arange(2,18,2), transform=ccrs.PlateCarree())
ax.set_extent([-180, 180, 30, 90], crs=ccrs.PlateCarree()); ax.coastlines();
#ax.set_title('DJF 1981 - 2010')
#fig_cbar = h2.colorbar
#fig_cbar.ax.set_ylabel("blocking frequency [%]")


plt.savefig('data/fig/era5_blockingfreq_DJF.png', dpi=300)


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
var_mean.sel(expver=5).plot()
block1 = blocking(filename='data/era5_2019_z500.nc')
block2 = blocking(ds=ds_era)

fig, ax = plt.subplots(figsize=(7, 5), subplot_kw={'projection': ccrs.NorthPolarStereo()})
var_mean.sel(expver=5).plot.contour(transform=ccrs.PlateCarree())
ax.set_extent([-180, 180, 30, 90], crs=ccrs.PlateCarree()); ax.coastlines();

blocking.set_raise_amt(1.05)
print(blocking.apply_raise)
print(block.raise_amt)
isinstance(block, blocking) # True
my_date = datetime.date(2016, 7, 11)
print(blocking.is_workday(my_date))



#%%

def blocks_lifecycle():
    """
    Blocking life cycle for individual blocking events:
    
    Read Blocking Flag (output of caltrack)
    Calculate blocking statistics at every timestep and write metrics to csv file.
    
    Input: FLAGxxxx.nc; APVanomxxxx; CLMPVmm.nc
    Output: xxxx.csv: ID, Date [yyyymmdd_hh], com-Latitude [1-361], com-Longitude [1-721], Size [m^2], Strength [PVU], VAPV-Strength [PVU],
            with com = center of mass
                      
    """
    
    ##### functions ######
    # great circle distance
    def dist(lon1,lat1,lon2,lat2):
        re=6371.    # mean earth radius
        erg=np.sin(np.deg2rad(lat1))*np.sin(np.deg2rad(lat2))+np.cos(np.deg2rad(lat1))*np.cos(np.deg2rad(lat2))*np.cos(np.deg2rad(lon1-lon2))
        if (erg < -1.): erg=-1.
        if (erg > 1.): erg=1.
        distkm=re*np.arccos(erg)
        return distkm  
    bsas = [-0.27, 51.28]
    paris = [49.0083899664, 2.53844117956]
    
    test = dist(-0.27, 51.28,-73.46,40.38)
        
    ##### End Functions ####
    
    # =====
    #import modules
    import numpy as np
    from datetime import datetime, timedelta
    import netCDF4 as nc
    
    # =====
    # define path
    inpath = outpath = '/net/litho/atmosdyn/steidani/cesm112_LENS/'
    
    # define climate
    #climate = "b.e112.B20TRLENS.f09_g16.ethz." #ensemble members 001 to 035
    climate = "b.e112.BRCP85LENS.f09_g16.ethz." #ensemble members 001 to 035
    
    #define parameters
    ens_member = [str(item).zfill(3) for item in range(1, 3)]
    #years = range(1990,2001) #present
    years = range(2091,2101) #present
    
    # =====
    # here loop through ens members
    
    # Define the grid and constants
    a = 6378137.0 # equatorial radius in m
    e = 0.00669437999014 # eccentricity squared       
    #resol
    stepsperday = 4 # for 12 hourly data use 2, for 6 hourly data use 4, for 1 hourly data us 24
    dx = 1.25   # taken from inputfile
    dy = 0.94240837696335078 # taken from inputfile
    #grid
    lon = np.arange(0,360,dx)
    lat = np.arange(-90,90.5,dy)
    nx = len(lon)
    ny = len(lat)
    lons, lats = np.meshgrid(lon, lat) #for flagbuffer  
    #initiate array for length of grids in km
    latlen = np.zeros(ny)    
    lonlen = np.zeros(ny)  
    phi = np.zeros(ny)  
    

    # =====
    # Calculate the length of a degree of latitude (latlen) and longitude (lonlen) in meters
    # First we need to convert the latitudes to radians :phi = np.deg2rad(lat)
    # Use the cosine of the converted latitudes as weights for the average: weights = np.cos(phi)
    for k in range(0,ny):
        #print(lat[k])
        phi[k] = (abs(lat[k])*np.pi/180) # definiton of the latitude in radian : same function is  np.deg2rad(lat) !!!but here only pos value!!!
        latlen[k]=111132.954-559.822*np.cos(2*phi[k])+1.175*np.cos(4*phi[k]) # length of a latitude degree in m
        lonlen[k]=(np.pi*a*np.cos(phi[k]))/(180*(1-e*np.sin(phi[k])**2)**(1/2)) # length of a longitude degree in m
    
    #What is the resolution?:
    latlen*=dy
    lonlen*=dx
      
    # ======    
    #loop through ensembe members
    for kk in ens_member:
        print('####### ensemble member ' +  kk + ' #######: ')
        # =====
        # loop through years
        for yy in years: 
            print('#### year %d ####' % yy)
                  
            #initialize wanted variables!!!!!!
            block_id = []
            time = []
            apv_intensity = []
            vapv_intensity = []
            size = []
            com_lon = []
            com_lat = []
              
            # =====        
            # Read variable "VAPV" = vertically averaged PV
            infile = inpath + climate + kk + '/vapv/' + 'VAPV_' + str(yy) + '.nc'
            with nc.Dataset(infile) as ncf:
                #print(ncf)
                vapv = ncf.variables['VAPV'][:] #Vertically (500-150hPa) ??? not same temporal length as flagin becasue of 5 day persistent
                timevapv = ncf.variables['time'][:] #  days since refdate
                times_units = ncf.variables['time'].units #  days since refdate
                
            # Read variable "FLAG" = blocking mask
            infile = inpath + climate + kk + '/block/' + 'BLOCKS' + str(yy) + '.nc'
            with nc.Dataset(infile) as ncf:
                #print(ncf)
                flagin = ncf.variables['FLAG'][:]
                #flagbuffer = ncf.variables['FLAG'][:]
                # convert hour to datetime
                timein = ncf.variables['time'][:] #  days since refdate
                date = [nc.num2date(xx,units = times_units, calendar="noleap") for xx in timein]
                       
            # Read variable "VAPVanom" = vertically averaged PV Anomaly
            infile = inpath + climate + kk + '/anom/' + 'Vanom' + str(yy)
            with nc.Dataset(infile) as ncf:
                #print(ncf)
                anomin = ncf.variables['VAPVanom'][:] #Vertically (500-150hPa) averaged PV anomaly   ??? not same temporal length as flagin becasue of 5 day persistent
                timeanom = ncf.variables['time'][:] #  days since refdate
            
            
            # =====
            # Define -999 as NA (nan) values
            # ??? check with np.min()
            vapv[np.where(vapv==-999.99)] = np.nan
            
            # =====
            # cut anomaly and vapv file so that same length as flagin. They are longer (more timesteps) than Block-File because 5 day persistence, thus we need to cut 5*stepsperday at start and 5*stepsperday at end
            anomsteps = len(anomin)   
            start = (5*stepsperday)
            end = anomsteps-5*stepsperday
            anomin = anomin[start:end,:,:]    
            timeanom = timeanom[start:end]
            if (yy == years[0]):# here we only added 5 days after
                vapv = vapv[(5*stepsperday):,:,:] 
                timevapv = timevapv[(5*stepsperday):]
            if (yy == years[-1]): # here we only added 5 days before
                vapv = vapv[:1440,:,:] 
                timevapv = timevapv[:1440]
    
            
            # check if all input files have now same length: time must be identical for all inputs
            if len(anomin) == len(flagin) == len(vapv):
                print("inputs all have same length: continue")
            else:
                print("inputs have NOT same length: stopp")
                break 
            
            # there is no redundant 721th row (where -180° = 180°) in CESM: DONT NEED TO DO THIS
            #flagin = np.delete(flagin, (-1), axis=2); anomin = np.delete(anomin, (-1), axis=2); 
            
            # =====
            timesteps = len(flagin)              
            for ii in range(0,timesteps): # XX should be startdate of blocking: newtime ii=0 ii=XX
                print(ii, 'from', timesteps)
                
                # =====
                # current time step
                currentstep = nc.num2date(timein[ii],units = times_units, calendar="noleap")
                currentstep = currentstep.strftime('%Y%m%d_%H') #current date as string (human readable)
                
                # =====
                # get Flag ID's and remove zero
                idunique = np.unique(flagin[ii]) # if needed convert to int with .astype(int)
                idunique = idunique[idunique != 0] #remove zero
                lenid = len(idunique)
                
                #check if block in this timestep
                if lenid == 0:
                    print('No Block in this timestep...go to next timestep!')
                    # jump back to for -->  continues with the next iteration of the loop
                    continue
                
                # =====
                # loop through IDs in this time step and calculation of area, PV-strength, center of mass and VAPV-strength per ID    
                for bid in idunique:          
                    print('Block ID:', bid)
                    
                    # =====
                    # get index of grids that are blocked
                    index = np.where(flagin[ii]==bid)            
                    loni = index[1] # index of blocked lons lon[loni]
                    lati = index[0] # index of blocked lats lat[lati]
                                
                    #initialize   
                    idydir = lati # index of blocked latitudes
                    idxdir = np.empty(len(loni)) * np.nan # index of blocked longitudes (they are maybe shifted if block is split over prime meridian)
                    idarea = [] #in m2
                    idstrength = [] #strength of the anomaly
                    idvapv = [] #strength of the vapv 
                    
                    # ===== 
                    #Calculate Area, PVanom = intensity and VAPV normalized with blocking area
                    
                    #loop through lat where block is located and calculate Area, PVanom and VAPV for each blocked grid: PVanom and VAPV are multiplied by grid size
                    for jj in range(0,len(lati)):
                        idarea.append(latlen[lati[jj]]*lonlen[lati[jj]]) # grid size in m2
                        idstrength.append(latlen[lati[jj]]*lonlen[lati[jj]]*anomin[ii,lati[jj],loni[jj]]) #strength of the anomaly*grid !!! so that northerly grids are less weighted !!!
                        idvapv.append(latlen[lati[jj]]*lonlen[lati[jj]]*vapv[ii,lati[jj],loni[jj]]) #vapv*grid !!! so that northerly grids are less weighted !!!
                    
                    # get the sum PVanom and VAPV over blocking region          
                    idareasum = np.sum(idarea)
                    idstrengthsum = np.sum(idstrength)
                    idvapvsum = np.sum(idvapv)
                    
                    # normalize PVanom and VAPV with area of blocking region
                    idstrengthperarea = idstrengthsum/idareasum # Area weighted APV-Strenght
                    idvapvperarea = idvapvsum/idareasum # Area weighted climatological VAPV
                    
                    # =====
                    # get center of mass:  adjuct problem if block extends over Prime meridian = block is splitted
                    
                    # prepare 
                    ulatis = np.unique(lati) # unique latitudes where grids are blocked
                    shift_status = 0 # check if block over prime meridian = is splitted
                    # prepare array with lon that need to be shifted
                    lon_shift = np.empty(len(ulatis)) * np.nan
                    
                    # Run through all blocked latitudes and get blocked longitudes 
                    temp_loni = []
                    for jj in ulatis:
                        numbers = np.where(lati==jj) #get index of current blocked latitude
                        temp_loni.extend(loni[numbers]) # get blocked longitudes of current blocked latitude
                    
                    # find largest zonal distance between blocked grids 
                    temp_loni = np.sort(np.unique(temp_loni))
                    if len(temp_loni) > 1:
                        id_diff = np.max(np.diff(temp_loni))
                    else:
                        id_diff = 0
                    
                    # check if difference between blocking regions is larger as the world_diff
                    world_diff=(nx)-np.max(temp_loni) + np.min(temp_loni)
                    if world_diff < id_diff:
                        shift_status = 1
                        # find most western grid of blocked longitudes (western flank of block)
                        lon_shift = temp_loni[np.where(np.diff(temp_loni) == np.max(np.diff(temp_loni)))[0] + 1]
                        if len(lon_shift) > 1:
                            print("IT HAPPEND")
                            lon_shift = lon_shift[1]
                    
                    # go through each latitude and shift if needed (shift_status = 0) to get inxdir = index of blocked longitudes
                    for jj in ulatis:
                        numbers = np.where(lati==jj) #get index of current blocked latitude
                        longitudes = loni[numbers] #get blocked longitude of current blocked latitude
                        if shift_status == 1:  # Do shift, only if shift status is larger than 0 and if current latitude contains longitudes smaller than lon.shift
                            if (len(longitudes) == 1) and (longitudes < lon_shift): # If only one longitude at this latitude
                                idxdir[numbers] = longitudes + nx
                                continue #jump back to for
                            shift_these = np.where(longitudes<lon_shift) # these longitudes need to be shifted at current latitude
                            longitudes[shift_these] = longitudes[shift_these] + nx # shifting
                            idxdir[numbers] = longitudes # assigning idxdir
                            del shift_these
                        else: # Do not shift if difference around the world is larger than between IDs, or if current latitude does not contain longitudes smaller than lon_shift
                            idxdir[numbers] = longitudes      
                    # idxdir is now the index for the blocked longitudes after shift
                            
    
                    # get now index of blocked lon and lat that are weighted with PVanom
                    idsumx = np.sum(idstrength*idxdir)
                    idsumy = np.sum(idstrength*idydir) 
    
                    # if after shift com_x is > 720: !!! meaning it crossed the date line !!!
                    if np.round(np.int(idsumx/idstrengthsum)) < nx:
                        idmeanx = lon[np.round(np.int(idsumx/idstrengthsum))]# X-center of mass in °E
                    else:
                        x = np.round(np.int(idsumx/idstrengthsum)) - nx
                        idmeanx = lon[x]
    
                    idmeany = lat[np.round(np.int(idsumy/idstrengthsum))] # Y-center of mass
    
                    
                    # Getting Flag ID's, timestamp, com-lon, com-lat, size, vapv-strength, mean vapv
                    block_id.append(bid)
                    time.append(currentstep)                
                    apv_intensity.append(idstrengthperarea)
                    vapv_intensity.append(idvapvperarea)
                    size.append(idareasum)
                    com_lon.append(idmeanx)
                    com_lat.append(idmeany)
    
            #list to array
            block_id = np.asarray(block_id)
            time = np.asarray(time)
            apv_intensity = np.asarray(apv_intensity)
            vapv_intensity = np.asarray(vapv_intensity)
            size = np.asarray(size)
            com_lon = np.asarray(com_lon)
            com_lat = np.asarray(com_lat)
            
            print('Writing to File:')        
            # Writing to
            with open(outpath + climate + kk + '/data/' + 'block_lifecycle_' + str(yy) +'.txt', "w") as f:
                #write here header
                f.write('{:>5}  {:>11}  {:>7}  {:>7}  {:>16}  {:>8}  {:>8}'.format('ID','Date','lon','lat','size[m2]','apv[pvu]','vapv[pvu]'))
                f.write("\n")
                f.write('---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n')
                line = np.column_stack((block_id, time, com_lon.astype(np.object), com_lat.astype(np.object), size, apv_intensity, vapv_intensity))               
                f.write("\n".join('{:>5}  {:>11}  {:>7.2f}  {:>7.2f}  {:>16.2f}  {:>8.3f}  {:>8.3f}'.format(*x) for x in line))
        







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
        if variable not in self.ds.variables:
            logger.warning(
                "\n'{}' not found.\n"
                "Available fields: {}".format(
                variable, ", ".join(self.variables))
            )
            return None
        

        # step 1: running mean and fill nans at start/end with mean over last window-day timesteps
        clim = self.ds[variable].rolling(time=window, center=True).mean().fillna(
            self.ds[variable][-window:].mean(dim='time')    
            )
        
        if not std_dev:            
            # step 3: calculate and create new variable anomaly     
            self.ds['anom'] = xr.Variable(
                self.ds.variables[variable].dims,
                self.ds.variables[variable].data - clim.data,
                attrs={
                    'units': self.ds[variable].attrs['units'],
                    'long_name': self.ds[variable].attrs['long_name'] + ' Anomaly',
                    'standard_name': self.ds[variable].attrs['long_name'] + ' anomaly',
                    'history': 'Calculated from {} with running mean={} days'.format(variable, window)}
            )
            logger.info('Calculating Anomaly... DONE')
        
        if std_dev:
            # step 4: calculate and create new variable standardized anomaly
            clim_std = self.ds[variable].rolling(time=window, center=True).std()
            fill_nans = self.ds[variable][-window:].std(dim='time')
            clim_std = clim_std.fillna(fill_nans)
            
            self.ds['std_anom'] = xr.Variable(
                self.ds.variables[variable].dims,
                xr.apply_ufunc(
                    lambda x, m, s: (x - m) / s,
                    self.ds.variables[variable].data,
                    clim.data,
                    clim_std.data,
                ),
                attrs={
                    'units': '', # standarized variable without unit
                    'long_name': self.ds[variable].attrs['long_name'] + ' Standardized Anomaly',
                    'standard_name': self.ds[variable].attrs['long_name'] + ' standardized anomaly',
                    'history': 'Calculated from {} with running mean={} days'.format(variable, window)}
            )
            logger.info('Calculating Standardized Anomaly... DONE')