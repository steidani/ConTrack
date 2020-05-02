#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 17:12:55 2020

@author: steidani (Daniel Steinfeld; daniel.steinfeld@alumni.ethz.ch)


TO DO
    - calculate clim: take precalculated era5 clim
        - use dask for out of memory problem
    - smooth anomaly field with 2 day running mean
    . detection: larger OR smaller as threshold
"""

# =======
# import packages

# data
import numpy as np
import xarray as xr
from scipy import ndimage
import datetime
from numpy.core import datetime64

# logs
import logging
logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

# parallel computing
try:
    import dask
except:
    logger.warning("Dask is not installed in your python environment. Xarray Dataset parallel computing will not work.")


# plotting
try:
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
except:
    logger.warning("Matplotlib and/or Cartopy is not installed in your python environment. Xarray Dataset plotting functions will not work.")


# =============================================================================
# blocking class
# =============================================================================

class contrack(object):
    """
    contrack class
    Author : Daniel Steinfeld, ETH Zurich , 2020
    """

    # number of block instances initiated
    num_of_blocks = 0

    def __init__(self, filename="", ds=None, **kwargs):
        """The constructor for contrack class. Initialize a contrack instance.
        
        If filename is given, try to load it directly.
        Arguments to the load function can be passed as key=value argument.

        Parameters
        ----------
            filename : string
                Datapath + filename.nc
            ds : dataset
                xarray dataset

        """
        if not filename:
            if ds is None:
                self._ds = None
            else:
                self._ds = ds
            return
        
        try:
            self._ds = None
            self.read(filename, **kwargs)
        except (OSError, IOError, RuntimeError):
            try:
                self.read(filename, **kwargs)
            except Exception:
                raise IOError("Unkown fileformat. Known formats " "are netcdf.")

        contrack.num_of_blocks += 1
    
    def __repr__(self):
        try:
            string = "\
            Xarray dataset with {} time steps. \n\
            Available fields: {}".format(
                self.ntime, ", ".join(self.variables)
            )
        except AttributeError:
            # Assume it's an empty Blocking()
            string = "\
            Empty contrack container.\n\
            Hint: use read() to load data."
        return string

    def __str__(self):
        return 'Class {}: \n{}'.format(self.__class__.__name__, self._ds)
  
    def __len__(self):
        return len(self._ds)
    
    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self._ds, attr)

    
    def __getitem__(self, key):
        return self._ds[key]

    @property
    def ntime(self):
        """Return the number of time steps"""
        if len(self._ds.dims) != 3:
            logger.warning(
                "\nBe careful with the dimensions, "
                "you want dims = 3 and shape:\n"
                "(latitude, longitude, time)"
            )
            return None
        return self._ds.dims['time']

    @property
    def variables(self):
        """Return the names of the variables"""
        return list(self._ds.data_vars)
    
    @property
    def dimensions(self):
        """Return the names of the dimensions"""
        return list(self._ds.dims)
    
    @property
    def grid(self):
        """Return the number of longitude and latitude grid"""
        if len(self._ds.dims) != 3:
            logger.warning(
                "\nBe careful with the dimensions, "
                "you want dims = 3 and shape:\n"
                "(latitude, longitude, time)"
            )
            return None
        string = "\
        latitude: {} \n\
        longitude: {}".format(
            self._ds.dims['latitude'], self._ds.dims['longitude']
        ) 
        print(string)

    @property
    def dataset(self):
        """Return the dataset"""
        return self._ds

# ----------------------------------------------------------------------------
# Read / Import data
    
    def read(self, filename, **kwargs):
        """
        Reads a file into a xarray dataset.
        
        Parameters
        ----------
            filename : string
                Valid path + filename
        """
        if self._ds is None:
            self._ds = xr.open_dataset(filename, **kwargs)
            logger.debug('read: {}'.format(self.__str__))
        else:
            errmsg = 'contrack() is already set!'
            raise ValueError(errmsg)
            
    def read_xarray(self, ds):
        """
        Read an existing xarray data set.
        
        Parameter:
        ----------
            ds: data set
                Valid xarray data set.
        """
        if self._ds is None:
            if not isinstance(ds, xr.core.dataset.Dataset):
                errmsg = 'ds has to be a xarray data set!'
                raise ValueError(errmsg)
            self._ds = ds
            logger.debug('read_xarray: {}'.format(self.__str__))
        else:
            errmsg = 'contrack() is already set!'
            raise ValueError(errmsg)
 
# ----------------------------------------------------------------------------
# Set up / Check variable and dimension
   

    def set_up(self,
               time_name=None,
               longitude_name=None,
               latitude_name=None,
               variable_name=None
    ):
        """
        Prepares the dataset for contour tracking. Does consistency checks
        and tests if all required information is available. Sets internal
        variables and dimensions.

        Parameters
        ----------
            time_name : TYPE, optional
                Name of time dimension. The default is None.
            longitude_name : TYPE, optional
                Name of longitude dimension. The default is None.
            latitude_name : TYPE, optional
                Name of latitude dimension. The default is None.
            variable_name : TYPE, optional
                Name of variable used for blocking. The default is 'Geopotential height'.

        Returns
        -------
            None.

        """

        # check dimensions
        if time_name is None:
            self._time_name = self._get_name_time()  
        else:
            self._time_name = time_name
        if longitude_name is None:
            self._longitude_name = self._get_name_longitude()
        else:
            self._longitude_name = longitude_name
        if latitude_name is None:
            self._latitude_name = self._get_name_latitude()
        else:
            self._longitude_name = latitude_name
        # check variable
        if variable_name is None:
            self._variable_name = self._get_name_variable()
        else:
            if variable_name not in self.variables:
                logger.warning(
                    "\n Variable '{}' not found. "
                    "Select from {}.".format(
                variable_name, self.variables)
                )
            else:
                self._variable_name = variable_name
        # set resolution
        if (self._longitude_name and self._latitude_name) is not None:
            self._dlon =  self._get_resolution(self._longitude_name)
            self._dlat =  self._get_resolution(self._latitude_name)

        if self._time_name is not None:
            self._dtime = self._get_resolution(self._time_name)
       
        # print names       
        logger.info(
            "\n time: '{}'\n"
            " longitude: '{}'\n"
            " latitude: '{}'\n"
            " variable: '{}'\n".format(
            self._time_name, 
            self._longitude_name,
            self._latitude_name,
            self._variable_name)
        )

    
    def _get_name_time(self):
        """
        check for 'time' dimension and return name
        """
        # check unit
        for dim in self._ds.dims:
            if (('units' in self._ds[dim].attrs and
                'since' in self._ds[dim].attrs['units']) or 
                ('units' in self._ds[dim].encoding and
                 'since' in self._ds[dim].encoding['units'])):
                return dim
        # check dtype
        for dim in self._ds.variables:
            try:
                var = self._ds[dim].data[0]
            except IndexError:
                var = self._ds[dim].data
            if isinstance(var, datetime64):
                return dim   
        # no 'time' dimension found
        logger.warning(
            "\n 'time' dimension (dtype='datetime64[ns]') not found."
        )
        return None     


    def _get_name_longitude(self):
        """
        check for 'longitude' dimension and return name
        """
        for dim in self._ds.dims:
            if ('units' in self._ds[dim].attrs and
               self._ds[dim].attrs['units'] in ['degree_east', 'degrees_east']):
                return dim
        # no 'longitude' dimension found
        logger.warning(
            "\n 'longitude' dimension (unit='degrees_east') not found."
        )
        return None


    def _get_name_latitude(self):
        """
        check for 'latitude' dimension and return name
        """
        for dim in self._ds.dims:
            if ('units' in self._ds[dim].attrs  and
                self._ds[dim].attrs['units'] in ['degree_north', 'degrees_north']):
                return dim
        # no 'latitude' dimension found
        logger.warning(
            "\n 'latitude' dimension (unit='degrees_north') not found."
        )
        return None

    def _get_name_variable(self):
        """
        check for 'Geopotential Height' variable and return name
        """
        for var in self._ds.data_vars:
            if ('units' in self._ds[var].attrs  and
                self._ds[var].attrs['units'] in ['gpm', 'm']):
                return var  
        # check for 'Geopotential'
        for var in self._ds.data_vars:
            if (('units' in self._ds[var].attrs  and
                self._ds[var].attrs['units'] in ['gp', 'm**2 s**-2']) or 
                (var in ['z', 'Geopotential'])):
                logger.warning(
                    "\n 'Geopotential height' variable (unit='gpm' or 'm') not found.\n"
                    "Hint: use 'calculate_gph_from_gp({})'.".format(var)
                )
                return None
        # no 'Geopotential height' dimension found
        logger.warning(
            "\n 'Geopotential height' variable (unit='gpm' or 'm') not found."
        )
        return None 
            
    def _get_resolution(self, dim, force=False):
        """
        set spatial (lat/lon) and temporal (time) resolution
        """
        # time dimension in days
        if dim == self._time_name:
            try:
                var = self._ds[dim].to_index()
                delta = np.unique((
                    self._ds[dim].to_index()[1:] - 
                    self._ds[dim].to_index()[:-1])
                    .astype('timedelta64[D]')
                )
            except AttributeError:  # dates outside of normal range
                # we can still move on if the unit is "days since ..."
                if ('units' in self._ds[dim].attrs and
                    'days' in self._ds[dim].attrs['units']):
                    var = self._ds[dim].data
                    delta = np.unique(var[1:] - var[:-1])
                else:
                    errmsg = 'Can not decode time with unit {}'.format(
                        self._ds[dim].attrs['units'])
                    raise ValueError(errmsg)
        # lat/lon dimension in Degree
        else:
            delta = abs(np.unique((
                self._ds[dim].data[1:] - 
                self._ds[dim].data[:-1])
            ))
        # check resolution
        if len(delta) > 1:
            errmsg = 'No regular grid found for dimension {}'.format(dim)
            if force:
                logging.warning(errmsg)
                logmsg = ' '.join(['force=True: using mean of non-equidistant',
                                   'grid {}'.format(delta)])
                logging.warning(logmsg)
                delta = [round(delta.mean(), 2)]
            else:
                raise ValueError(errmsg)
        elif delta[0] == 0:
            errmsg = 'Two equivalent values found for dimension {}.'.format(
                dim)
            raise ValueError(errmsg)
        elif delta[0] < 0:
            errmsg = ' '.join(['{} not increasing. This should',
                                   'not happen?!']).format(dim)
            raise ValueError(errmsg)
            
        return delta
        
        
# ----------------------------------------------------------------------------
# calculations
  
    def calculate_gph_from_gp(self,
                              gp_name='z',
                              gp_unit='m**2 s**-2',
                              gph_name='z_height'
    ):
        """
        Creates a new variable with name gph_name from the variable gp_name
        by dividing it through the mean gravitational accelerating g=9.80665
        m s**-2.
        
        Parameters:
        ----------
            gp_name='z': string, optional
                Name of the variable containing the geopotential
            gp_unit='m**2 s**-2':  string, optional
                Unit of gp_name
            gph_name='GeopotentialHeight': string, optional
                Name of the newly created variable containing the geopotential height
            
        Returns
        -------
            ds: xarray dataset
                Dataset containing gph_name
        """
        g = 9.80665  # m s**-2
        # https://dx.doi.org/10.6028/NIST.SP.330e2008
        if self._ds[gp_name].attrs['units'] != gp_unit:
            errmsg = 'Geopotential unit should be {} not {}'.format(
                gp_unit, self._ds[gp_name].attrs['units'])
            raise ValueError(errmsg)

        self._ds[gph_name] = xr.Variable(
            self._ds.variables[gp_name].dims,
            self._ds.variables[gp_name].data / g,
            attrs={
                'units': 'm',
                'long_name': 'Geopotential Height',
                'standard_name': 'geopotential height',
                'history': 'Calculated from {} with g={}'.format(gp_name, g)})
        logger.info('Calculating GPH from GP... DONE')
    
    def calc_mean(self, variable=""):
        """
        Calculate mean along time axis for variable

        Parameters
        ----------
            variable : string
                Date

        Returns
        -------
            array:  float
                Climatological mean

        """
        if not variable:
            var_mean = self['z'].mean(dim="time")
        else:
            if variable not in self.variables:
                logger.warning(
                    "\n Variable '{}' not found. "
                    "Select from {}.".format(
                variable, self.variables)
                )
                return None
            else:
                var_mean = self[variable].mean(dim="time")
        return var_mean
  
    
    def calc_clim(self, 
                  variable, 
                  window=31
    ):
        """
        Calculate daily (long-term) climatology, smoothed with a rolling average.
        If one year or less of data: calculate running mean.

        Parameters
        ----------
            variable : string
                Date

        Returns
        -------
            array:  float
                daily (long-term) climatological mean

        """
        
        # step 1: long-term daily mean
        clim = self[variable].groupby('time.dayofyear').mean('time')
     
        clim = clim.chunk({'dayofyear': None})
        # step 2: running mean ( with periodic boundary)
        clim = clim.rolling(dayofyear=window, center=True).mean().fillna(
            clim[-window:].mean(dim='dayofyear')    
        )

        
        #roll1 = clim.roll(dayofyear=(window*3), roll_coords=True).rolling(dayofyear=window, center=True).mean()
        #roll2 = clim.rolling(dayofyear=window, center=True).mean()
        #clim = xr.concat([roll1, roll2], dim='r').mean('r')
        #del roll1, roll2
        
        return clim
 
    
    def calc_anom(self, 
                  variable,
                  window=31
    ):
        """
        Creates a new variable with name "anom" from variable.
        Anomalies are computed for each grid point and time step as the departure from a climatology.

        Parameters
        ----------
            variable : string
                Input variable.
            window : int, optional
                number of timesteps to for climatological running mean

        Returns
        -------
            array: float
                Anomalie field

        """
        
        # step 1: calculate clim
        clim = self.calc_clim(variable=variable, window=window)  
               
        # step 2: calculate and create new variable anomaly     
        self._ds['anom'] = xr.Variable(
            self._ds[variable].dims,
            self._ds[variable].groupby('time.dayofyear') - clim,
            attrs={
                'units': self._ds[variable].attrs['units'],
                'long_name': self._ds[variable].attrs['long_name'] + ' Anomaly',
                'standard_name': self._ds[variable].attrs['long_name'] + ' anomaly',
                'history': 'Calculated from {}.'.format(variable)}
        )
        logger.info('Calculating Anomaly... DONE')
            
            
    def run_contrack(self,
                  variable="anom",
                  threshold=150,
                  overlap=0.5,
                  persistence=5
                  
    ):
        """
        Define closed contours by a threshold value

        """
    
        logger.info(
            "\nRun ConTrack \n"
            "########### \n"
            "    threshold:    {}\n"
            "    overlap:      {} %\n"
            "    persistency:  {} days".format(
                threshold, overlap, persistence)
        )
        
        
        # step 1: find contours (circulation anomalies)
        logger.info("Find individual contours...")
        flag = xr.where(self._ds['anom'] >= threshold, 1, 0)
        
        # step 2: identify individual contours
        flag, num_features = ndimage.label(flag.data, structure= np.array([[[0, 0, 0], [0,0,0], [0,0,0]],
                                                                          [[1, 1, 1], [1,1,1], [1,1,1]],
                                                                          [[0, 0, 0], [0,0,0], [0,0,0]]])
                                          )
        # periodic boundry: allow contours to cross date border
        for tt in range(flag.shape[0]):
            for y in range(flag.shape[1]):
                if flag[tt, y, 0] > 0 and flag[tt, y, -1] > 0:
                    flag[tt][flag[tt] == flag[tt, y, -1]] = flag[tt, y, 0]
                          
                    
        #step 3: overlapping
        logger.info("Check for overlap...")
        var = self._ds['time'].to_index()
        dime = np.unique((var[1:] - var[:-1]).astype('timedelta64[D]'))
        var = self._ds['latitude'].data
        dlat = abs(np.unique((var[1:] - var[:-1])))
        var = self._ds['longitude'].data
        dlon = np.unique((var[1:] - var[:-1]))
       
        weight_lat = np.cos(self._ds['latitude'].data*np.pi/180)
        weight_grid = np.ones((181, 360)) * np.array((111 * dlat * 111 * dlon * weight_lat)).astype(np.float32)[:, None]

        for tt in range(1,flag.shape[0]-1): 
            # loop over individual contours
            slices = ndimage.find_objects(flag[tt])
            label = 0
            for slice_ in slices:
                label = label+1
                if slice_ is None:
                    #no feature with this flag
                    continue
                areacon = np.sum(weight_grid[slice_][flag[tt][slice_] == label])
                areaover_forward = np.sum(weight_grid[slice_][(flag[tt][slice_] == label) & (flag[tt+1][slice_] >= 1)])
                areaover_backward = np.sum(weight_grid[slice_][(flag[tt][slice_] == label) & (flag[tt-1][slice_] >= 1)])
        
                fraction_backward = (1 / areacon) * areaover_backward
                fraction_forward = (1 / areacon) * areaover_forward 
             
                # middle
                if fraction_backward != 0 and fraction_forward != 0:
                    if (fraction_backward < overlap) or (fraction_forward < overlap):
                        flag[tt][slice_][(flag[tt][slice_] == label)] = 0.
                # decay
                if fraction_backward != 0 and fraction_forward == 0:
                    if (fraction_backward < overlap):
                        flag[tt][slice_][(flag[tt][slice_] == label)] = 0.
                # onset
                if fraction_backward == 0 and fraction_forward != 0:        
                    if (fraction_forward < overlap):
                        flag[tt][slice_][(flag[tt][slice_] == label)] = 0.
                        
        # step 4: persistency
        # find features along time axis
        logger.info("Check for persistency...")
        flag, num_features = ndimage.label(flag, structure = np.array([[[0, 0, 0], [0,1,0], [0,0,0]],
                                                                      [[1, 1, 1], [1,1,1], [1,1,1]],
                                                                      [[0, 0, 0], [0,1,0], [0,0,0]]]))
        # periodic boundry: allow features to cross date border
        for tt in range(flag.shape[0]):
            for y in range(flag.shape[1]):
                if flag[tt, y, 0] > 0 and flag[tt, y, -1] > 0:
                    flag[tt][flag[tt] == flag[tt, y, -1]] = flag[tt, y, 0]
        
        label = 0
        for slice_ in ndimage.find_objects(flag):
            label = label+1
            if slice_ is None:
                #no feature with this flag
                continue
            if (slice_[0].stop - slice_[0].start) < persistence:
                flag[slice_][(flag[slice_] == label)] = 0.        
        
        num_features = len(np.unique(flag))        
        # create new variable flag
        self._ds['flag'] = xr.Variable(
            self._ds[variable].dims,
            flag,
            attrs={
                'units': self._ds[variable].attrs['units'],
                'long_name': self._ds[variable].attrs['long_name'] + ' Anomaly',
                'standard_name': self._ds[variable].attrs['long_name'] + ' anomaly',
                'history': 'Calculated from {}.'.format(variable)}
        )
        logger.info("Calculating Blocking.. DONE\n"
                    "{} blocks tracked".format(num_features)
                    )
    
   

    # def fullname(self):
    #     """Return the fullname of the instance"""
    #     return '{} {}_{}'.format(self.model, self.variable, self.pressure_lvl)

    # def apply_raise(self):
    #     """Apply the raise_amt to the pressure_level"""
    #     self.pay = int(self.pressure_lvl * self.raise_amt)

    # @classmethod
    # def set_raise_amt(cls, amount):
    #     """Set the raise amount."""
    #     cls.raise_amt = amount

    # @classmethod
    # def from_string(cls, emp_str):
    #     """
    #     Initialize a blocking object from a string.

    #     Parameters
    #     ----------
    #     emp_str : string
    #         String in the format of model-variable-pressure_lvl

    #     Returns
    #     -------
    #     object
    #         A new class instance
    
          # Examples
          # --------
          # >>> emp_str_1 = 'ERA5-z-500'
          # >>> block_3 = blocking.from_string(emp_str_1)

    #     """
    #     # cls = object holder of class itself, not an instance of the class. 
    #     model, variable, pressure_lvl = emp_str.split('-')
    #     return cls(model, variable, pressure_lvl)

    # @staticmethod
    # def is_workday(day):
    #     """
    #     Check if day is a workday.

    #     Parameters
    #     ----------
    #     day : datetime object
    #         Date

    #     Returns
    #     -------
    #     bool
    #         True if workday, False otherwise

    #     """
    #     if day.weekday() == 5 or day.weekday() == 6:
    #         return False
    #     return True    


    

