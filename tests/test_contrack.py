#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 23:04:55 2020

@author: steidani (Daniel Steinfeld; daniel.steinfeld@alumni.ethz.ch)
"""

# =======
# import packages
import numpy as np
import xarray as xr
import pandas as pd
import pytest
import warnings

from contrack import contrack

dataset = 'tests/test_data/anom_test.nc'
wrong_dataset = 'tests/test_data/anom_test.txt'

@pytest.fixture
def contracks():
    contracks = contrack()
    contracks.read(dataset)
    return contracks

def test_init_empty():
    contracks = contrack()
    assert contracks.ds is None

def test_init_netcdf(contracks):
    assert type(contracks.ds) is xr.Dataset

def test_read_netcdf():
    contracks = contrack(dataset)
    assert type(contracks) == contrack
    
def test_read_wrong():
    try:
        contrack(wrong_dataset)
    except IOError as err:
        assert err.args[0] == "Unkown fileformat. Known formats " \
                               "are netcdf."
def test_read_xarray():
    data = xr.open_dataset(dataset)
    contracks = contrack()
    contracks.read_xarray(data)
    assert type(contracks.ds) is xr.Dataset
    
def test_len(contracks):
    assert len(contracks) == 1

def test_ntime(contracks):
    assert contracks.ntime == 11
    
def test_dimensions(contracks):
    assert contracks.dimensions == ['latitude', 'longitude', 'time']
    
def test_variables(contracks):
    assert contracks.variables == ['anom']
    
def test_set_up_manually(contracks):
    contracks.set_up(time_name='time',
                     longitude_name='longitude',
                     latitude_name='latitude')    
    assert contracks._time_name == 'time'
    assert contracks._longitude_name == 'longitude'
    assert contracks._latitude_name == 'latitude'

def test_set_up_automatic(contracks):
    contracks.set_up()   
    assert contracks._time_name == 'time'
    assert contracks._longitude_name == 'longitude'
    assert contracks._latitude_name == 'latitude'
    
def test_calc_clim(contracks):
    contracks.set_up()  
    assert type(contracks.calc_clim('anom')) is xr.DataArray

def test_run_caltrack(contracks):
    contracks.run_contrack(variable='anom', 
                          threshold=150,
                          gorl='>=',
                          overlap=0.5,
                          persistence=5,
                          twosided=False)  
    assert contracks.variables == ['anom', 'flag']
    assert len(np.unique(contracks.flag)) - 1 == 3
    
def test_run_lifecycle(contracks):
    contracks.run_contrack(variable='anom', 
                          threshold=150,
                          gorl='>=',
                          overlap=0.5,
                          persistence=5,
                          twosided=False)  
    test = contracks.run_lifecycle(flag='flag', variable='anom')
    assert type(test) == pd.DataFrame
    assert len(test.Flag.unique()) == 3
    assert len(test) == 28