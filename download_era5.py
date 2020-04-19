# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# =======
# import packages
import cdsapi

# =======
# download data from https://cds.climate.copernicus.eu/cdsapp#!/dataset/
c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-pressure-levels',
    {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'day': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
            '13', '14', '15',
            '16', '17', '18',
            '19', '20', '21',
            '22', '23', '24',
            '25', '26', '27',
            '28', '29', '30',
            '31',
        ],
        'variable': [
            'geopotential',
        ],
        'pressure_level': '500',
        'year': [
                '1981', '1982', '1983',
                '1984', '1985', '1986',
                '1987', '1988', '1989',
                '1990', '1991', '1992',
                '1993', '1994', '1995',
                '1996', '1997', '1998',
                '1999', '2000', '2001',
                '2002', '2003', '2004',
                '2005', '2006', '2007',
                '2008', '2009', '2010',
        ],
        'month': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
        ],
        'time': [
            '00:00', '12:00',
        ],
        'grid': [1.0, 1.0], # Latitude/longitude grid. Default: 0.25 x 0.25
    },
    'data/era5_1981-2010_z_500.nc')