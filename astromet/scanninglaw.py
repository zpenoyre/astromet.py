from astropy.table import Table, Column
import numpy as np
import astropy
import healpy as hp
import os

local_dir = os.path.dirname(__file__)
data_path=local_dir+'/data/gost_full_extended_mission_hp6.fits'

gaiatimes=astropy.time.Time(["2014-07-25T10:31:25.554960001",
                            "2015-09-16T16:21:27.121893186",
                            "2016-05-23T11:36:27.459006034",
                            "2017-05-28T08:46:28.954612431",
                            "2020-01-20T22:01:30.250520158",
                            "2025-01-15T00:00:00"], format='isot', scale='tcb')
tstart=gaiatimes[0].decimalyear
tdr1=gaiatimes[1].decimalyear
tdr2=gaiatimes[2].decimalyear
tdr3=gaiatimes[3].decimalyear
tdr4=gaiatimes[4].decimalyear
tdr5=gaiatimes[5].decimalyear

data=astropy.table.Table.read(data_path, format="fits")

#ras=np.rad2deg(data['ra[rad]'])
#decs=np.rad2deg(data['dec[rad]'])
times=2000+(data['ObservationTimeAtBarycentre[BarycentricJulianDateInTCB]']-2451545.0)/365.25
angles=data['scanAngle[rad]']
healpixels=data['Target']

def scanninglaw(ra,dec,tstart=tstart,tend=tdr3):
    healpix=hp.ang2pix(2**6,ra,dec,lonlat=True,nest=True)
    entries=np.flatnonzero((healpixels==healpix) & (times>tstart) & (times<tend))
    return np.array(times[entries]),np.array(angles[entries])
