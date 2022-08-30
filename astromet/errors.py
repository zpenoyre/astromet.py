import os
import numpy as np
import scipy

# loads data needed to find astrometric error as functon of magnitude
# data digitized from Lindegren+2020 fig A.1
local_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in

ast_abs_file_path = local_dir+'/data/scatteral_edr3.csv'
sigma_al_data = np.genfromtxt(ast_abs_file_path, skip_header=1, delimiter=',', unpack=True)
mags = sigma_al_data[0]
sigma_als = sigma_al_data[1]
sigma_ast = scipy.interpolate.interp1d(mags, sigma_als, bounds_error=False)

# equivalent for spectroscopic errors (using DR3 RVS)
spec_abs_file_path=local_dir+'/data/dr3_rvs_spec_error.csv'
sigma_spec_data = np.genfromtxt(spec_abs_file_path, skip_header=1, delimiter=',', unpack=True)
spec_mags = np.unique(sigma_spec_data[0])
spec_cols = np.unique(sigma_spec_data[1])
spec_sigmas = np.reshape(sigma_spec_data[2],(spec_mags.size,spec_cols.size)).T
sigma_spec_int=scipy.interpolate.RegularGridInterpolator((spec_mags,spec_cols),
    np.log10(spec_sigmas),method='linear')
def sigma_spec(mags,cols):
    mags=np.array(mags)
    cols=np.array(cols)
    return 10**sigma_spec_int(np.vstack([mags,cols]).T)

# equivalent for photometric errors (using DR3 RVS)
phot_abs_file_path=local_dir+'/data/dr3_rvs_phot_error.csv'
sigma_phot_data = np.genfromtxt(phot_abs_file_path, skip_header=1, delimiter=',', unpack=True)
phot_mags = np.unique(sigma_phot_data[0])
phot_cols = np.unique(sigma_phot_data[1])
phot_sigmas = np.reshape(sigma_phot_data[2],(phot_mags.size,phot_cols.size)).T
sigma_phot_int=scipy.interpolate.RegularGridInterpolator((phot_mags,phot_cols),
    np.log10(phot_sigmas),method='linear')
def sigma_phot(mags,cols):
    mags=np.array(mags)
    cols=np.array(cols)
    return 10**sigma_phot_int(np.vstack([mags,cols]).T)
