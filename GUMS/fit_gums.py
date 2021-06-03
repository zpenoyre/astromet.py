import sys, os, numpy as np

# Load data
import h5py

gums_file = '/data/vault/asfe2/Conferences/EDR3_workshop/gums_sample.h'
with h5py.File(gums_file, 'r') as f:
    gums = {}
    for key in f.keys():
        gums[key] = f[key][...]


# Transform variables
gums['parallax'] = 1e3/gums['barycentric_distance']
gums['period'] = gums.pop('orbit_period')
gums['l'] = 10**(-(gums['primary_mag_g'] - gums['secondary_mag_g'])/2.5)
gums['q'] = gums['primary_mass']/gums['secondary_mass']
gums['a'] = gums.pop('semimajor_axis')
gums['e'] = gums.pop('eccentricity')
gums['vtheta'] = gums['periastron_argument']
gums['vphi'] = gums['longitude_ascending_node']
gums['vomega'] = gums['inclination']
gums['tperi'] = gums['periastron_date']
gums['phot_g_mean_mag'] = np.log10(10**(-gums['primary_mag_g']/2.5) + 10**(-gums['secondary_mag_g']/2.5))



# Load scanning law
import scanninglaw.times
from scanninglaw.source import Source
dr3_sl=scanninglaw.times.dr2_sl(version='dr3_nominal')



# Run fits
sys.path.append('../')
import tqdm, astromet

results = {'system_id':[], 'phot_g_mean_mag':[]}
for ii in tqdm.tqdm(range(100)):#len(gums['system_id']))):

    params=astromet.params()
    for key in ['ra','dec','pmra','pmdec','parallax',
              'period','l','q','a','e',
              'vtheta','vphi','vomega','tperi']:
        setattr(params, key, gums[key][ii])

    c=Source(params.ra,params.dec,unit='deg')
    sl=dr3_sl(c, return_times=True, return_angles=True)
    ts=2010+np.squeeze(np.hstack(sl['times']))/365.25
    sort=np.argsort(ts)
    ts=ts[sort]

    phis=np.squeeze(np.hstack(sl['angles']))[sort]

    trueRacs,trueDecs=astromet.track(ts,params)

    # Need to change this to total magnitude
    al_err = astromet.sigma_ast(gums['phot_g_mean_mag'][ii])
    t_obs,x_obs,phi_obs,rac_obs,dec_obs=astromet.mock_obs(ts,phis,trueRacs,trueDecs,err=al_err)

    fitresults=astromet.fit(t_obs,x_obs,phi_obs,al_err,params.ra,params.dec)
    gaia_output=astromet.gaia_results(fitresults)

    results['system_id'].append(gums['system_id'])
    results['phot_g_mean_mag'].append(gums['phot_g_mean_mag'])
    for key in gaia_output.keys():
        if ii==0: results[key] = [gaia_output[key]]
        else: results[key].append(gaia_output[key])


# Save results
save_file = '/data/vault/asfe2/Conferences/EDR3_workshop/gums_fits.h'
with h5py.File(save_file, 'w') as f:
    for key in results.keys():
        f.create_dataset(key, data=results[key])
