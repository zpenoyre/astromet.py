# astrometpy

A simple python package for generating astrometric tracks of single stars and the center of light of binary systems.

Most of the functionality is mapped out in the exaple.ipynb notebook.

Requires:
numpy
astropy
matplotlib (for notebook)

To do:
Pip package
More explicit choice of epoch
1D scans

Suggestions welcome


# AGIS emulator
---------------

We've included code to predict outputs from the _Gaia_ pipeline given the source position on the sky.
This combines astrometpy with [scanninglaw](https://github.com/gaiaverse/scanninglaw).
An example script for predicting the _Gaia_ outputs is given in [agis_emulator.ipynb](https://github.com/gaiaverse/astrometpy/blob/master/agis_emulator.ipynb).

There are three parts:
1) Get source scan times and angles:
  ```python
  >>> ra, dec = 0,0; G=16
  >>> c = Source(l=l, b=b, unit='deg', frame='galactic', photometry={'gaia_g':G})
  >>> scan_law = dr2_sl(c, return_times=True, return_angles=True)

  # Time in jyear
  >>> times = 2010+np.hstack(scan_law['times'])[0]/365.25
  >>> angles = np.hstack(scan_law['angles'])[0]
  ```

2) Define source motion:
    - 5D astrometry - position, parallax, proper motion
    - Excess motion function e.g. binary, microlens etc.
  ```python
  # Astrometry - deg, deg, mas, mas/y, mas/y
  >>> r5d = np.array([ra, dec, 12., 20., 20.])

  # Add 5mas excess noise
  >>> def excess(t, e=5):
  >>>     noise = np.random.normal(0,e,size=(2,len(t)))
  >>>     return noise
  ```

3) Get _Gaia_ pipeline output:
  ```python
  >>> import agis

  # Individual measurement error
  >>> x_err = np.zeros(len(times))+0.3

  >>> gaia = agis.agis(r5d, times, angles, x_err, extra=excess)
  >>> gaia

  {'astrometric_chi2_al': 40623.26867197566,
    'astrometric_excess_noise': 4.569150584265574,
    'astrometric_matched_transits': 18,
    'astrometric_n_good_obs_al': 162,
    'astrometric_n_obs_al': 162,
    'astrometric_params_solved': 31,
    'dec': 2.0,
    'dec_error': 0.2779772382198002,
    'dec_parallax_corr': -0.3581180678081153,
    'dec_pmdec_corr': -0.3833760211381195,
    'dec_pmra_corr': -0.1518427970510252,
    'parallax': 14.712123647332191,
    'parallax_error': 0.31884109681369793,
    'parallax_pmdec_corr': 0.2268586945193064,
    'parallax_pmra_corr': -0.23712031696093308,
    'pmdec': 20.10006219148636,
    'pmdec_error': 0.5228904026166806,
    'pmra': 19.930380627512932,
    'pmra_error': 0.5075960881843364,
    'pmra_pmdec_corr': 0.26960013287195167,
    'ra': 272.0,
    'ra_dec_corr': 0.24063663846119082,
    'ra_error': 0.25894934494663174,
    'ra_parallax_corr': -0.08274832927577698,
    'ra_pmdec_corr': -0.26376481339235414,
    'ra_pmra_corr': -0.20984996228867586,
    'visibility_periods_used': 10}
  ```
