import numpy as np
import astropy.coordinates
from astropy import units as u
from astropy.time import Time

import astromet.track

def get_obmt(times):
    return 1717.6256+((np.array(times) + 2455197.5 - 2457023.5 - 0.25)*4)
def get_gaiat(times):
    return (np.array(times) - 1717.6256)/4 -  (2455197.5 - 2457023.5 - 0.25)

def downweight(R, err, aen):
    """
    Downweighting function used in AGIS. Lindegren 2012.
    Args:
        - R, ndarray - residual of observed source position from astrometric solution.
        - err, ndarray - astrometric uncertainty of each observation
        - aen, ndarray - source astrometric excess noise ?why not scalar?
    Returns:
        - w, ndarray - observation weights
    """
    z = np.sqrt( R**2/(err**2 + aen**2) )
    w = np.where(z<2, 1, 1 - 1.773735*(z-2)**2 + 1.141615*(z-2)**3)
    w = np.where(z<3, w, np.exp(-z/3))
    # ? w = 0 or 1 only ?
    return w

def en_fit(R, err, w):
    """
    Iterative optimization to fit excess noise in AGIS (inner iteration). Lindegren 2012.
    Args:
        - R, ndarray - residual of observed source position from astrometric solution.
        - err, ndarray - astrometric uncertainty of each observation
        - w, ndarray - observation weights
    Returns:
        - aen, ndarray - astrometric_excess_noise
    """
    y = 0
    nu = np.sum(w>=0.2)-5

    W = w/(err**2 + y)
    Q = np.sum(R**2 * W)

    W_prime = -w/(err**2 + y)**2
    Q_prime = np.sum(R**2 * W_prime)

    for i in range(4):
        W = w/(err**2 + y)
        Q = np.sum(R**2 * W)
        if (i==0)&(Q<=nu): break

        W_prime = -w/(err**2 + y)**2
        Q_prime = np.sum(R**2 * W_prime)

        y = y + (1-Q/nu)*Q/Q_prime

    return np.sqrt(y)

def agis_2d_prior(ra, dec, G):

    coord=astropy.coordinates.SkyCoord(ra, dec, unit='deg', frame='icrs')
    _l=coord.galactic.l.rad; _b=coord.galactic.b.rad

    # Prior
    s0 = 2.187 - 0.2547*G + 0.006382*G**2
    s1 = 0.114 - 0.0579*G + 0.01369*G**2 - 0.000506*G**3
    s2 = 0.031 - 0.0062*G
    sigma_pi_f90 = 10**(s0 + s1*np.abs(np.sin(_b)) + s2*np.cos(_b)*np.cos(_l))

    prior_cov = np.eye(5)
    prior_cov[:2,:2] *= 1000**2
    prior_cov[2:4,2:4] *= sigma_pi_f90**2
    prior_cov[4,4] *= (10*sigma_pi_f90)**2

    prior_prec = np.linalg.inv(prior_cov)

    return prior_prec

def fit_model(x_obs, x_err, M_matrix, prior=None):
    """
    Iterative optimization to fit astrometric solution in AGIS (outer iteration). Lindegren 2012.
    Args:
        - x_obs,    ndarray - observed along-scan source position at each epoch.
        - x_err,    ndarray - astrometric measurement uncertainty for each observation.
        - M_matrix, ndarray - Design matrix.
    Returns:
        - r5d_mean
        - r5d_cov
        - R
        - aen_i
        - W_i
    """

    if prior is None: prior=np.zeros((5,5))

    # Initialise - get initial astrometry estimate with weights=1
    aen = 0
    weights = np.ones(len(x_obs))

    W = np.eye(len(x_obs))*weights/(x_err**2 + aen)
    r5d_cov = np.linalg.inv(np.matmul(M_matrix.T, np.matmul(W, M_matrix))+prior)
    r5d_mean = np.matmul(r5d_cov, np.matmul(M_matrix.T, np.matmul(W, x_obs)))
    R = x_obs - np.matmul(M_matrix, r5d_mean)

    # Step 1: initial Weights
    # Intersextile range
    ISR = np.diff(np.percentile(R, [100.*1./6, 100.*5./6.]))[0]
    # Evaluate weights
    weights = downweight(R, ISR/2., 0.)

    for ii in range(10):

        # Step 2 - Astrometry linear regression
        W = np.eye(len(x_obs))*weights/(x_err**2 + aen)
        r5d_cov = np.linalg.inv(np.matmul(M_matrix.T, np.matmul(W, M_matrix))+prior)
        r5d_mean = np.matmul(r5d_cov, np.matmul(M_matrix.T, np.matmul(W, x_obs)))
        R = x_obs - np.matmul(M_matrix, r5d_mean)

        # Step 3 - astrometric_excess_noise
        aen = en_fit(R, x_err, weights)

        # Step 4 - Observation Weights
        weights = downweight(R, x_err, aen)

        # Step 3 - astrometric_excess_noise
        aen = en_fit(R, x_err, weights)

    # Final Astrometry Linear Regression fit
    r5d_cov = np.linalg.inv(np.matmul(M_matrix.T, np.matmul(W, M_matrix))+prior)
    r5d_mean = np.matmul(r5d_cov, np.matmul(M_matrix.T, np.matmul(W, x_obs)))
    R = x_obs - np.matmul(M_matrix, r5d_mean)

    return r5d_mean, r5d_cov, R, aen, weights


def agis(r5d, t, phi, x_err, extra=None, epoch=2016.0, G=None):
    """
    Iterative optimization to fit astrometric solution in AGIS (outer iteration). Lindegren 2012.
    Args:
        - r5d,        ndarray - 5D astrometry of source - (ra, dec (deg), parallax (mas), mura, mudec (mas/y))
        - t,          ndarray - source observation times - (julian days relative to Gaia observation start?)
        - phi,        ndarray - source observation scan angles
        - x_err,      ndarray - scan measurement error
        - extra,  function or None - Takes times and returns offset of centroid from CoM in mas.
    Returns:
        - gaia_dr2, dict - output data Gaia would produce
    """

    results = {}
    results['astrometric_matched_transits']     = len(t)
    results['visibility_periods_used'] = np.sum(np.sort(t)[1:]*astromet.T-np.sort(t)[:-1]*astromet.T>4)

    t = np.repeat(t, 9)
    phi = np.repeat(phi, 9)
    x_err = np.repeat(x_err, 9)


    results['astrometric_n_obs_al']     = len(t)

    # Add prior on components if fewer that 6 visibility periods
    if results['visibility_periods_used']<6:
        prior = agis_2d_prior(r5d[0], r5d[1], G)
        results['astrometric_params_solved']=3
    else:
        prior = np.zeros((5,5))
        results['astrometric_params_solved']=31

    # Design matrix
    design = astromet.design_matrix(t, phi, r5d[0], r5d[1], epoch=epoch)

    # Transform ra,dec to milliarcsec
    r5d[:2] = r5d[:2]*(3600*1000)

    # Astrometric position of source
    x_obs  = np.matmul(design, r5d)
    # Excess motion
    if extra is not None:
        x_obs += np.sum(np.vstack((np.sin(np.deg2rad(phi)), np.cos(np.deg2rad(phi))))*extra(t), axis=0)
    # Measurement Error
    x_obs += np.random.normal(0, x_err)

    r5d_mean, r5d_cov, R, aen, weights = fit_model(x_obs, x_err, design, prior=prior)
    # Transform ra,dec to degrees
    r5d_mean[:2] = r5d_mean[:2]/(3600*1000)

    coords = ['ra', 'dec', 'parallax', 'pmra', 'pmdec']
    for i in range(5):
        results[coords[i]] = r5d_mean[i]
        results[coords[i]+'_error'] = np.sqrt(r5d_cov[i,i])
        for j in range(i):
            results[coords[j]+'_'+coords[i]+'_corr']=\
                r5d_cov[i,j]/np.sqrt(r5d_cov[i,i]*r5d_cov[j,j])

    results['astrometric_excess_noise'] = aen
    results['astrometric_chi2_al']      = np.sum(R**2 / x_err**2)
    results['astrometric_n_good_obs_al']= np.sum(weights>0.2)
    nparam=5 #results['astrometric_params_solved'].bit_count()
    results['UWE']= np.sqrt(np.sum(R**2 / x_err**2)/(np.sum(weights>0.2)-nparam))

    return results

def gaia_fit(ts, xs, phis, errs, ra, dec, G=12, epoch=2016.0):
    """
    Iterative optimization to fit astrometric solution in AGIS (outer iteration).
    Lindegren 2012.
    Args:
        - ts,          ndarray - source observation times, jyear
        - xs,          ndarray - source 1d positions, mas
        - phis,        ndarray - source observation scan angles, deg
        - errs,        ndarray - scan measurement error, mas
        - ra,          float - RA for design_matrix, deg
        - dec,         float - Dec for design_matrix, deg
        - G,           float - Apparent magnitude, 2 parameter prior only, mag
        - epoch        float - Epoch at which results are calculated, jyear
            Returns:
        - results      dict - output data Gaia would produce
    """

    results = {}
    results['astrometric_matched_transits']     = len(ts)
    results['visibility_periods_used'] = np.sum(np.sort(ts)[1:]*astromet.T-np.sort(ts)[:-1]*astromet.T>4)

    t = np.repeat(ts, 9)
    phi = np.repeat(phis, 9)
    x = np.repeat(xs, 9)
    x_err = np.repeat(errs, 9)

    results['astrometric_n_obs_al']     = len(t)

    # Add prior on components if fewer that 6 visibility periods
    if results['visibility_periods_used']<6:
        prior = agis_2d_prior(ra, dec, G)
        results['astrometric_params_solved']=3
    else:
        prior = np.zeros((5,5))
        results['astrometric_params_solved']=31

    # Design matrix
    design = astromet.design_matrix(t, phi, ra, dec, epoch=epoch)

    r5d_mean, r5d_cov, R, aen, weights = fit_model(x, x_err, design, prior=prior)
    # Transform ra,dec to degrees
    r5d_mean[1] = r5d_mean[1]*astromet.mas
    r5d_mean[0] = r5d_mean[0]*astromet.mas/np.cos(np.deg2rad(r5d_mean[1]))

    coords = ['ra', 'dec', 'parallax', 'pmra', 'pmdec']
    for i in range(5):
        results[coords[i]] = r5d_mean[i]
        # WARNING - all RA and Dec terms have wrong units, to fix!
        results[coords[i]+'_error'] = np.sqrt(r5d_cov[i,i])
        for j in range(i):
            results[coords[j]+'_'+coords[i]+'_corr']=\
                r5d_cov[i,j]/np.sqrt(r5d_cov[i,i]*r5d_cov[j,j])

    results['astrometric_excess_noise'] = aen
    results['astrometric_chi2_al']      = np.sum(R**2 / x_err**2)
    results['astrometric_n_good_obs_al']= np.sum(weights>0.2)
    nparam=5 #results['astrometric_params_solved'].bit_count()
    results['UWE']= np.sqrt(np.sum(R**2 / x_err**2)/(np.sum(weights>0.2)-nparam))

    return results
