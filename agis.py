import numpy as np
import astropy.coordinates
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import get_body_barycentric

import astromet


def downweight(R, err, aen):
    """
    Downweighting function used in AGIS. Lindegren 2012.
    Args:
        - R, ndarray - residual of observed source position from astrometric solution.
        - err, ndarray - astrometric uncertainty of each observation
        - aen, ndarray - source astrometric excess noise
    Returns:
        - w, ndarray - observation weights
    """
    z = np.sqrt( R**2/(err**2 + aen**2) )
    w = np.where(z<2, 1, 1 - 1.773735*(z-2)**2 + 1.141615*(z-2)**3)
    w = np.where(z<3, w, np.exp(-z/3))
    return w

def en_fit(R, err, w):
    """
    Iterative optimization to fit excess noise in AGIS (inner iteration). Lindegren 2012.
    Args:
        - R, ndarray - residual of observed source position from astrometric solution.
        - err, ndarray - astrometric uncertainty of each observation
        - w, ndarray - observation weights
    Returns:
        - aen, ndarry - astrometric_excess_noise
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

def fit_model(x_obs, x_err, M_matrix):
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
    # Initialise
    aen = 0
    weights = np.ones(len(x_obs))
    W = np.eye(len(x_obs))*weights/(x_err**2 + aen)

    for ii in range(10):

        # Step 2 - Astrometry linear regression
        r5d_cov = np.linalg.inv(np.matmul(M_matrix.T, np.matmul(W, M_matrix)))
        r5d_mean = np.matmul(r5d_cov, np.matmul(M_matrix.T, np.matmul(W, x_obs)))
        R = x_obs - np.matmul(M_matrix, r5d_mean)

        # Step 3 - Observation Weights
        weights = downweight(R, x_err, aen)
        W = np.eye(len(x_obs))*weights/(x_err**2 + aen)

        # Step 4 - astrometric_excess_noise
        aen = en_fit(R, x_err, weights)

        # Step 1 - Observation weights
        weights = downweight(R, x_err, aen)
        W = np.eye(len(x_obs))*weights/(x_err**2 + aen)

    # Final Astrometry Linear Regression fit
    r5d_cov = np.linalg.inv(np.matmul(M_matrix.T, np.matmul(W, M_matrix)))
    r5d_mean = np.matmul(r5d_cov, np.matmul(M_matrix.T, np.matmul(W, x_obs)))
    R = x_obs - np.matmul(M_matrix, r5d_mean)

    return r5d_mean, r5d_cov, R, aen, weights


def agis(r5d, t, phi, x_err, extra=None, t0=2015.5):
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

    # Design matrix
    design = astromet.design_matrix(t, phi, r5d[0], r5d[1], t0=t0)

    # Astrometric position of source - UNITS??
    x_obs  = np.matmul(design, r5d)
    # Excess motion
    if extra is not None:
        x_obs += np.sum(np.vstack((np.sin(np.deg2rad(phi)), np.cos(np.deg2rad(phi))))*extra(t), axis=0)
    # Measurement Error
    x_obs += np.random.normal(0, x_err)

    r5d_mean, r5d_cov, R, aen, weights = fit_model(x_obs, x_err, design)

    results = {}
    coords = ['ra', 'dec', 'parallax', 'pmra', 'pmdec']
    for i in range(5):
        results[coords[i]] = r5d_mean[i]
        results[coords[i]+'_error'] = np.sqrt(r5d_cov[i,i])
        for j in range(i):
            results[coords[j]+'_'+coords[i]+'_corr']=\
                r5d_cov[i,j]/np.sqrt(r5d_cov[i,i]*r5d_cov[j,j])

    results['astrometric_excess_noise'] = aen
    results['astrometric_chi2_al']      = np.sum(R**2 / x_err**2)
    results['astrometric_n_obs_al']     = len(t)
    results['astrometric_n_good_obs_al']= np.sum(weights>0.2)
    results['visibility_periods_used'] = np.sum(np.sort(t)[1:]*astromet.T-np.sort(t)[:-1]*astromet.T>4)

    return results
