import numpy as np
import astropy.coordinates
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import get_body_barycentric


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

def fit_model(x_obs, x_err):
    """
    Iterative optimization to fit astrometric solution in AGIS (outer iteration). Lindegren 2012.
    Args:
        - x_obs, ndarray - observed along-scan source position at each epoch.
        - x_err, ndarray - astrometric measurement uncertainty for each observation.
    Returns:
        - r5d_mean
        - r5d_cov
        - R
        - aen_i
        - W_i
    """
    # Initialise
    en = 0
    weights = np.ones(len(x_obs))
    W = np.eye(len(x_obs))*weights/(x_err**2 + en)

    for ii in range(10):

        # Step 2 - Astrometry linear regression
        r5d_cov = np.linalg.inv(np.matmul(M_matrix.T, np.matmul(W, M_matrix)))
        r5d_mean = np.matmul(r5d_cov, np.matmul(M_matrix.T, np.matmul(W, x_obs)))
        R = x_obs - np.matmul(M_matrix, r5d_mean)

        # Step 3 - Observation Weights
        weights = downweight(R, x_err, en)
        W = np.eye(len(x_obs))*weights/(x_err**2 + en)

        # Step 4 - astrometric_excess_noise
        en = en_fit(R, al_err, weights)

        # Step 1 - Observation weights
        weights = downweight(R, x_err, en)
        W = np.eye(len(x_obs))*weights/(x_err**2 + en)

    # Final Astrometry Linear Regression fit
    r5d_cov = np.linalg.inv(np.matmul(M_matrix.T, np.matmul(W, M_matrix)))
    r5d_mean = np.matmul(r5d_cov, np.matmul(M_matrix.T, np.matmul(W, x_obs)))

    return r5d_mean, r5d_cov, R, en, np.mean(weights)
