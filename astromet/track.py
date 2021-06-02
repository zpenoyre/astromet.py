import numpy as np
import astropy.coordinates
from astropy import units as u
from astropy import constants
from astropy.time import Time
import scipy.interpolate
import sys,os

# Create units used elsewhere
mSun = constants.M_sun.to(u.kg).value
lSun = constants.L_sun.to(u.W).value
kpc = constants.kpc.to(u.m).value
Gyr = (1.0*u.Gyr).to(u.s).value
day = (1.0*u.day).to(u.s).value
G = constants.G.to(u.m**3/u.kg/u.s**2).value
AU = (1.0*u.AU).to(u.m).value
c = constants.c.to(u.m/u.s).value
T = (1.0*u.year).to(u.day).value
year = (1.0*u.year).to(u.s).value
AU_c = (1.0*u.AU/constants.c).to(u.day).value
Galt = constants.G.to(u.AU**3/u.M_sun/u.year**2).value
mas2rad = (1.0*u.mas).to(u.rad).value
mas = (1.0*u.mas).to(u.deg).value

# loads data needed to find astrometric error as functon of magnitude
local_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
rel_path = '/data/scatteral_edr3.csv'
abs_file_path = local_dir+rel_path #os.path.join(local_dir, rel_path)
sigma_al_data = np.genfromtxt(abs_file_path,skip_header=1,delimiter=',',unpack=True)
mags=sigma_al_data[0]
sigma_als=sigma_al_data[1]
sigma_ast = scipy.interpolate.interp1d(mags, sigma_als, bounds_error=False)

# ----------------
# -User functions
# ----------------
class params():
    def __init__(self):
        # astrometric parameters
        self.ra = 45  # degree
        self.dec = 45  # degree
        self.drac = 0 # mas
        self.ddec = 0 # mas
        self.pmrac = 0  # mas/year
        self.pmdec = 0  # mas/year
        self.parallax = 1  # mas
        # binary parameters
        self.period = 1 # year
        self.a = 0  # AU
        self.e = 0.1
        self.q = 0
        self.l = 0  # assumed < 1 (though may not matter)
        self.vtheta = 45
        self.vphi = 45
        self.vomega = 0
        self.tperi = 0  # jyear

        # Below are assumed to be derived from other params
        # (I.e. not(!) specified by user)
        self.totalmass = -1  # solar mass
        self.Delta = -1

        # the epoch determines when RA and Dec (and other astrometry)
        # are centred - for dr3 it's 2016.0, dr2 2015.5, dr1 2015.0
        self.epoch=2016.0

def bjyr_to_bjd(jyrdate):
    return 1721057.5 + jyrdate*T
def bjd_to_bjyr(bjddate):
    return (bjddate - 1721057.5)/T

def totalmass(ps):
    ps.totalmass=4*(np.pi**2)*Galt/((ps.period**2)*(ps.a**3))
    return ps.totalmass

def Delta(ps):
    ps.Delta = np.abs(ps.q-ps.l)/((1+ps.q)*(1+ps.l))
    return ps.Delta

def track(ts, ps, comOnly=False, allComponents=False):
    """
    Astrometric track in RAcos(Dec) and Dec [mas] for a given binary
    Args:
        - ts,       ndarray - Observation times, jyear.
        - ps,       params object - Astrometric and binary parameters.
        - comOnly,  bool - If True return only c.o.m track (no binary)
        - allComponents,     bool - If True return pos. of c.o.l. & both components
    Returns:
        - racs      ndarry - RAcosDec at each time, mas
        - decs      ndarry - Dec at each time, mas
    """
    xij = design_matrix(ts, np.deg2rad(ps.ra), np.deg2rad(ps.dec), epoch=ps.epoch)

    r5d = np.array([ps.drac, ps.ddec, ps.parallax, ps.pmrac, ps.pmdec])
    dracs, ddecs = xij@r5d # all in mas

    if comOnly == True:
        return dracs, ddecs
    # extra c.o.l. correction due to binary
    px1s, py1s, px2s, py2s, pxls, pyls = binaryMotion(
        ts-ps.tperi, ps.period, ps.q, ps.l, ps.a, ps.e, ps.vtheta, ps.vphi)
    rls = ps.parallax*(pxls*np.cos(ps.vomega)+pyls*np.sin(ps.vomega))
    dls = ps.parallax*(pyls*np.cos(ps.vomega)-pxls*np.sin(ps.vomega))
    if allComponents==False:
        return dracs+rls, ddecs+dls # return just the position of the c.o.l.
    else: # returns all 3 components
        r1s = ps.parallax*(px1s*np.cos(ps.vomega)+py1s*np.sin(ps.vomega))
        d1s = ps.parallax*(py1s*np.cos(ps.vomega)-px1s*np.sin(ps.vomega))
        r2s = ps.parallax*(px2s*np.cos(ps.vomega)+py2s*np.sin(ps.vomega))
        d2s = ps.parallax*(py2s*np.cos(ps.vomega)-px2s*np.sin(ps.vomega))
        return dracs+rls, ddecs+dls, dracs+r1s, ddecs+d1s, dracs+r2s, ddecs+d2s

# ----------------
# -On-sky motion
# ----------------


def design_matrix(ts, ra, dec, phis=None, epoch=2016.0):
    """
    design_matrix - Design matrix for ra,dec source track
    Args:
        - t,       ndarray - Observation times, jyear.
        - phis,     ndarray - scan angles.
        - ra, dec,  float - reference right ascension and declination of source, radians
        - epoch     float - time at which position and pm are measured, years CE
    Returns:
        - design, ndarry - Design matrix
    """
    # Barycentric coordinates of Gaia at time t
    bs = barycentricPosition(ts)
    # unit vector in direction of increasing ra - the local west unit vector
    p0 = np.array([-np.sin(ra), np.cos(ra), 0])
    # unit vector in direction of increasing dec - the local north unit vector
    q0 = np.array([-np.cos(ra)*np.sin(dec), -np.sin(ra)*np.sin(dec), np.cos(dec)])

    # Construct design matrix for ra and dec positions
    design = np.zeros((2, ts.shape[0], 5))
    design[0,:,0] = 1 # ra*cos(dec)
    design[1,:,1] = 1 # dec
    design[0,:,2] = np.dot(p0, bs.T) # parallax (ra component)
    design[1,:,2] = np.dot(q0, bs.T) # parallax (dec component)
    design[0,:,3] = ts-epoch # pmra
    design[1,:,4] = ts-epoch # pmdec

    if np.size(phis)>1:
        # sin and cos angles
        angles = np.deg2rad(phis)
        sina = np.sin(angles)
        cosa = np.cos(angles)

        # Construct design matrix
        design = design[0]*sina[:,None] + design[1]*cosa[:,None]

    return design

'''
def design_1d(ts, phis, ra, dec, epoch=2016.0):
    """
    design_1d - Design matrix for 1d source track in along-scan direction
    Args:
        - t,       ndarray - Observation times, jyear.
        - phis,     ndarray - scan angles.
        - ra, dec,  float - reference right ascension and declination of source, radians
        - epoch     float - time at which position and pm are measured, years CE
    Returns:
        - design, ndarry - Design matrix
    """
    # Barycentric coordinates of Gaia at time t
    bs = barycentricPosition(ts)
    # unit vector in direction of increasing ra - the local west unit vector
    p0 = np.array([-np.sin(ra), np.cos(ra), 0])
    # unit vector in direction of increasing dec - the local north unit vector
    q0 = np.array([-np.cos(ra)*np.sin(dec), -np.sin(ra)*np.sin(dec), np.cos(dec)])

    # sin and cos angles
    angles = np.deg2rad(phis)
    sina = np.sin(angles)
    cosa = np.cos(angles)
    pifactor = sina*np.dot(p0, bs.T) + cosa*np.dot(q0, bs.T)

    # Construct design matrix
    design = np.zeros((ts.shape[0], 5))
    design[:, 0] = sina
    design[:, 1] = cosa
    design[:, 2] = pifactor
    design[:, 3] = sina*(ts-epoch)
    design[:, 4] = cosa*(ts-epoch)

    return design'''


def barycentricPosition(time):
    pos = astropy.coordinates.get_body_barycentric('earth', astropy.time.Time(time, format='jyear'))
    xs = pos.x.value  # all in AU
    ys = pos.y.value
    zs = pos.z.value
    # gaia satellite is at Earth-Sun L2
    l2corr = 1+np.power(3e-6/3, 1/3)  # 3(.003)e-6 is earth/sun mass ratio
    return l2corr*np.vstack([xs, ys, zs]).T


# binary orbit


def conditional_njit(backup = None):
    def decorator(func):
        try:
            from numba import njit
            return njit(func)
        except ImportError:
            if backup == None:
                return func
            else:
                return backup
    return decorator

def findEtas_backup(ts, period, eccentricity, tPeri=0, N_it = None):  # finds an (approximate) eccentric anomaly (see Penoyre & Sandford 2019, appendix A)
    eta0s =  ((2*np.pi/period)*(ts-tPeri)) % (2.0*np.pi) # (2*np.pi/period)*(ts-tPeri)
    eta1s = eccentricity*np.sin(eta0s)
    eta2s = (eccentricity**2)*np.sin(eta0s)*np.cos(eta0s)
    eta3s = (eccentricity**3)*np.sin(eta0s)*(1-(3/2)*np.sin(eta0s)**2)
    return eta0s+eta1s+eta2s+eta3s

@conditional_njit(findEtas_backup)
def findEtas(ts, period, eccentricity, tPeri=0, N_it = 10):
    """Solve Kepler's equation, E - e sin E = ell, via the contour integration method of Philcox et al. (2021)
    This uses techniques described in Ullisch (2020) to solve the `geometric goat problem'.
    Args:
        ts (np.ndarray): Times.
        period (float): Period of orbit.
        eccentricity (float): Eccentricity. Must be in the range 0<e<1.
        tPeri (float): Pericentre time.
        N_it (float): Number of grid-points.
    Returns:
        np.ndarray: Array of eccentric anomalies, E.

    Slightly edited version of code taken from https://github.com/oliverphilcox/Keplers-Goat-Herd
    """

    ell_array = ((2*np.pi/period)*(ts-tPeri)) % (2.0*np.pi)

    # Check inputs
    if eccentricity<=0.:
        raise Exception("Eccentricity must be greater than zero!")
    elif eccentricity>=1:
        raise Exception("Eccentricity must be less than unity!")
    if np.max(ell_array)>2.*np.pi:
        raise Exception("Mean anomaly should be in the range (0, 2 pi)")
    if np.min(ell_array)<0:
        raise Exception("Mean anomaly should be in the range (0, 2 pi)")
    if N_it<2:
        raise Exception("Need at least two sampling points!")

    # Define sampling points
    N_points = N_it - 2
    N_fft = (N_it-1)*2

    # Define contour radius
    radius = eccentricity/2

    # Generate e^{ikx} sampling points and precompute real and imaginary parts
    j_arr = np.arange(N_points)
    freq = (2*np.pi*(j_arr+1.)/N_fft).reshape((-1, 1))#[:,np.newaxis]
    exp2R = np.cos(freq)
    exp2I = np.sin(freq)
    ecosR= eccentricity*np.cos(radius*exp2R)
    esinR = eccentricity*np.sin(radius*exp2R)
    exp4R = exp2R*exp2R-exp2I*exp2I
    exp4I = 2.*exp2R*exp2I
    coshI = np.cosh(radius*exp2I)
    sinhI = np.sinh(radius*exp2I)

    # Precompute e sin(e/2) and e cos(e/2)
    esinRadius = eccentricity*np.sin(radius);
    ecosRadius = eccentricity*np.cos(radius);

    # Define contour center for each ell and precompute sin(center), cos(center)
    filt = ell_array<np.pi
    center = ell_array-eccentricity/2.
    center[filt] += eccentricity
    sinC = np.sin(center)
    cosC = np.cos(center)
    output = center

    ## Accumulate Fourier coefficients
    # NB: we halve the integration range by symmetry, absorbing factor of 2 into ratio

    ## Separate out j = 0 piece, which is simpler

    # Compute z in real and imaginary parts (zI = 0 here)
    zR = center + radius

    # Compute e*sin(zR) from precomputed quantities
    tmpsin = sinC*ecosRadius+cosC*esinRadius

    # Compute f(z(x)) in real and imaginary parts (fxI = 0)
    fxR = zR - tmpsin - ell_array

     # Add to arrays, with factor of 1/2 since an edge
    ft_gx2 = 0.5/fxR
    ft_gx1 = 0.5/fxR

    ## Compute j = 1 to N_points pieces

    # Compute z in real and imaginary parts
    zR = center + radius*exp2R
    zI = radius*exp2I

    # Compute f(z(x)) in real and imaginary parts
    # can use precomputed cosh / sinh / cos / sin for this!
    tmpsin = sinC*ecosR+cosC*esinR # e sin(zR)
    tmpcos = cosC*ecosR-sinC*esinR # e cos(zR)

    fxR = zR - tmpsin*coshI-ell_array
    fxI = zI - tmpcos*sinhI

    # Compute 1/f(z) and append to array
    ftmp = fxR*fxR+fxI*fxI;
    fxR /= ftmp;
    fxI /= ftmp;

    ft_gx2 += np.sum(exp4R*fxR+exp4I*fxI,axis=0)
    ft_gx1 += np.sum(exp2R*fxR+exp2I*fxI,axis=0)

    ## Separate out j = N_it piece, which is simpler

    # Compute z in real and imaginary parts (zI = 0 here)
    zR = center - radius

    # Compute sin(zR) from precomputed quantities
    tmpsin = sinC*ecosRadius-cosC*esinRadius

    # Compute f(z(x)) in real and imaginary parts (fxI = 0 here)
    fxR = zR - tmpsin-ell_array

    # Add to sum, with 1/2 factor for edges
    ft_gx2 += 0.5/fxR;
    ft_gx1 += -0.5/fxR;

    ### Compute and return the solution E(ell,e)
    output += radius*ft_gx2/ft_gx1;

    return output


def bodyPos(pxs, pys, l, q):  # given the displacements transform to c.o.m. frame
    px1s = pxs*q/(1+q)
    px2s = -pxs/(1+q)
    py1s = pys*q/(1+q)
    py2s = -pys/(1+q)
    pxls = -pxs*(l-q)/((1+l)*(1+q))
    pyls = -pys*(l-q)/((1+l)*(1+q))
    return px1s, py1s, px2s, py2s, pxls, pyls


def binaryMotion(ts, P, q, l, a, e, vTheta, vPhi, tPeri=0):  # binary position (in projected AU)
    etas = findEtas(ts, P, e, tPeri=tPeri)
    phis = 2*np.arctan(np.sqrt((1+e)/(1-e))*np.tan(etas/2)) % (2*np.pi)
    vPsis = vPhi-phis
    rs = a*(1-e*np.cos(etas))
    g = np.power(1-(np.cos(vPhi)**2)*(np.sin(vTheta)**2), -0.5)
    # projected positions in the c.o.m frame (in AU)
    pxs = rs*g*(np.cos(phis)-np.cos(vPsis)*np.cos(vPhi)*(np.sin(vTheta)**2))
    pys = rs*g*np.sin(phis)*np.cos(vTheta)
    # positions of sources 1 and 2 and the center of light
    px1s, py1s, px2s, py2s, pxls, pyls = bodyPos(pxs, pys, l, q)
    # x, y posn of each body and c.o.l.
    # in on-sky coords such that x is projected onto i dirn and y has no i component
    return px1s, py1s, px2s, py2s, pxls, pyls

# ----------------------
# -Analytic solutions
# ----------------------
def sigmagamma(eta1, eta2):
    deta = eta2-eta1
    sigma1 = (np.sin(eta2)-np.sin(eta1))/deta
    sigma2 = (np.sin(2*eta2)-np.sin(2*eta1))/deta
    sigma3 = (np.sin(3*eta2)-np.sin(3*eta1))/deta
    gamma1 = (np.cos(eta2)-np.cos(eta1))/deta
    gamma2 = (np.cos(2*eta2)-np.cos(2*eta1))/deta
    gamma3 = (np.cos(3*eta2)-np.cos(3*eta1))/deta
    return sigma1, sigma2, sigma3, gamma1, gamma2, gamma3

def dtheta_simple(ps):
    # assuming ~uniform sampling of pos between t1 and t2 can estimate UWE
    # CURRENTLY HAS A BUG I HAVEN'T CHASED DOWN GIVING NANS SOMETIMES
    if ps.Delta == -1:
        _ = Delta(ps)
    Omega = np.sqrt(1-(np.cos(ps.vphi)**2) * (np.sin(ps.vtheta)**2))
    Kappa = np.sin(ps.vphi)*np.cos(ps.vphi)*(np.sin(ps.vtheta)**2)
    pre = ps.parallax*ps.Delta*ps.a/Omega
    #print('pre: ',pre)

    epsx = -pre*(3/2)*ps.e*Omega**2
    epsy = 0

    epsxsq = (pre**2)*(Omega**4)*(1/2)*(1+2*ps.e**2)
    epsysq = (pre**2)*(np.cos(ps.vtheta)**2)*(1/2)*(1-ps.e**2)
    return np.sqrt(epsxsq+epsysq-(epsx**2)-(epsy**2))

def dtheta_full(ps, t1, t2):
    # assuming ~uniform sampling of pos between t1 and t2 can estimate UWE
    # CURRENTLY HAS A BUG I HAVEN'T CHASED DOWN GIVING NANS SOMETIMES
    if ps.Delta == -1:
        _ = Delta(ps)
    eta1 = findEtas(t1, ps.period, ps.e, tPeri=ps.tperi)
    eta2 = findEtas(t2, ps.period, ps.e, tPeri=ps.tperi)
    sigma1, sigma2, sigma3, gamma1, gamma2, gamma3 = sigmagamma(eta1, eta2)
    # print(sigma1,sigma2,sigma3,gamma1,gamma2,gamma3)
    # expected
    nu = 1/(1-ps.e*sigma1)
    #print('nu: ',nu)
    Omega = np.sqrt(1-(np.cos(ps.vphi)**2) * (np.sin(ps.vtheta)**2))
    Kappa = np.sin(ps.vphi)*np.cos(ps.vphi)*(np.sin(ps.vtheta)**2)
    pre = ps.parallax*ps.Delta*ps.a/Omega
    #print('pre: ',pre)

    epsx1 = (1+ps.e**2)*sigma1 - ps.e*(1.5 + sigma2/4)
    epsx2 = gamma1-ps.e*gamma2/4
    epsx = nu*pre*(epsx1*Omega**2 + Kappa*np.sqrt(1-ps.e**2)*epsx2)

    epsy = -nu*pre*np.cos(ps.vtheta)*np.sqrt(1-ps.e**2)*(gamma1-ps.e*gamma2/4)

    epsxsq1 = (1+2*ps.e**2)*(0.5+sigma2/4)-ps.e*(2+ps.e**2)*sigma1
    -ps.e*(3*sigma1/4 + sigma3/12)+ps.e**2
    epsxsq2 = (1+ps.e**2)*(gamma2/4)-ps.e*gamma1-ps.e*(gamma1/4 + gamma3/12)
    epsxsq3 = 0.5-sigma2/4-ps.e*(sigma1/4 - sigma3/12)
    epsxsq = nu*(pre**2)*((Omega**4)*epsxsq1
                          + 2*(Omega**2)*Kappa*np.sqrt(1-ps.e**2)*epsxsq2
                          + (Kappa**2)*(1-ps.e**2)*epsxsq3)

    epsysq = nu*(pre**2)*(np.cos(ps.vtheta)**2)*(1-ps.e**2) * \
        (0.5 - sigma2/4 - ps.e*(sigma1/4 - sigma3/12))
    return np.sqrt(epsxsq+epsysq-(epsx**2)-(epsy**2))

# ----------------------
# -Utilities
# ----------------------
# returns a number to a given significant digits (if extra true also returns exponent)


def sigString(number, significantFigures, extra=False):
    roundingFactor = significantFigures - int(np.floor(np.log10(np.abs(number)))) - 1
    rounded = np.round(number, roundingFactor)
    # np.round retains a decimal point even if the number is an integer (i.e. we might expect 460 but instead get 460.0)
    if roundingFactor <= 0:
        rounded = rounded.astype(int)
    string = rounded.astype(str)
    if extra == False:
        return string
    if extra == True:
        return string, roundingFactor

# generating, sampling and fitting a split normal (see https://authorea.com/users/107850/articles/371464-direct-parameter-finding-of-the-split-normal-distribution)
def splitNormal(x, mu, sigma, cigma):
    epsilon = cigma/sigma
    alphas = sigma*np.ones_like(x)
    alphas[x > mu] = cigma
    return (1/np.sqrt(2*np.pi*sigma**2))*(2/(1+epsilon))*np.exp(-0.5*((x-mu)/alphas)**2)


def splitInverse(F, mu, sigma, cigma):  # takes a random number between 0 and 1 and returns draw from split normal
    epsilon = cigma/sigma
    # print(cigma)
    alphas = np.ones_like(F)
    alphas[F > 1/(1+epsilon)] = cigma
    alphas[F < 1/(1+epsilon)] = sigma
    betas = np.ones_like(F)
    betas[F > (1/(1+epsilon))] = 1/epsilon
    return mu + np.sqrt(2*alphas**2)*scipy.special.erfinv(betas*((1+epsilon)*F - 1))


def splitFit(xs):  # fits a split normal distribution to an array of data
    xs = np.sort(xs)
    N = xs.size
    Delta = int(N*stdErf)  # hardcoded version of erf(1/sqrt(2))

    js = np.arange(1, N-Delta-2)
    w_js = xs[js+Delta]-xs[js]
    J = np.argmin(w_js)
    w_J = w_js[J]
    x_J = xs[J]

    ks = np.arange(J+1, J+Delta-2)
    theta_ks = (ks/N) - ((xs[ks]-x_J)/w_J)

    theta_kms = ((ks-1)/N) - ((xs[ks-1]-x_J)/w_J)
    theta_kps = ((ks+1)/N) - ((xs[ks+1]-x_J)/w_J)
    K = ks[np.argmin(np.abs(theta_ks-np.median(theta_ks)))]
    mu = xs[K]
    sigma = mu-x_J
    cigma = w_J-sigma

    beta = w_J/(xs[ks]-x_J)
    phi_ks = ((ks-J)/Delta) - (stdErf*(beta-1)/beta)
    Z = ks[np.argmin(np.abs(phi_ks))]

    return xs[Z], sigma, cigma
