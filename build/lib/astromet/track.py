import numpy as np
import astropy.coordinates
from astropy import units as u
from astropy.time import Time

# All units SI
mSun = 2e30
lSun = 3.826e26
kpc = 3e19
Gyr = 3.15e16
day = 24*(60**2)
G = 6.67e-11
AU = 1.496e+11
c = 299792458
# e - eccentricity of Earth's orbit
e = 0.0167
# T - year in days
T = 365.242199
# year - year in seconds
year = T*day
# AU_c - time taken (here given in days) for light to travel from sun to Earth
AU_c = AU/(c*day)
# G in units of AU, years and solar masses
Galt = G * AU**-3 * mSun**1 * year**2
# mas2rad - conversion factor which multiplies a value in milli-arcseconds to give radians
mas2rad = 4.8481368110954e-9
# mas - conversion factor which multiplies a value in milli-arcseconds to give degrees
mas = mas2rad*180/np.pi


# ----------------
# -User functions
# ----------------
class params():
    def __init__(self):
        # astrometric parameters
        self.RA = 45  # degree
        self.Dec = 45  # degree
        self.pmRAc = 0  # mas/year
        self.pmDec = 0  # mas/year
        self.pllx = 1  # mas
        # binary parameters
        self.M = 1  # solar mass
        self.a = 1  # AU
        self.e = 0
        self.q = 0
        self.l = 0  # assumed < 1 (though may not matter)
        self.vTheta = 45
        self.vPhi = 45
        self.vOmega = 0
        self.tPeri = 0  # years

        # Below are assumed to be derived from other params
        # (I.e. not(!) specified by user)
        self.P = -1
        self.Delta = -1

        # the epoch determines when RA and Dec (and other astrometry)
        # are centred - for dr3 it's 2016.0, dr2 2015.5, dr1 2015.0
        self.epoch=2016.0

def bjyr_to_bjd(jyrdate):
    return 1721057.5 + jyrdate*T
def bjd_to_bjyr(bjddate):
    return (bjddate - 1721057.5)/T

def period(ps):
    totalMass = ps.M*(1+ps.q)
    ps.period=np.sqrt(4*(np.pi**2)*(ps.a**3)/(Galt*totalMass))
    return ps.period

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
    N = ts.size
    design_ts=np.hstack([ts,ts])
    design_phis=np.hstack([90*np.ones_like(ts),np.zeros_like(ts)])
    xij = design_matrix(design_ts,design_phis,ps.RA,ps.Dec, epoch=ps.epoch)

    RAc0=ps.RA*np.cos(np.deg2rad(ps.RA))/mas # RAcos(Dec) in mas
    Dec0=ps.Dec/mas # Dec in mas

    r = np.array([RAc0, Dec0, ps.pllx, ps.pmRAc, ps.pmDec])
    pos = xij@r # all in mas
    ras, decs = pos[:N], pos[N:]
    if comOnly == True:
        return ras, decs

    # extra c.o.l. correction due to binary
    px1s, py1s, px2s, py2s, pxls, pyls = binaryMotion(
        ts-ps.tPeri, ps.M, ps.q, ps.l, ps.a, ps.e, ps.vTheta, ps.vPhi)
    rls = ps.pllx*(pxls*np.cos(ps.vOmega)+pyls*np.sin(ps.vOmega))
    dls = ps.pllx*(pyls*np.cos(ps.vOmega)-pxls*np.sin(ps.vOmega))
    if allComponents==False:
        return ras+rls, decs+dls # return just the position fo the c.o.l.
    else:
        r1s = ps.pllx*(px1s*np.cos(ps.vOmega)+py1s*np.sin(ps.vOmega))
        d1s = ps.pllx*(py1s*np.cos(ps.vOmega)-px1s*np.sin(ps.vOmega))
        r2s = ps.pllx*(px2s*np.cos(ps.vOmega)+py2s*np.sin(ps.vOmega))
        d2s = ps.pllx*(py2s*np.cos(ps.vOmega)-px2s*np.sin(ps.vOmega))
        return ras+rls, decs+dls, ras+r1s, decs+d1s, ras+r2s, decs+d2s

def mock_obs(phis, racs, decs, errs=0):
    """
    Converts positions to comparable observables to real astrometric measurements
    (i.e. 1D psoitions along some scan angle, optionlly with errors added)
    Args:
        - ts,       ndarray - Observation times, jyear.
        - phis,     ndarray - Scanning angles (0 north, 90 east), degrees.
        - racs,     ndarray - RAcosDec at each scan, mas
        - decs,     ndarray - Dec at each scan, mas
        - errs,     float or ndarray - optional normal distributed error to be added
    Returns:
        - xs        ndarray - 1D projected displacements
    """
    xs=racs*np.sin(np.deg2rad(phis)) + decs*np.cos(np.deg2rad(phis)) + errs*np.random.randn(phis.size)
    return xs

# ----------------
# -On-sky motion
# ----------------


'''def XijSimple(ts, ra, dec, epoch=2016.0):
    N = ts.size
    bs = barycentricPosition(ts)
    p0 = np.array([-np.sin(ra), np.cos(ra), 0])
    q0 = np.array([-np.cos(ra)*np.sin(dec), -np.sin(ra)*np.sin(dec), np.cos(dec)])
    xij = np.zeros((2*N, 5))
    xij[:N, 0] = 1
    xij[N:, 1] = 1
    xij[:N, 2] = ts-epoch
    xij[N:, 3] = ts-epoch
    xij[:N, 4] = -(1/np.cos(dec))*np.dot(bs, p0)
    xij[N:, 4] = -np.dot(bs, q0)
    return xij'''


def design_matrix(ts, phis, ra, dec, epoch=2016.0):
    """
    Iterative optimization to fit astrometric solution in AGIS (outer iteration)
    Lindegren 2012
    See Everall+ 2021
    Args:
        - t,       ndarray - Observation times, jyear.
        - phis,     ndarray - scan angles.
        - ra, dec,  float - reference right ascension and declination of source, deg
        - epoch     float - time at which position and pm are measured, years CE
    Returns:
        - design, ndarry - Design matrix
    """
    ra, dec = np.deg2rad(ra), np.deg2rad(dec)
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

    return design


def barycentricPosition(time):
    pos = astropy.coordinates.get_body_barycentric('earth', astropy.time.Time(time, format='jyear'))
    xs = pos.x.value  # all in AU
    ys = pos.y.value
    zs = pos.z.value
    # gaia satellite is at Earth-Sun L2
    l2corr = 1+np.power(3e-6/3, 1/3)  # 3(.003)e-6 is earth/sun mass ratio
    return l2corr*np.vstack([xs, ys, zs]).T


# binary orbit


def findEtas(ts, P, e, tPeri=0):  # finds an (approximate) eccentric anomaly (see Penoyre & Sandford 2019, appendix A)
    eta0s = (2*np.pi/P)*(ts-tPeri)
    eta1s = e*np.sin(eta0s)
    eta2s = (e**2)*np.sin(eta0s)*np.cos(eta0s)
    eta3s = (e**3)*np.sin(eta0s)*(1-(3/2)*np.sin(eta0s)**2)
    return eta0s+eta1s+eta2s+eta3s


def bodyPos(pxs, pys, l, q):  # given the displacements transform to c.o.m. frame
    px1s = pxs*q/(1+q)
    px2s = -pxs/(1+q)
    py1s = pys*q/(1+q)
    py2s = -pys/(1+q)
    pxls = -pxs*(l-q)/((1+l)*(1+q))
    pyls = -pys*(l-q)/((1+l)*(1+q))
    return px1s, py1s, px2s, py2s, pxls, pyls


def binaryMotion(ts, M, q, l, a, e, vTheta, vPhi, tPeri=0):  # binary position (in projected AU)
    totalMass = M*(1+q)
    P = np.sqrt(4*(np.pi**2)*(a**3)/(Galt*totalMass))
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
# -Analytic solutions (written significantly later - need to go back at some point and double-check for conflicts/ duplications)
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
    if ps.P == -1:
        _ = period(ps)
    if ps.Delta == -1:
        _ = Delta(ps)
    Omega = np.sqrt(1-(np.cos(ps.vPhi)**2) * (np.sin(ps.vTheta)**2))
    Kappa = np.sin(ps.vPhi)*np.cos(ps.vPhi)*(np.sin(ps.vTheta)**2)
    pre = ps.pllx*ps.Delta*ps.a/Omega
    #print('pre: ',pre)

    epsx = -pre*(3/2)*ps.e*Omega**2
    epsy = 0

    epsxsq = (pre**2)*(Omega**4)*(1/2)*(1+2*ps.e**2)
    epsysq = (pre**2)*(np.cos(ps.vTheta)**2)*(1/2)*(1-ps.e**2)

    return np.sqrt(epsxsq+epsysq-(epsx**2)-(epsy**2))

def dtheta_full(ps, t1, t2):
    # assuming ~uniform sampling of pos between t1 and t2 can estimate UWE
    if ps.P == -1:
        _ = period(ps)
    if ps.Delta == -1:
        _ = Delta(ps)
    eta1 = findEtas(t1, ps.P, ps.e, tPeri=ps.tPeri)
    eta2 = findEtas(t2, ps.P, ps.e, tPeri=ps.tPeri)
    sigma1, sigma2, sigma3, gamma1, gamma2, gamma3 = sigmagamma(eta1, eta2)
    # print(sigma1,sigma2,sigma3,gamma1,gamma2,gamma3)
    # expected
    nu = 1/(1-ps.e*sigma1)
    #print('nu: ',nu)
    Omega = np.sqrt(1-(np.cos(ps.vPhi)**2) * (np.sin(ps.vTheta)**2))
    Kappa = np.sin(ps.vPhi)*np.cos(ps.vPhi)*(np.sin(ps.vTheta)**2)
    pre = ps.pllx*ps.Delta*ps.a/Omega
    #print('pre: ',pre)

    epsx1 = (1+ps.e**2)*sigma1 - ps.e*(1.5 + sigma2/4)
    epsx2 = gamma1-ps.e*gamma2/4
    epsx = nu*pre*(epsx1*Omega**2 + Kappa*np.sqrt(1-ps.e**2)*epsx2)

    epsy = -nu*pre*np.cos(ps.vTheta)*np.sqrt(1-ps.e**2)*(gamma1-ps.e*gamma2/4)

    epsxsq1 = (1+2*ps.e**2)*(0.5+sigma2/4)-ps.e*(2+ps.e**2)*sigma1
    -ps.e*(3*sigma1/4 + sigma3/12)+ps.e**2
    epsxsq2 = (1+ps.e**2)*(gamma2/4)-ps.e*gamma1-ps.e*(gamma1/4 + gamma3/12)
    epsxsq3 = 0.5-sigma2/4-ps.e*(sigma1/4 - sigma3/12)
    epsxsq = nu*(pre**2)*((Omega**4)*epsxsq1
                          + 2*(Omega**2)*Kappa*np.sqrt(1-ps.e**2)*epsxsq2
                          + (Kappa**2)*(1-ps.e**2)*epsxsq3)

    epsysq = nu*(pre**2)*(np.cos(ps.vTheta)**2)*(1-ps.e**2) * \
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


# ----------------------------------------------
# - Legacy (old code I'm keeping for reference)
# ----------------------------------------------

'''def uweObs(ts, racs, decs, r5d, phis=None, astError=1):
    nTs = ts.size
    medDec = mas*np.median(decs) # deg
    medRa = mas*np.median(racs)/np.cos(np.deg2rad(medDec)) # deg
    # Design matrix
    design_ts=np.hstack([ts,ts])
    design_phis=np.hstack([90*np.ones_like(ts),np.zeros_like(ts)])
    xij = design_matrix(design_ts,design_phis,ps.RA,ps.Dec, epoch=ps.epoch)

    xij = XijSimple(ts, medRa*np.pi/180, medDec*np.pi/180)

    dRas = ras-medRa-mas*(xij@fitParams)[:nTs]
    dDecs = decs-medDec-mas*(xij@fitParams)[nTs:]
    diff = np.sqrt(dRas**2 + dDecs**2)
    return np.sqrt(np.sum((diff/(mas*astError))**2)/(nTs-5))'''
'''# For more details on the fit see section 1 of Hogg, Bovy & Lang 2010
def fit(ts, ras, decs, astError=1):
    # Error precision matrix
    if np.isscalar(astError):  # scalar astrometric error given
        astPrec = np.diag((astError**-2)*np.ones(2*ts.size))
    elif len(astError.shape) == 1:  # vector astrometric error given
        astPrec = np.diag((astError**-2))
    else:
        astPrec = astError**-2
    # convenient to work entirely in mas, relative to median RA and Dec
    medRa = np.median(ras)
    medDec = np.median(decs)
    diffRa = (ras-medRa)/mas
    diffDec = (decs-medDec)/mas
    # Design matrix
    xij = XijSimple(ts, medRa*np.pi/180, medDec*np.pi/180)
    # Astrometry covariance matrix
    cov = np.linalg.inv(xij.T@astPrec@xij)
    params = cov@xij.T@astPrec@np.hstack([diffRa, diffDec])
    # all parameters in mas(/yr) - ra and dec give displacement *from median*
    return params, cov'''
'''
# T0 - interval between last periapse before survey (2456662.00 BJD)
# and start of survey (2456863.94 BJD) in days
T0 = 201.938

def path(ts,ps,t0=0):
    # need to transofrm to eclitpic coords centered on periapse
    # (natural frame for parralax ellipse) to find on-sky c.o.m motion
    azimuth,polar,pmAzimuth,pmPolar=icrsToPercientric(ps.RA,ps.Dec,ps.pmRA,ps.pmDec)
    # centre of mass motion in pericentric frame in mas
    dAz,dPol=comMotion(ts,polar*np.pi/180,azimuth*np.pi/180,pmPolar,pmAzimuth,ps.pllx)
    # and then tranform back
    ras,decs=pericentricToIcrs(azimuth+mas*dAz,polar+mas*dPol)

    # extra c.o.l. correction due to binary
    px1s,py1s,px2s,py2s,pxls,pyls=binaryMotion(ts-ps.tPeri,ps.M,ps.q,ps.l,ps.a,ps.e,ps.vTheta,ps.vPhi)
    rls=mas*ps.pllx*(pxls*np.cos(ps.vOmega)+pyls*np.sin(ps.vOmega))
    dls=mas*ps.pllx*(pyls*np.cos(ps.vOmega)-pxls*np.sin(ps.vOmega))

    return ras+rls,decs+dls

# c.o.m motion in mas - all time in years, all angles mas except phi and theta (rad)
# needs azimuth and polar (0 to pi) in ecliptic coords with periapse at azimuth=0
def comMotion(ts,polar,azimuth,muPolar,muAzimuth,pllx):
    taus=2*np.pi*ts+(T0/T)
    tau0=2*np.pi*T0/T
    psis=azimuth-taus
    psi0=azimuth-tau0
    dAs=((ts-AU_c*np.cos(polar)*(np.cos(psis)-np.cos(psi0)
        +e*(np.sin(taus)*np.sin(psis) - np.sin(tau0)*np.sin(psi0))))*muAzimuth
        -(pllx/np.cos(polar))*(np.cos(psis)+e*(np.sin(taus)*np.sin(psis)-np.cos(azimuth))))
    dDs=((ts-AU_c*np.cos(polar)*(np.cos(psis)-np.cos(psi0)
        +e*(np.sin(taus)*np.sin(psis) - np.sin(tau0)*np.sin(psi0))))*muPolar
        -pllx*np.sin(polar)*(np.sin(psis)+e*(np.sin(taus)*np.cos(psis)+np.sin(azimuth))))
    return dAs,dDs

# c.o.m motion in mas - all time in years, all angles mas except phi and theta (rad)
# needs azimuth and polar (0 to pi) in ecliptic coords with periapse at azimuth=0
def comSimple(ts, ra, dec, pmRa, pmDec, pllx, t0=0):
    bs = barycentricPosition(ts)
    p0 = np.array([-np.sin(ra), np.cos(ra), 0])
    q0 = np.array([-np.cos(ra)*np.sin(dec), -np.sin(ra)*np.sin(dec), np.cos(dec)])
    deltaRa = pmRa*(ts-t0) - (pllx/np.cos(dec))*np.dot(bs, p0)
    deltaDec = pmDec*(ts-t0) - pllx*np.dot(bs, q0)
    return mas*deltaRa, mas*deltaDec

# 'pericentric' frame is in the ecliptic plane, with azimuth=0 at periapse
def icrsToPercientric(ra,dec,pmra=0,pmdec=0):
    coord=astropy.coordinates.SkyCoord(ra=ra*u.degree, dec=dec*u.degree,
        pm_ra_cosdec=pmra*np.cos(dec*np.pi/180)*u.mas/u.yr,
        pm_dec=pmdec*u.mas/u.yr, frame='icrs')
    bary=coord.barycentrictrueecliptic
    polar=bary.lat.degree
    azimuth=bary.lon.degree+75 # 75° is offset from periapse to equinox
    if (pmra==0) & (pmdec==0):
        return azimuth,polar
    else:
        pmPolar=bary.pm_lat.value # in mas/yr
        pmAzimuth=bary.pm_lon_coslat.value/np.cos(polar*np.pi/180)
        return azimuth,polar,pmAzimuth,pmPolar
def pericentricToIcrs(az,pol,pmaz=0,pmpol=0):
    coords=astropy.coordinates.SkyCoord(lon=(az-75)*u.degree, lat=pol*u.degree,
    pm_lon_coslat=pmaz*np.cos(pol*np.pi/180)*u.mas/u.yr,
    pm_lat=pmpol*u.mas/u.yr, frame='barycentrictrueecliptic')
    icrs=coords.icrs
    ra=icrs.ra.degree
    dec=icrs.dec.degree
    if (pmaz==0) & (pmpol==0):
        return ra,dec
    else:
        pmDec=bary.pm_dec.value # in mas/yr
        pmRa=bary.pm_ra_cosdec.value/np.cos(dec*np.pi/180)
        return ra,dec,pmRa,pmDec

def Xij(ts,phi,theta):
    N=ts.size
    taus=2*np.pi*ts+(T0/T)
    tau0=2*np.pi*T0/T
    psis=phi-taus
    psi0=phi-tau0
    tb=AU_c*np.cos(theta)*(np.cos(psis)-np.cos(psi0)
        +e*(np.sin(taus)*np.sin(psis) - np.sin(tau0)*np.sin(psi0)))
    xij=np.zeros((2*N,5))
    xij[:N,0]=1
    xij[N:,1]=1
    xij[:N,2]=ts-tb
    xij[N:,3]=ts-tb
    xij[:N,4]=-(1/np.cos(theta))*(np.cos(psis)+e*(np.sin(taus)*np.sin(psis)-np.cos(phi)))
    xij[N:,4]=-np.sin(theta)*(np.sin(psis)+e*(np.sin(taus)*np.cos(psis)+np.sin(phi)))
    return xij'''
