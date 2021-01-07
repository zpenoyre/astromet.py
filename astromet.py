import numpy as np
import astropy.coordinates
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import get_body_barycentric

# All units SI
mSun = 1.9891e30
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
T = 365.2422
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
        self.pmRA = 0  # mas/year
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


# epoch - zero time of observations in BJD (default is dr3 epoch 2016.0 CE)
epoch = 2457388.5000000
# I'm v. open to suggestion about better ways to set epoch!


def setEpoch(newEpoch):
    global epoch
    if isinstance(newEpoch, str):
        if 'dr3' in newEpoch.lower():
            epoch = 2457388.50
        if 'dr2' in newEpoch.lower():
            epoch = 2457206.37
        if 'dr1' in newEpoch.lower():
            epoch = 2457023.50
    else:
        epoch = newEpoch


def path(ts, ps, comOnly=False, t0=0):
    N = ts.size
    xij = XijSimple(ts, ps.RA*np.pi/180, ps.Dec*np.pi/180, t0=t0)
    r = np.array([0, 0, ps.pmRA, ps.pmDec, ps.pllx])
    pos = xij@r
    ras, decs = ps.RA+mas*pos[:N], ps.Dec+mas*pos[N:]
    if comOnly == True:
        return ras, decs

    # extra c.o.l. correction due to binary
    px1s, py1s, px2s, py2s, pxls, pyls = binaryMotion(
        ts-ps.tPeri, ps.M, ps.q, ps.l, ps.a, ps.e, ps.vTheta, ps.vPhi)
    rls = mas*ps.pllx*(pxls*np.cos(ps.vOmega)+pyls*np.sin(ps.vOmega))
    dls = mas*ps.pllx*(pyls*np.cos(ps.vOmega)-pxls*np.sin(ps.vOmega))

    return ras+rls, decs+dls


'''def comPath(ts, ps, t0=0):
    dras, ddecs = comSimple(ts, ps.RA*np.pi/180, ps.Dec*np.pi /
                            180, ps.pmRA, ps.pmDec, ps.pllx, t0=t0)
    ras = ps.RA+dras
    decs = ps.Dec+ddecs
    return ras, decs'''

# For more details on the fit see section 1 of Hogg, Bovy & Lang 2010


def fit(ts, ras, decs, astError=1, t0=0):
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
    xij = XijSimple(ts-t0, medRa*np.pi/180, medDec*np.pi/180)
    # Astrometry covariance matrix
    cov = np.linalg.inv(xij.T@astPrec@xij)
    params = cov@xij.T@astPrec@np.hstack([diffRa, diffDec])
    return params, cov


'''def fit(ts, ras, decs, astError=1, t0=0):
    medRa = np.median(ras)
    medDec = np.median(decs)
    diffRa = (ras-medRa)/mas
    diffDec = (decs-medDec)/mas
    xij = XijSimple(ts-t0, medRa*np.pi/180, medDec*np.pi/180)
    inv = np.linalg.inv(xij.T@xij)
    params = inv@xij.T@np.hstack([diffRa, diffDec])
    if np.isscalar(astError): # scalar astrometric error given
        astError = np.diag(astError*np.ones(2*ts.size))
    if len(astError.shape)==1: # vector astrometric error given
        astError = np.diag(astError)
    paramError = inv@xij.T@(astError**2)@xij@inv
    return params, paramError'''

def uwe(ts,ras,decs,fitParams,astError=1):
    nTs=ts.size
    medRa = np.median(ras)
    medDec = np.median(decs)
    # Design matrix
    xij = XijSimple(ts, medRa*np.pi/180, medDec*np.pi/180)
    comPath=np.hstack([medRa*np.ones(nTs),medDec*np.ones(nTs)])

    dRas=ras-medRa-mas*(xij@fitParams)[:nTs]
    dDecs=decs-medDec-mas*(xij@fitParams)[nTs:]
    diff=np.sqrt(dRas**2 + dDecs**2)
    return np.sqrt(np.sum((diff/(mas*astError))**2)/(nTs-5))

def period(ps):
    totalMass = ps.M*(1+ps.q)
    return np.sqrt(4*(np.pi**2)*(ps.a**3)/(Galt*totalMass))

# ----------------
# -On-sky motion
# ----------------


def XijSimple(ts, ra, dec, t0=0):
    N = ts.size
    bs = barycentricPosition(ts, bjdStart=epoch)
    p0 = np.array([-np.sin(ra), np.cos(ra), 0])
    q0 = np.array([-np.cos(ra)*np.sin(dec), -np.sin(ra)*np.sin(dec), np.cos(dec)])
    xij = np.zeros((2*N, 5))
    xij[:N, 0] = 1
    xij[N:, 1] = 1
    xij[:N, 2] = ts-t0
    xij[N:, 3] = ts-t0
    xij[:N, 4] = -(1/np.cos(dec))*np.dot(bs, p0)
    xij[N:, 4] = -np.dot(bs, q0)
    return xij


def barycentricPosition(time, bjdStart=epoch):  # time in years after gaia start (2456863.94 BJD)
    t = time*T + bjdStart
    poss = astropy.coordinates.get_body_barycentric('earth', astropy.time.Time(t, format='jd'))
    xs = poss.x.value  # all in AU
    ys = poss.y.value
    zs = poss.z.value
    # gaia satellite is at Earth-Sun L2
    l2corr = 1+np.power(3e-6/3, 1/3)  # 3(.003)e-6 is earth/sun mass ratio
    return l2corr*np.vstack([xs, ys, zs]).T


# binary orbit


def findEtas(ts, M, a, e, tPeri=0):  # finds an (approximate) eccentric anomaly (see Penoyre & Sandford 2019, appendix A)
    eta0s = np.sqrt(Galt*M/(a**3))*(ts-tPeri)
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
    delta = np.abs(q-l)/((1+q)*(1+l))
    etas = findEtas(ts, M*(1+q), a, e, tPeri=tPeri)
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

# generating, sampling and fitting a split normal (see https://authorea.com/users/107850/articles/371464-direct-parameter-finding-of-the-split-normal-distribution?commit=ad3d419474f75af951a55c40481506c5a3d1a5e4)


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
