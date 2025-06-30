import numpy as np
import astropy.coordinates
from astropy import units as u
from astropy import constants
from astropy.time import Time
import scipy.interpolate
import sys
import os
from .lensing import *


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
earth_sun_mass_ratio = (constants.M_earth/constants.M_sun).value
tbegin = 2014.6670  # time (in years) of Gaia's first observations


# ----------------
# -User functions
# ----------------
class params():
    def __init__(self):
        # astrometric parameters
        self.ra = 45  # degree
        self.dec = 45  # degree
        self.drac = 0  # mas
        self.ddec = 0  # mas
        self.pmrac = 0  # mas/year
        self.pmdec = 0  # mas/year
        self.parallax = 0  # mas
        # binary parameters
        self.period = 1  # year
        self.a = 0  # AU
        self.e = 0.1
        self.q = 0
        self.l = 0  # assumed < 1 (though may not matter)
        self.vtheta = np.pi/4
        self.vphi = np.pi/4
        self.vomega = 0
        self.tperi = 0  # jyear
        # blend parameters
        self.blenddrac = 0  # mas
        self.blendddec = 0  # mas
        self.blendpmrac = 0  # mas/year
        self.blendpmdec = 0  # mas/year
        self.blendparallax = 0  # mas
        self.thetaE = 0  # mas
        self.blendl = 0

        # Below are assumed to be derived from other params
        # (I.e. not(!) specified by user)
        self.totalmass = -1  # solar mass
        self.Delta = -1

        # the epoch determines when RA and Dec (and other astrometry)
        # are centred - for dr3 it's 2016.0, dr2 2015.5, dr1 2015.0
        self.epoch = 2016.0


def bjyr_to_bjd(jyrdate):
    return 1721057.5 + jyrdate*T


def bjd_to_bjyr(bjddate):
    return (bjddate - 1721057.5)/T


def totalmass(ps):
    ps.totalmass = (4*(np.pi**2)/Galt)*((ps.a**3)/(ps.period**2))
    return ps.totalmass


def Delta(ps):
    ps.Delta = (ps.l-ps.q)/((1+ps.q)*(1+ps.l))
    return ps.Delta


def track(ts, ps, comOnly=False, allComponents=False):
    """
    Astrometric track in RAcos(Dec) and Dec [mas] for a given binary (or lensing event)
    Args:
        - ts,       ndarray - Observation times, jyear.
        - ps,       params object - Astrometric, binary and lensing parameters.
        - comOnly,  bool - If True return only c.o.m track (no binary)
        - allComponents,     bool - If True return pos. of c.o.l. & both components
    Returns:
        - racs      ndarry - RAcosDec at each time, mas
        - decs      ndarry - Dec at each time, mas
        (optionally) - mag_diff     ndarray - difference from baseline magnitude at each time (for lensing events)
    """
    xij = design_matrix(ts, np.deg2rad(ps.ra), np.deg2rad(ps.dec), epoch=ps.epoch)

    r5d = np.array([ps.drac, ps.ddec, ps.parallax, ps.pmrac, ps.pmdec])
    dracs, ddecs = xij@r5d  # all in mas

    if comOnly == False:
        # extra c.o.l. correction due to binary
        px1s, py1s, px2s, py2s, pxls, pyls = binaryMotion(
            ts, ps.period, ps.q, ps.l, ps.a, ps.e, ps.vtheta, ps.vphi,tPeri=ps.tperi)
        rls = ps.parallax*(pxls*np.cos(ps.vomega)+pyls*np.sin(ps.vomega))
        dls = ps.parallax*(pyls*np.cos(ps.vomega)-pxls*np.sin(ps.vomega))
        if allComponents == True or (ps.a > 0 and ps.thetaE > 0): # gets all 3 components
            r1s = ps.parallax*(px1s*np.cos(ps.vomega)+py1s*np.sin(ps.vomega))
            d1s = ps.parallax*(py1s*np.cos(ps.vomega)-px1s*np.sin(ps.vomega))
            r2s = ps.parallax*(px2s*np.cos(ps.vomega)+py2s*np.sin(ps.vomega))
            d2s = ps.parallax*(py2s*np.cos(ps.vomega)-px2s*np.sin(ps.vomega))
            if ps.thetaE == 0:
                return dracs+rls, ddecs+dls, dracs+r1s, ddecs+d1s, dracs+r2s, ddecs+d2s
            else: #lensing of both components
                r5d_blend = np.array([ps.blenddrac, ps.blendddec, ps.blendparallax, ps.blendpmrac, ps.blendpmdec])
                dracs_blend, ddecs_blend = xij@r5d_blend  # all in mas
                dracs_lbin, ddecs_lbin, mag_diff, dracs_1_lensed, ddecs_1_lensed, mag_diff_1, dracs_2_lensed, ddecs_2_lensed, mag_diff_2 = lensed_binary(ps, dracs+r1s, ddecs+d1s, dracs+r2s, ddecs+d2s, dracs_blend, ddecs_blend)
                if allComponents == True:
                    return dracs_lbin, ddecs_lbin, mag_diff, dracs_1_lensed, ddecs_1_lensed, mag_diff_1, dracs_2_lensed, ddecs_2_lensed, mag_diff_2
                return dracs_lbin, ddecs_lbin, mag_diff
        dracs, ddecs = dracs+rls, ddecs+dls # corrected track to be passed to lensing/blending functions

    if ps.thetaE > 0: #lensing
        # track of the lens
        r5d_blend = np.array([ps.blenddrac, ps.blendddec, ps.blendparallax, ps.blendpmrac, ps.blendpmdec])
        dracs_blend, ddecs_blend = xij@r5d_blend  # all in mas
        dracs_lensed, ddecs_lensed, mag_diff = onsky_lens(dracs, ddecs, dracs_blend, ddecs_blend, ps.thetaE, ps.blendl)
        return dracs_lensed, ddecs_lensed, mag_diff

    else:
        if ps.blendl > 0: # blending
            # track of the blend
            r5d_blend = np.array([ps.blenddrac, ps.blendddec, ps.blendparallax, ps.blendpmrac, ps.blendpmdec])
            dracs_blend, ddecs_blend = xij@r5d_blend  # all in mas
            dracs_blended, ddecs_blended = blend(dracs, ddecs, dracs_blend, ddecs_blend, ps.blendl)
            return dracs_blended, ddecs_blended

    return dracs, ddecs


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
    design[0, :, 0] = 1  # ra*cos(dec)
    design[1, :, 1] = 1  # dec
    design[0, :, 2] = -np.dot(p0, bs.T)  # parallax (ra component)
    design[1, :, 2] = -np.dot(q0, bs.T)  # parallax (dec component)
    design[0, :, 3] = ts-epoch  # pmra
    design[1, :, 4] = ts-epoch  # pmdec

    if np.size(phis) > 1:
        # sin and cos angles
        sina = np.sin(phis)
        cosa = np.cos(phis)

        # Construct design matrix
        design = design[0]*sina[:, None] + design[1]*cosa[:, None]

    return design




def barycentricPosition(time):
    pos = astropy.coordinates.get_body_barycentric('earth', astropy.time.Time(time, format='jyear'))
    xs = pos.x.value  # all in AU
    ys = pos.y.value
    zs = pos.z.value
    # gaia satellite is at Earth-Sun L2
    l2corr = 1+np.power(earth_sun_mass_ratio/3, 1/3)
    return l2corr*np.vstack([xs, ys, zs]).T


# binary orbit
def findEtas(ts, period, ecc, tPeri=0, N_it=10, precision=1e-5):
    # finds eccentric anomaly with iterative Halley's method
    phase=2*np.pi*(((ts-tPeri)/period) ) # deleted a %1 from brackets (24/10/23) - hope nothing breaks!

    sph=np.sin(phase)
    cph=np.cos(phase)
    # initial guess expanding to third order in eccentricity
    eta=phase + ecc*sph + (ecc**2)*sph*cph + 0.5*(ecc**3)*sph*(3*(cph**2)-1)
    deltaeta=1

    it=0
    while ((np.max(np.abs(deltaeta))>precision) & (it<N_it)):
    # Halley's method
        it+=1
        sineta=np.sin(eta)
        coseta=np.cos(eta)
        f  = eta - ecc*sineta - phase
        df = 1.  - ecc*coseta
        d2f= ecc*sineta
        deltaeta  = -f*df / (df*df - 0.5*f*d2f)
        eta      += deltaeta
    # since the Halley method converges cubically, a correction < precision=1e-5 at the current iteration
    # implies that it would be <~1e-15 at the next iteration, which is beyond the precision limit
    return eta

def findEtasHyperbolic(ts, tau, ecc, tPeri=0, N_it=10, precision=1e-5):
    # finds eccentric anomaly for unbound orbits with modified iterative Halley's method
    # initial guess
    phase=2*np.pi*(ts-tPeri)/tau
    zeta=np.arcsinh(phase/ecc)

    deltazeta=1

    it=0
    while ((np.max(np.abs(deltazeta))>precision) & (it<N_it)):
    # Halley's method
        it+=1
        sinhzeta=np.sinh(zeta)
        coshzeta=np.cosh(zeta)
        f  = ecc*sinhzeta - zeta - phase
        df = ecc*coshzeta - 1.
        d2f= ecc*sinhzeta
        deltazeta  = -f*df / (df*df - 0.5*f*d2f)
        zeta      += deltazeta
    # since the Halley method converges cubically, a correction < precision=1e-5 at the current iteration
    # implies that it would be <~1e-15 at the next iteration, which is beyond the precision limit
    return zeta

def findPhisParabolic(ts,tau,tPeri=0, N_it=10, precision=1e-5):
    alpha=3*(ts-tPeri)/tau
    phi=np.sign(alpha)*np.arccos((1-alpha**2)/(1+alpha**2))
    deltaphi=1
    it=0
    while ((np.max(np.abs(deltaphi))>precision) & (it<N_it)):
    # Halley's method
        it+=1
        sinphi=np.sin(phi)
        cosphi=np.cos(phi)
        f  = sinphi*(2+cosphi)/(1+cosphi)**2 - alpha
        df = 3/(1+cosphi)**2
        d2f= 6*sinphi/(1+cosphi)**3
        deltaphi  = -f*df / (df*df - 0.5*f*d2f)
        phi      += deltaphi
    return phi





def bodyPos(pxs, pys, l, q):  # given the displacements transform to c.o.m. frame
    px1s = -pxs*q/(1+q)
    px2s = pxs/(1+q)
    py1s = -pys*q/(1+q)
    py2s = pys/(1+q)
    pxls = pxs*(l-q)/((1+l)*(1+q))
    pyls = pys*(l-q)/((1+l)*(1+q))
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
# unfinished

def sigmagamma(eta1, eta2):
    deta = eta2-eta1
    sigma1 = (np.sin(eta2)-np.sin(eta1))/deta
    sigma2 = (np.sin(2*eta2)-np.sin(2*eta1))/deta
    sigma3 = (np.sin(3*eta2)-np.sin(3*eta1))/deta
    gamma1 = (np.cos(eta2)-np.cos(eta1))/deta
    gamma2 = (np.cos(2*eta2)-np.cos(2*eta1))/deta
    gamma3 = (np.cos(3*eta2)-np.cos(3*eta1))/deta
    return sigma1, sigma2, sigma3, gamma1, gamma2, gamma3


def sigmagammahat(eta1, eta2):
    deta = eta2-eta1
    sigmahat1 = (eta2*np.sin(eta2)-eta1*np.sin(eta1))/deta
    sigmahat2 = (eta2*np.sin(2*eta2)-eta1*np.sin(2*eta1))/deta
    gammahat1 = (eta2*np.cos(eta2)-eta1*np.cos(eta1))/deta
    gammahat2 = (eta2*np.cos(2*eta2)-eta1*np.cos(2*eta1))/deta
    return sigmahat1, sigmahat2, gammahat1, gammahat2


def dtheta_simple(ps):
    # assuming ~uniform sampling in time and one period observed
    if ps.Delta == -1:
        _ = Delta(ps)
    Omega = np.sqrt(1-(np.cos(ps.vphi)**2) * (np.sin(ps.vtheta)**2))
    Kappa = np.sin(ps.vphi)*np.cos(ps.vphi)*(np.sin(ps.vtheta)**2)
    pre = ps.parallax*ps.Delta*ps.a/Omega
    #print('pre: ',pre)

    epsx = -pre*(3/2)*ps.e*Omega**2
    epsy = 0

    epsxsq = (pre**2)*((Omega**4)*((1+4*ps.e**2)/2)
                       + (1/2)*(Kappa**2)*(1-ps.e**2))
    epsysq = (pre**2)*(np.cos(ps.vtheta)**2)*(1/2)*(1-ps.e**2)
    return np.sqrt(epsxsq+epsysq-(epsx**2)-(epsy**2))


def dtheta_full(ps, t1, t2, return_pm=False):
    # assuming ~uniform sampling in time between t1 and t2
    # and some known period
    if ps.Delta == -1:
        _ = Delta(ps)
    eta1 = findEtas(t1, ps.period, ps.e, tPeri=ps.tperi)
    eta2 = findEtas(t2, ps.period, ps.e, tPeri=ps.tperi)

    # using latest periapse time before t1
    tperi = ps.tperi+ps.period*np.floor((t1-ps.tperi)/ps.period)
    tm = (t1+t2)/2
    # findEtas always(?) returns values betwen 0 and 2*pi
    # for most uses this is what we want (eta mostly appears in trig.)
    # here however we don't want to lose fators of 2*pi
    ##print('\n__t1: ',t1,' t2: ',t2,' ps.tperi: ',ps.tperi,' tperi: ',tperi)
    ##print('eta1: ',eta1,' eta2: ',eta2)
    eta1 = eta1  # between 0 and 2 pi
    eta2 = eta2+2*np.pi*np.floor((t2-tperi)/ps.period)
    ##print('eta1: ',eta1,' eta2: ',eta2,'\n')

    sigma1, sigma2, sigma3, gamma1, gamma2, gamma3 = sigmagamma(eta1, eta2)
    sigmahat1, sigmahat2, gammahat1, gammahat2 = sigmagammahat(eta1, eta2)

    nu = 1/(1-ps.e*sigma1)

    av_s = nu*(-gamma1+(ps.e/4)*gamma2)
    av_c = nu*(-(ps.e/2)+sigma1-(ps.e/4)*sigma2)
    av_s2 = nu*((1/2)-(ps.e/4)*sigma1-(1/4)*sigma2+(ps.e/12)*sigma3)
    av_sc = nu*((ps.e/4)*gamma1-(1/4)*gamma2+(ps.e/12)*gamma3)
    av_c2 = nu*((1/2)-(3*ps.e/4)*sigma1+(1/4)*sigma2-(ps.e/12)*sigma3)
    av_etas = nu*(sigma1-(ps.e/8)*sigma2-gammahat1+(ps.e/4)*gammahat2)
    av_etac = nu*(gamma1-(ps.e/8)*gamma2-(ps.e/4)*(eta1+eta2)+sigmahat1-(ps.e/4)*sigmahat2)

    Omega = np.sqrt(1-(np.cos(ps.vphi)**2) * (np.sin(ps.vtheta)**2))
    Kappa = np.sin(ps.vphi)*np.cos(ps.vphi)*(np.sin(ps.vtheta)**2)

    pre = ps.parallax*ps.Delta*ps.a/Omega

    #epsx1 = (1+ps.e**2)*sigma1 - ps.e*(1.5 + sigma2/4)
    #epsxa = -(3*ps.e/2) + (1+ps.e**2)*sigma1 - (ps.e/4)*sigma2
    #epsxb = gamma1-(ps.e/4)*gamma2
    epsx = pre*((Omega**2)*(av_c-ps.e) - Kappa*np.sqrt(1-ps.e**2)*av_s)
    epsy = pre*np.cos(ps.vtheta)*np.sqrt(1-ps.e**2)*av_s

    ##print('analytical epsx: ',epsx,' and epsy: ',epsy)

    #epsxsq1 = (1+2*ps.e**2)*(0.5+sigma2/4)-ps.e*(2+ps.e**2)*sigma1
    #-ps.e*(3*sigma1/4 + sigma3/12)+ps.e**2
    #epsxsqa = ((1+4*ps.e**2)/2)-((11+4*ps.e**2)/4)*ps.e*sigma1\
    #+((1+2*ps.e**2)/4)*sigma2 - (ps.e/12)*sigma3
    #epsxsq2 = (1+ps.e**2)*(gamma2/4)-ps.e*gamma1-ps.e*(gamma1/4 + gamma3/12)
    #epsxsqb = (5*ps.e/4)*gamma1 - ((1+ps.e**2)/4)*gamma2 + (ps.e/12)*gamma3
    #epsxsq3 = 0.5-sigma2/4-ps.e*(sigma1/4 - sigma3/12)
    #epsxsqc = (1/2) - (ps.e/4)*sigma1 - (1/4)*sigma2 + (ps.e/12)*sigma3
    epsxsq = (pre**2)*((Omega**4)*(av_c2 - 2*ps.e*av_c + ps.e**2)
                       - 2*(Omega**2)*Kappa*np.sqrt(1-ps.e**2)*(av_sc-ps.e*av_s)
                       + (Kappa**2)*(1-ps.e**2)*av_s2)

    epsysq = (pre**2)*(np.cos(ps.vtheta)**2)*(1-ps.e**2)*av_s2

    av_epssq = epsxsq+epsysq
    aveps_sq = epsx**2 + epsy**2

    ##print('analytical eps2: ',epsxsq+epsysq)
    ##print('analytical dtheta2 old: ',epsxsq+epsysq-(epsx**2 + epsy**2))

    '''crossepsa=((4-(ps.e**2))/4)*gamma1+(ps.e/8)*gamma2-((ps.e**2)/12)*gamma3\
        +(1+(ps.e**2))*sigmahat1 - (ps.e/4)*sigmahat2 - (3*ps.e/4)*(eta1+eta2)
    print('old crossepsa: ',crossepsa)
    crossepsb=(ps.e/2) - ((4+ps.e**2)/4)*sigma1 - (ps.e/8)*sigma2\
        +(ps.e**2/12)*sigma3 + gammahat1 - (ps.e/4)*gammahat2

    av_t_c=(tperi-tm)*(-(ps.e/2)+sigma1-(ps.e/4)*sigma2)+(ps.period/(2*np.pi))*crossepsa
    av_t_s=(tperi-tm)*(-gamma1+(ps.e/4)*gamma2)+(ps.period/(2*np.pi))*crossepsb
    print('old av_t_s: ',av_t_s)
    print('old av_t_c: ',av_t_c)'''
    av_t_s = (tperi-tm)*av_s+(ps.period/(2*np.pi))*(av_etas-ps.e*av_s2)
    av_t_c = (tperi-tm)*av_c+(ps.period/(2*np.pi))*(av_etac-ps.e*av_sc)
    ##print('av_t_s: ',av_t_s)
    ##print('av_t_c: ',av_t_c)

    crossepsx = pre*((Omega**2)*av_t_c - Kappa*np.sqrt(1-ps.e**2)*av_t_s)
    crossepsy = pre*np.cos(ps.vtheta)*np.sqrt(1-ps.e**2)*av_t_s

    ##print('analytical av eta eps: ',crossepsx,crossepsy)

    eps1x = pre*((Omega**2)*(np.cos(eta1)-ps.e) - Kappa*np.sqrt(1-ps.e**2)*np.sin(eta1))
    eps2x = pre*((Omega**2)*(np.cos(eta2)-ps.e) - Kappa*np.sqrt(1-ps.e**2)*np.sin(eta2))
    # epscx=(eps1x+eps2x)/2

    eps1y = pre*np.cos(ps.vtheta)*np.sqrt(1-ps.e**2)*np.sin(eta1)
    eps2y = pre*np.cos(ps.vtheta)*np.sqrt(1-ps.e**2)*np.sin(eta2)

    epsdotx = (eps2x-eps1x)/(t2-t1)
    epsdoty = (eps2y-eps1y)/(t2-t1)
    ##print('old epsdot: ',epsdotx,epsdoty)
    epsdotx = 3*crossepsx/(tm**2 - t1*t2)
    epsdoty = 3*crossepsy/(tm**2 - t1*t2)

    ##print('analytical epsdot: ',epsdotx,epsdoty)
    #print('analytical av epsc2: ',epsdotterm)

    #crossepstermx=(tperi-tm)*epsx + (ps.period/(2*np.pi))*crossepsx
    #crossepstermy=(tperi-tm)*epsy + (ps.period/(2*np.pi))*crossepsy
    #print('analytical mux: ',3*crossepstermx/(tm**2 - t1*t2))
    #print('analytical muy: ',3*crossepstermy/(tm**2 - t1*t2))
    #print('analytical av (t-tm) eps: ',crossepstermx,crossepstermy)
    #print('magnitude: ',np.sqrt(crossepstermx**2 + crossepstermy**2))
    #crossepsdotterm=2*(crossepstermx*epsdotx + crossepstermy*epsdoty)
    #print('analytical epsdotterm: ',epsdotterm)
    #print('analytical crossepsdotterm: ',crossepsdotterm)

    '''if ((epsdotterm-crossepsdotterm)>0):
        print('term greater than 0')
        print('__epsdotterm: ',epsdotterm)
        print('__crossepsdotterm: ',crossepsdotterm)

    if (-(epsdotterm-crossepsdotterm)>av_epssq - aveps_sq):
        print('sqrt negative')
        print('__epsdotterm: ',epsdotterm)
        print('__crossepsdotterm: ',crossepsdotterm)'''

    dtheta = np.sqrt(av_epssq - aveps_sq - 3*(crossepsx**2 + crossepsy**2)/(tm**2 - t1*t2))
    ##print('_analytic dtheta2 full: ',av_epssq - aveps_sq -3*(crossepsx**2 + crossepsy**2)/(tm**2 - t1*t2))
    if return_pm == True:
        return dtheta, epsdotx, epsdoty
    else:
        return dtheta


def dtheta_wrong(ps, t1, t2, return_pm=False):
    # assuming ~uniform sampling in time between t1 and t2
    # and some known period
    if ps.Delta == -1:
        _ = Delta(ps)
    eta1 = findEtas(t1, ps.period, ps.e, tPeri=ps.tperi)
    eta2 = findEtas(t2, ps.period, ps.e, tPeri=ps.tperi)

    # using latest periapse time before t1
    tperi = ps.tperi+ps.period*np.floor((t1-ps.tperi)/ps.period)
    tm = (t1+t2)/2
    # findEtas always(?) returns values betwen 0 and 2*pi
    # for most uses this is what we want (eta mostly appears in trig.)
    # here however we don't want to lose fators of 2*pi
    ##print('\n__t1: ',t1,' t2: ',t2,' ps.tperi: ',ps.tperi,' tperi: ',tperi)
    ##print('eta1: ',eta1,' eta2: ',eta2)
    eta1 = eta1  # between 0 and 2 pi
    eta2 = eta2+2*np.pi*np.floor((t2-tperi)/ps.period)
    ##print('eta1: ',eta1,' eta2: ',eta2,'\n')

    sigma1, sigma2, sigma3, gamma1, gamma2, gamma3 = sigmagamma(eta1, eta2)
    sigmahat1, sigmahat2, gammahat1, gammahat2 = sigmagammahat(eta1, eta2)

    nu = 1/(1-ps.e*sigma1)
    Omega = np.sqrt(1-(np.cos(ps.vphi)**2) * (np.sin(ps.vtheta)**2))
    Kappa = np.sin(ps.vphi)*np.cos(ps.vphi)*(np.sin(ps.vtheta)**2)

    pre = ps.parallax*ps.Delta*ps.a/Omega

    #epsx1 = (1+ps.e**2)*sigma1 - ps.e*(1.5 + sigma2/4)
    epsxa = -(3*ps.e/2) + (1+ps.e**2)*sigma1 - (ps.e/4)*sigma2
    epsxb = gamma1-(ps.e/4)*gamma2
    epsx = nu*pre*(epsxa*Omega**2 + Kappa*np.sqrt(1-ps.e**2)*epsxb)

    epsy = -nu*pre*np.cos(ps.vtheta)*np.sqrt(1-ps.e**2)*(gamma1-(ps.e/4)*gamma2)

    ##print('analytical epsx: ',epsx,' and epsy: ',epsy)

    #epsxsq1 = (1+2*ps.e**2)*(0.5+sigma2/4)-ps.e*(2+ps.e**2)*sigma1
    #-ps.e*(3*sigma1/4 + sigma3/12)+ps.e**2
    epsxsqa = ((1+4*ps.e**2)/2)-((11+4*ps.e**2)/4)*ps.e*sigma1\
        + ((1+2*ps.e**2)/4)*sigma2 - (ps.e/12)*sigma3
    #epsxsq2 = (1+ps.e**2)*(gamma2/4)-ps.e*gamma1-ps.e*(gamma1/4 + gamma3/12)
    epsxsqb = (5*ps.e/4)*gamma1 - ((1+ps.e**2)/4)*gamma2 + (ps.e/12)*gamma3
    #epsxsq3 = 0.5-sigma2/4-ps.e*(sigma1/4 - sigma3/12)
    epsxsqc = (1/2) - (ps.e/4)*sigma1 - (1/4)*sigma2 + (ps.e/12)*sigma3
    epsxsq = nu*(pre**2)*((Omega**4)*epsxsqa
                          - 2*(Omega**2)*Kappa*np.sqrt(1-ps.e**2)*epsxsqb
                          + (Kappa**2)*(1-ps.e**2)*epsxsqc)

    epsysq = nu*(pre**2)*(np.cos(ps.vtheta)**2)*(1-ps.e**2) *\
        ((1/2) - (ps.e/4)*sigma1 - (1/4)*sigma2 + (ps.e/12)*sigma3)

    av_epssq = epsxsq+epsysq
    aveps_sq = epsx**2 + epsy**2

    ##print('analytical eps2: ',epsxsq+epsysq)
    ##print('analytical dtheta2 old: ',epsxsq+epsysq-(epsx**2 + epsy**2))

    eps1x = pre*((Omega**2)*(np.cos(eta1)-ps.e) - Kappa*np.sqrt(1-ps.e**2)*np.sin(eta1))
    eps2x = pre*((Omega**2)*(np.cos(eta2)-ps.e) - Kappa*np.sqrt(1-ps.e**2)*np.sin(eta2))
    # epscx=(eps1x+eps2x)/2

    eps1y = pre*np.cos(ps.vtheta)*np.sqrt(1-ps.e**2)*np.sin(eta1)
    eps2y = pre*np.cos(ps.vtheta)*np.sqrt(1-ps.e**2)*np.sin(eta2)

    epsdotx = (eps2x-eps1x)/(t2-t1)
    epsdoty = (eps2y-eps1y)/(t2-t1)

    epsdotterm = (epsdotx**2+epsdoty**2)*(tm**2 - t1*t2)/3

    ##print('analytical epsdot: ',epsdotx,epsdoty)
    ##print('analytical av epsc2: ',epsdotterm)

    crossepsxa = ((4-(ps.e**2))/4)*gamma1+((1+2*(ps.e**2))/8)*ps.e*gamma2-((ps.e**2)/12)*gamma3\
        + (1+(ps.e**2))*sigmahat1 - (ps.e/4)*sigmahat2 - (3*ps.e/4)*(eta1+eta2)
    crossepsxb = (ps.e/2) - ((4+ps.e**2)/4)*sigma1 - (ps.e/8)*sigma2\
        + (ps.e**2/12)*sigma3 + gammahat1 - (ps.e/4)*gammahat2

    crossepsx = nu*(pre)*((Omega**2)*crossepsxa + Kappa*np.sqrt(1-ps.e**2)*crossepsxb)
    crossepsy = -nu*pre*np.cos(ps.vtheta)*np.sqrt(1-ps.e**2)*((ps.e/2)
                                                              - ((4+ps.e**2)/4)*sigma1 - (ps.e/8) *
                                                              sigma2 + (ps.e**2/12)*sigma3
                                                              + gammahat1 - (ps.e/4)*gammahat2)
    ##print('analytical av eta eps: ',crossepsx,crossepsy)
    crossepstermx = (tperi-tm)*epsx + (ps.period/(2*np.pi))*crossepsx
    crossepstermy = (tperi-tm)*epsy + (ps.period/(2*np.pi))*crossepsy
    ##print('analytical mux: ',3*crossepstermx/(tm**2 - t1*t2))
    ##print('analytical muy: ',3*crossepstermy/(tm**2 - t1*t2))
    ##print('analytical av (t-tm) eps: ',crossepstermx,crossepstermy)
    ##print('magnitude: ',np.sqrt(crossepstermx**2 + crossepstermy**2))
    crossepsdotterm = 2*(crossepstermx*epsdotx + crossepstermy*epsdoty)
    ##print('analytical epsdotterm: ',epsdotterm)
    ##print('analytical crossepsdotterm: ',crossepsdotterm)

    '''if ((epsdotterm-crossepsdotterm)>0):
        print('term greater than 0')
        print('__epsdotterm: ',epsdotterm)
        print('__crossepsdotterm: ',crossepsdotterm)

    if (-(epsdotterm-crossepsdotterm)>av_epssq - aveps_sq):
        print('sqrt negative')
        print('__epsdotterm: ',epsdotterm)
        print('__crossepsdotterm: ',crossepsdotterm)'''

    dtheta = np.sqrt(av_epssq - aveps_sq + epsdotterm - crossepsdotterm)
    ##print('_analytic dtheta2 full: ',av_epssq - aveps_sq + epsdotterm - crossepsdotterm)
    if return_pm == True:
        return dtheta, epsdotx, epsdoty
    else:
        return dtheta


def dtheta_old(ps, t1, t2):
    # assuming ~uniform sampling in time between t1 and t2
    # and some known period
    if ps.Delta == -1:
        _ = Delta(ps)
    eta1 = findEtas(t1, ps.period, ps.e, tPeri=ps.tperi)
    eta2 = findEtas(t2, ps.period, ps.e, tPeri=ps.tperi)

    # using latest periapse time before t1
    tperi = ps.tperi+ps.period*np.floor((t1-ps.tperi)/ps.period)
    # findEtas always(?) returns values betwen 0 and 2*pi
    # for most uses this is what we want (eta mostly appears in trig.)
    # here however we don't want to lose fators of 2*pi
    eta1 = eta1  # between 0 and 2 pi
    eta2 = eta2+2*np.pi*np.floor((t2-tperi)/ps.period)

    sigma1, sigma2, sigma3, gamma1, gamma2, gamma3 = sigmagamma(eta1, eta2)
    sigmahat1, sigmahat2, gammahat1, gammahat2 = sigmagammahat(eta1, eta2)

    nu = 1/(1-ps.e*sigma1)
    Omega = np.sqrt(1-(np.cos(ps.vphi)**2) * (np.sin(ps.vtheta)**2))
    Kappa = np.sin(ps.vphi)*np.cos(ps.vphi)*(np.sin(ps.vtheta)**2)

    pre = ps.parallax*ps.Delta*ps.a/Omega

    #epsx1 = (1+ps.e**2)*sigma1 - ps.e*(1.5 + sigma2/4)
    epsxa = -(3*ps.e/2) + (1+ps.e**2)*sigma1 - (ps.e/4)*sigma2
    epsxb = gamma1-(ps.e/4)*gamma2
    epsx = nu*pre*(epsxa*Omega**2 + Kappa*np.sqrt(1-ps.e**2)*epsxb)

    epsy = -nu*pre*np.cos(ps.vtheta)*np.sqrt(1-ps.e**2)*(gamma1-(ps.e/4)*gamma2)

    #epsxsq1 = (1+2*ps.e**2)*(0.5+sigma2/4)-ps.e*(2+ps.e**2)*sigma1
    #-ps.e*(3*sigma1/4 + sigma3/12)+ps.e**2
    epsxsqa = ((1+4*ps.e**2)/2)-((11+4*ps.e**2)/4)*ps.e*sigma1\
        + ((1+2*ps.e**2)/4)*sigma2 - (ps.e/12)*sigma3
    #epsxsq2 = (1+ps.e**2)*(gamma2/4)-ps.e*gamma1-ps.e*(gamma1/4 + gamma3/12)
    epsxsqb = (5*ps.e/4)*gamma1 - ((1+ps.e**2)/4)*gamma2 + (ps.e/12)*gamma3
    #epsxsq3 = 0.5-sigma2/4-ps.e*(sigma1/4 - sigma3/12)
    epsxsqc = (1/2) - (ps.e/4)*sigma1 - (1/4)*sigma2 + (ps.e/12)*sigma3
    epsxsq = nu*(pre**2)*((Omega**4)*epsxsqa
                          - 2*(Omega**2)*Kappa*np.sqrt(1-ps.e**2)*epsxsqb
                          + (Kappa**2)*(1-ps.e**2)*epsxsqc)

    epsysq = nu*(pre**2)*(np.cos(ps.vtheta)**2)*(1-ps.e**2) *\
        ((1/2) - (ps.e/4)*sigma1 - (1/4)*sigma2 + (ps.e/12)*sigma3)

    av_epssq = epsxsq+epsysq
    aveps_sq = epsx**2 + epsy**2

    return np.sqrt(av_epssq-aveps_sq)


def radial_velocity(ts, ps, source='p'):
    # source: are we seeing the radial velocity of the photocenter (combined)
    # or of the primary ('p') or secondary ('s')?
    if ps.a == 0:
        return np.zeros(ts.size)
    if (source == 'c') or (source == 'combined'):
        Delta = (ps.l-ps.q)/((1+ps.q)*(1+ps.l))
    elif (source == 'p') or (source == 'primary'):
        Delta = -ps.q/(1+ps.q)
    elif (source == 's') or (source == 'secondary'):
        Delta = 1/(1+ps.q)
    etas = findEtas(ts, ps.period, ps.e, tPeri=ps.tperi)
    bracket = (np.cos(ps.vphi)*np.sin(etas) -
               np.sqrt(1-ps.e**2)*np.sin(ps.vphi)*np.cos(etas))/(1-ps.e*np.cos(etas))
    # expect a in AU and period in years, convert to ms-1
    unitconv = ((1.0*u.AU).to(u.m).value)/((1.0*u.yr).to(u.s).value)
    return unitconv*(2*np.pi*ps.a/ps.period)*Delta*np.sin(ps.vtheta)*bracket

def equilibrium_tide(ts,ps,R,beta=1):
    # quick approximate equilibrium tide computation for eclipsing binaries
    # taken from Penoyre & Stone 2019 eq 78
    # R is the stellar radius in AU
    # beta is a dimensionles factor (order unity) to capture unmodelled variations
    if ps.a == 0:
        return np.zeros(ts.size)
    etas = findEtas(ts, ps.period, ps.e, tPeri=ps.tperi)
    phis = np.arctan2(np.sqrt(1-ps.e**2)*np.sin(etas),np.cos(etas)-ps.e)
    rs = ps.a*(1-ps.e*np.cos(etas))
    viewterm = 3*(np.sin(ps.vtheta)**2)*(np.cos(ps.vphi-phis)**2)-1
    return -beta*ps.q*np.power(R/rs,3)*viewterm

def seperation(ts, ps, phis=None):
    if ps.a == 0:
        return np.zeros(ts.size)
    # seperation of two sources in a binary, using scan angle if supplied
    Omega = np.sqrt(1-(np.cos(ps.vphi)**2) * (np.sin(ps.vtheta)**2))
    Kappa = np.sin(ps.vphi)*np.cos(ps.vphi)*(np.sin(ps.vtheta)**2)

    pre = ps.parallax*ps.a/Omega

    etas = findEtas(ts, ps.period, ps.e, tPeri=ps.tperi)
    epsx = pre*((Omega**2)*(np.cos(etas)-ps.e) - Kappa*np.sqrt(1-ps.e**2)*np.sin(etas))
    epsy = pre*np.cos(ps.vtheta)*np.sqrt(1-ps.e**2)*np.sin(etas)

    dracs = epsx*np.cos(ps.vomega)-epsy*np.sin(ps.vomega)
    ddecs = epsy*np.cos(ps.vomega)+epsx*np.sin(ps.vomega)

    if phis is not None:
        return np.sqrt(dracs**2 + ddecs**2)
    else:
        return dracs*np.sin(phis) + ddecs*np.cos(phis)

# ----------------------
# -Utilities
# ----------------------


def viewing_angles(la, i, ap):
    """Converts conentional orbital elements to our coordinate system
    Args:
        la (radians): Longitude of ascending node
        i (radians): Inclination
        ap (radians): Periastron argument
    Returns:
        vtheta (radians): polar viewing angle
        vphi (radians): azimuthal viewing angle
        vomega (radians): rotation of the viewing plane
    """
    vtheta = np.pi - i
    vphi = np.pi/2 - la
    num = np.sin(la)*np.cos(ap) + np.cos(la)*np.cos(i)*np.sin(ap)
    denom = np.cos(la)*np.cos(ap) - np.sin(la)*np.cos(i)*np.sin(ap)
    vomega = np.arctan2(num, denom)
    return vtheta, vphi, vomega

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

# converts a phase (phi in radians) to a time in years
# rp should be in AU and Mtotal in Msun
# note that seems unreliable for phi=pi (but fine for infinitesimally below)
def phi_to_t(phi,rp,e,Mtotal,tPeri=0):
    cphi=np.cos(phi)
    sphi=np.sin(phi)
    if e==1:
        tau=np.sqrt((rp*(1+e))**3 /(Galt*Mtotal))
        return tau*sphi*(2+cphi)/(3*(1+cphi)**2)+tPeri
    if e<1:
        norbit=np.floor((phi+np.pi)/(2*np.pi))
        ceta=(cphi+e)/(1+e*cphi)
        seta=np.sqrt(1-e**2)*sphi/(1+e*cphi)
        eta=np.arctan2(seta,ceta)
        tau=np.sqrt((rp/(1-e))**3 /(Galt*Mtotal))
        return tau*(eta-e*seta+2*np.pi*norbit)+tPeri
    if e>1:
        chzeta=(cphi+e)/(1+e*cphi)
        shzeta=np.sqrt(e**2 - 1)*sphi/(1+e*cphi)
        zeta=np.arctanh(shzeta/chzeta)
        tau=np.sqrt(np.abs(rp/(1-e))**3 /(Galt*Mtotal))
        return tau*(e*shzeta-zeta)+tPeri

# inverse of above function
def t_to_phi(t,rp,e,Mtotal,tPeri=0):
    if e==1:
        tau=np.sqrt((rp*(1+e))**3 /(Galt*Mtotal))
        phi=findPhisParabolic(t,tau,tPeri=tPeri)
    if e<1:
        tau=2*np.pi*np.sqrt((rp/(1-e))**3 /(Galt*Mtotal))
        norbit=np.floor(0.5+(t-tPeri)/tau)
        eta=findEtas(t,tau,e,tPeri=tPeri)
        cphi=(np.cos(eta)-e)/(1-e*np.cos(eta))
        sphi=np.sqrt(1-e**2)*np.sin(eta)/(1-e*np.cos(eta))
        phi=np.arctan2(sphi,cphi)+2*np.pi*norbit
    if e>1:
        tau=2*np.pi*np.sqrt(np.abs(rp/(1-e))**3 /(Galt*Mtotal))
        zeta=findEtasHyperbolic(t,tau,e,tPeri=tPeri)
        cphi=(e-np.cosh(zeta))/(e*np.cosh(zeta)-1)
        sphi=np.sqrt(e**2-1)*np.sinh(zeta)/(e*np.cosh(zeta)-1)
        phi=np.arctan2(sphi,cphi)
    return phi

# ----------------------
# -Plots
# ----------------------
def plottrack(ts,params,ax=0,s=5,c=None,alpha=0.5,lw=0,ls='-',nts=1000):
    if ax==0:
        ax=plt.gca()
    if len(ts)==2: # if we just give two times treats these as start and end and generates nts times
        ts=np.linspace(ts[0],ts[1],nts)
    dracs,ddecs=track(ts, params)
    if lw==0:
        if c!=None:
            ax.scatter(dracs,ddecs,c=c,s=s,alpha=alpha)
        else:
            ax.scatter(dracs,ddecs,s=s,alpha=alpha)
    else:
        if c!=None:
            ax.plot(dracs,ddecs,c=c,alpha=alpha,lw=lw,ls=ls)
        else:
            ax.plot(dracs,ddecs,alpha=alpha,lw=lw,ls=ls)
    return ax

def plotresults(ts,results,error=False,refra=np.nan,refdec=np.nan,
        ax=0,s=5,c=None,alpha=0.5,lw=0,ls='-',nts=1000):
    from .fits import resultsparams
    rparams=resultsparams(results,error=error,refra=refra,refdec=refdec)
    ax=plottrack(ts,rparams,ax=ax,s=s,c=c,alpha=alpha,lw=lw,ls=ls,nts=nts)
    return ax
