import numpy as np
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import get_body_barycentric

# All units SI
mSun=2e30
lSun=3.826e26
kpc=3e19
AU=1.496e+11
Gyr=3.15e16
day=24*(60**2)
year=365*day
G=6.67e-11 # in SI
e=0.0167
T=365.2422
year=T*24*(60**2)
T0=201.938
AU=1.496e+11
day=86400
c=299792458
AU_c=AU/(c*day)
#Galt=G/((kpc**3)/(mSun*(Gyr**2)))
#M0=1e11*mSun # mass of MW internal to sun (only used in point mass potential, whichModel=0)
#R0=15*kpc # approximate scale radius of MW NFW profile (from Piffl+2015)
#rho0=2e7*mSun/(kpc**3) # approximate density of MW NFW profile (from Piffl+2015)
mas2rad=4.84814e-9

#----------------
#-User functions
#----------------
def params

#----------------
#-On-sky motion
#----------------

# c.o.m motion in mas - all time in years, all angles mas except phi and theta (rad)
def comMotion(ts,phi,theta,dA,dD,mA,mD,p):
    taus=2*np.pi*(ts+T0/T)
    tau0=2*np.pi*T0/T
    psis=phi-taus
    psi0=phi-tau0
    dAs=dA+((ts-AU_c*np.cos(theta)*(np.cos(psis)-np.cos(psi0)
        +e*(np.sin(taus)*np.sin(psis) - np.sin(tau0)*np.sin(psi0))))*mA
        -(p/np.cos(theta))*(np.cos(psis)+e*(np.sin(taus)*np.sin(psis)-np.cos(phi))))
    dDs=dD+((ts-AU_c*np.cos(theta)*(np.cos(psis)-np.cos(psi0)
        +e*(np.sin(taus)*np.sin(psis) - np.sin(tau0)*np.sin(psi0))))*mD
        -p*np.sin(theta)*(np.sin(psis)+e*(np.sin(taus)*np.cos(psis)+np.sin(phi))))
    return dAs,dDs

# binary orbit
def findEtas(ts,M,a,e,t0=0): # finds an (approximate) eccentric anomaly (see Penoyre & Sandford 2019, appendix A)
    eta0s=np.sqrt(G*M/(a**3))*(ts-t0)
    eta1s=e*np.sin(eta0s)
    eta2s=(e**2)*np.sin(eta0s)*np.cos(eta0s)
    eta3s=(e**3)*np.sin(eta0s)*(1-(3/2)*np.sin(eta0s)**2)
    return eta0s+eta1s+eta2s+eta3s
def bodyPos(pxs,pys,l,q): # given the displacements transform to c.o.m. frame
    px1s=pxs*q/(1+q)
    px2s=-pxs/(1+q)
    py1s=pys*q/(1+q)
    py2s=-pys/(1+q)
    pxls=-pxs*(l-q)/((1+l)*(1+q))
    pyls=-pys*(l-q)/((1+l)*(1+q))
    return px1s,py1s,px2s,py2s,pxls,pyls
def binaryMotion(ts,M,q,l,a,e,vTheta,vPhi): # binary position (in projected AU)
    delta=np.abs(q-l)/((1+q)*(1+l))
    etas=findEtas(ts,M,a,e)
    phis=2*np.arctan(np.sqrt((1+e)/(1-e))*np.tan(etas/2)) % (2*np.pi)
    vPsis=vPhi-phis

    rs=a*(1-e*np.cos(etas))

    g=np.power(1-(np.cos(vPhi)**2)*(np.sin(vTheta)**2),-0.5)

    # projected positions in the c.o.m frame (in AU)
    pxs=rs*g*(np.cos(phis)-np.cos(vPsis)*np.cos(vPhi)*(np.sin(vTheta)**2))
    pys=rs*g*np.sin(phis)*np.cos(vTheta)

    # positions of sources 1 and 2 and the center of light
    px1s,py1s,px2s,py2s,pxls,pyls=bodyPos(pxs,pys,l,q)
    return px1s,py1s,px2s,py2s,pxls,pyls # x, y posn of each body and c.o.l.
    # in on-sky coords such that x is projected onto i dirn and y has no i component
# posn of earth
def get_R(mjd):
    R = get_body_barycentric('earth', Time(mjd, format='mjd', scale='tcb'),
                                  ephemeris="de430")
    return np.array([R.x.to(u.AU) / u.AU,
                     R.y.to(u.AU) / u.AU,
                     R.z.to(u.AU) / u.AU]).T


# combination of both motions, as offset from phi, theta position relative to ecliptic
def fullMotion(ts,M,q,l,P,e,vTheta,vPhi,twist,phi,theta,muAlpha,muDelta,pllx,error,returnBinary=False,t0=0,dA=0,dD=0): # get c.o.l. motion w. binary and error (can get just c.o.m. w. error by setting q=l)

    a=np.power((1+q)*G*M*((P*year)**2)/(2*np.pi)**2,1/3) # in m

    cAlpha,cDelta=comMotion(ts,phi,theta,dA,dD,muAlpha,muDelta,pllx) # c.o.m. motion

    px1s,py1s,px2s,py2s,pxls,pyls=binaryMotion((ts-t0)*year,M,q,l,a,e,vTheta,vPhi) # c.o.l. correction
    rls=pllx*((pxls/AU)*np.cos(twist)+(pyls/AU)*np.sin(twist))
    dls=pllx*((pyls/AU)*np.cos(twist)-(pxls/AU)*np.sin(twist))

    rands=np.random.normal(loc=0,scale=error,size=ts.size) # astrometric error
    dirs=2*np.pi*np.random.random(ts.size)
    drs=np.cos(dirs)*rands
    dds=np.sin(dirs)*rands
    if returnBinary==True: # also returns the projected position of both bodies in the binary
        fac1=q*(1+l)/(q-l)
        fac2=l*(1+q)/(l-q)
        return cAlpha+rls+drs, cDelta+dls+dds, cAlpha+fac1*rls+drs, cDelta+fac1*dls+dds, cAlpha+fac2*rls+drs, cDelta+fac2*dls+dds
    else:
        return cAlpha+rls+drs, cDelta+dls+dds

#----------------------
#-Fitting
#----------------------
def Xij(ts,phi,theta):
    N=ts.size
    taus=2*np.pi*(ts+T0/T)
    tau0=2*np.pi*T0/T
    psis=phi-taus
    psi0=phi-tau0
    xij=np.zeros((2*N,5))
    xij[:N,0]=1
    xij[N:,1]=1
    xij[:N,2]=ts-AU_c*np.cos(theta)*(np.cos(psis)-np.cos(psi0)
        +e*(np.sin(taus)*np.sin(psis) - np.sin(tau0)*np.sin(psi0)))
    xij[N:,3]=ts-AU_c*np.cos(theta)*(np.cos(psis)-np.cos(psi0)
        +e*(np.sin(taus)*np.sin(psis) - np.sin(tau0)*np.sin(psi0)))
    xij[:N,4]=-(1/np.cos(theta))*(np.cos(psis)+e*(np.sin(taus)*np.sin(psis)-np.cos(phi)))
    xij[N:,4]=-np.sin(theta)*(np.sin(psis)+e*(np.sin(taus)*np.cos(psis)+np.sin(phi)))
    return xij

#----------------------
#-Utilities
#----------------------

def sigString(number,significantFigures,extra=False): # returns a number to a given significant digits (if extra is true also returns base of first significant figure)
    roundingFactor=significantFigures - int(np.floor(np.log10(np.abs(number)))) - 1
    rounded=np.round(number, roundingFactor)
    # np.round retains a decimal point even if the number is an integer (i.e. we might expect 460 but instead get 460.0)
    if roundingFactor<=0:
        rounded=rounded.astype(int)
    string=rounded.astype(str)
    if extra==False:
        return string
    if extra==True:
        return string,roundingFactor

# generating, sampling and fitting a split normal (see https://authorea.com/users/107850/articles/371464-direct-parameter-finding-of-the-split-normal-distribution?commit=ad3d419474f75af951a55c40481506c5a3d1a5e4)
def splitNormal(x,mu,sigma,cigma):
    epsilon=cigma/sigma
    alphas=sigma*np.ones_like(x)
    alphas[x>mu]=cigma
    return (1/np.sqrt(2*np.pi*sigma**2))*(2/(1+epsilon))*np.exp(-0.5*((x-mu)/alphas)**2)

def splitInverse(F,mu,sigma,cigma): # takes a random number between 0 and 1 and returns draw from split normal
    epsilon=cigma/sigma
    #print(cigma)
    alphas=np.ones_like(F)
    alphas[F>1/(1+epsilon)]=cigma
    alphas[F<1/(1+epsilon)]=sigma
    betas=np.ones_like(F)
    betas[F>(1/(1+epsilon))]=1/epsilon
    return mu + np.sqrt(2*alphas**2)*scipy.special.erfinv(betas*((1+epsilon)*F - 1))

def splitFit(xs): # fits a split normal distribution to an array of data
    xs=np.sort(xs)
    N=xs.size
    Delta=int(N*stdErf) #hardcoded version of erf(1/sqrt(2))

    js=np.arange(1,N-Delta-2)
    w_js=xs[js+Delta]-xs[js]
    J=np.argmin(w_js)
    w_J=w_js[J]
    x_J=xs[J]

    ks=np.arange(J+1,J+Delta-2)
    theta_ks=(ks/N) - ((xs[ks]-x_J)/w_J)

    theta_kms=((ks-1)/N) - ((xs[ks-1]-x_J)/w_J)
    theta_kps=((ks+1)/N) - ((xs[ks+1]-x_J)/w_J)
    K=ks[np.argmin(np.abs(theta_ks-np.median(theta_ks)))]
    mu=xs[K]
    sigma=mu-x_J
    cigma=w_J-sigma

    beta=w_J/(xs[ks]-x_J)
    phi_ks=((ks-J)/Delta) - (stdErf*(beta-1)/beta)
    Z=ks[np.argmin(np.abs(phi_ks))]

    return xs[Z],sigma,cigma
