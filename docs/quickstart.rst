quickstart
==========
Single body
-----------
Let's start with a simple single body example:
::

    import astromet
    import numpy as np

    # an object wich holds all the necessary parameters to generate the astrometric track
    params=astromet.params()

    # center of mass parameters
    params.ra=160     #[deg]
    params.dec=-50    #[deg]
    params.drac=0     #[mas]
    params.ddec=0     #[mas]
    params.pmrac=8    #[mas/yr]
    params.pmdec=-2   #[mas/yr]
    params.pllx=5     #[mas]

You may notice that we've defined 7 parameters, even though for a single star we
expect a 5 parameter solution. The reason for this is twofold:
- position is measured as a local deviation in milli-arcseconds [mas], instead of globally in degrees
- rather than using RA as a coordinate we can use RA*cos(Dec) locally (often shortened in the code to rac)
We need to define the RA and Dec approximately, as this sets the orientation of
the parallax ellipse, but then all local motion is described in (rac,dec), with
drac and ddec keeping track of the offset between our approximate RA and Dec and the actual position of the star.

All we need now is some times at which we want to find the position
::

    # random times between 2014 and 2018 - a very rough approximation to Gaia eDR3
    ts=2014 + 4*np.random.rand(100)

and we can find the track
::

    # finds the exact position in RAcos(Dec) and Dec [mas]
    racs,decs=astromet.track(ts,params)

and plot it
::

    ax=plt.gca()
    ax.scatter(racs,decs)
    ax.set_xlabel(r'$RA \cos(Dec)$ [mas]')
    ax.set_ylabel(r'$Dec$ [mas]')
    plt.show()

.. image:: plots/singleBody.pdf
  :width: 400
  :alt: single body astrometric track

Binary system
-------------

A binary system behaves in exactly the same way, just with a few extra parameters
to fully define the motion of the centre of light around the centre of masses

We can use the same params object as before and just adjust the other parameters

::

    # binary parameters
    # (for single stars leave these blank or set l=q)
    params.period=2      #[yr]
    params.a=2    #[AU]
    params.e=0.8
    params.q=0.5
    params.l=0.1
    # viewing angle
    params.vphi=4.5   #[rad]
    params.vtheta=1.5 #[rad]
    params.vomega=5.6 #[rad]
    # time of some periapse passage
    params.tperi=2016 #[jyear]

we've actually been working with a binary system the whole time, but the default is to set
q=l(=0), meaning the centre of light sits atop the centre of mass always and we don't see
any excess motion.

Let's compare the one-body motion with this binary system

::

    bracs,bdecs=astromet.track(ts,params)

    ax=plt.gca()
    ax.scatter(racs,decs)
    ax.scatter(bracs,bdecs)
    ax.set_xlabel(r'$RA \cos(Dec)$ [mas]')
    ax.set_ylabel(r'$Dec$ [mas]')
    plt.show()

giving
.. image:: plots/twoBody.pdf
  :width: 400
  :alt: binary astrometric track
