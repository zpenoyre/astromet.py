quickstart
==========

This is a simple example:
::

    import astromet
    import numpy as np

    # an object wich holds all the necessary parameters to generate the astrometric track
    params=astromet.params()

    # For this example we'll use a random binary system

    # center of mass parameters
    params.RA=160     #[deg]
    params.Dec=-50    #[deg]
    params.pmRAc=8    #[mas/yr]
    params.pmDec=-2   #[mas/yr]
    params.pllx=5     #[mas]

    # binary parameters
    # (for single stars leave these blank or set l=q)
    params.M=800      #[mSun]
    params.a=10       #[AU]
    params.e=0.8
    params.q=2e-5
    params.l=0.25
    # viewing angle
    params.vPhi=4.5   #[rad]
    params.vTheta=1.5 #[rad]
    params.vOmega=5.6 #[rad]
    # time of some periapse passage
    params.tPeri=2016 #[jyear]
