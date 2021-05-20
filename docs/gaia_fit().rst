gaia_fit()
==========

In the quickstart we've shown how to generate a 2D astrometric track (in RA cos(Dec) and Dec)
and to fit a single body motion to that data.

However, this issn't a perfect replica of the data Gaia records, nor how it is fitted.

In this section we'll bridge that gap to give as exact an analog as possible to the gaia results and pipeline.

scanning angles
---------------
The first major difference is that Gaia (or any similar telescope) doesn't record positions
in any given co-ordinate system - instead the precession of the telescope means that each observation
is along a particular axis - the scanning angle.

For bright sources Gaia measures positions both along (parallel) and across (perpendicular)
to the scan direction - with the former being a much more accurate measurement than the latter
(by a factor of about 5?). For dim sources (G>13) only along scan measurements are recorded.

Working with angles such that 0 degrees points towards Equatorial North and 90 degrees towards East
we can define a set of viewing angles, or better yet use the nominal Gaia scanning-law_ to find the actual
times and angles Gaia visited a patch of sky.

::

    import astromet
    import numpy as np
    import matplotlib.pyplot as plt
    import scanninglaw.times
    from scanninglaw.source import Source

    ra=160
    dec=-50
    c=Source(ra,dec,unit='deg')

    dr3_sl=scanninglaw.times.dr2_sl(version='dr3_nominal') # slow step - run only once
    sl=dr3_sl(c, return_times=True, return_angles=True)

    ts=np.squeeze(np.hstack(sl['times']))
    sort=np.argsort(ts)
    ts=2010+ts[sort]/365.25                         # [jyr]
    phis=np.squeeze(np.hstack(sl['angles']))[sort]  # [deg]

which we can have a look at
::

    qPl=plt.gca()
    qPl.scatter(ts,phis)
    qPl.set_xlabel(r'observation time')
    qPl.set_ylabel(r'scan angle')
    plt.show()

.. image:: plots/scanningLaw.png
  :width: 400
  :alt: example scanning law

of course if we want to skip this step it's not the end of the world to generate randomly
distributed ts and phis - but as we can see there is some structure here we'd miss out on.

Let's generate a fresh astrometric track (see the quickstart for more details)
::
    params=astromet.params()

    params.ra=ra
    params.dec=dec
    params.drac=0
    params.ddec=0
    params.pmrac=8
    params.pmdec=-2
    params.pllx=5

    params.period=2
    params.a=2
    params.e=0.8
    params.q=0.5
    params.l=0.1

    params.vphi=4.5
    params.vtheta=1.5
    params.vomega=5.6

    params.tperi=2016

    racs,decs=astromet.track(ts,params)

(this is the same system as the orange binary in quickstart)

Now we want to project the true positions (racs,decs) along our scanning angle and
add some random errors - let's assume we only have along scan measurements
(across scan barely contribute due to larger error anyway). If we know the magnitude
we can even use appropriate Gaia-like astrometric error!



.. _scanning-law: https://github.com/gaiaverse/scanninglaw




https://ui.adsabs.harvard.edu/abs/2012A%26A...538A..78L/abstract
