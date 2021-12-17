fit()
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

    dr3_sl=scanninglaw.times.Times(version='dr3_nominal') # slow step - run only once

    ra=160
    dec=-50
    c=Source(ra,dec,unit='deg')
    sl=dr3_sl(c, return_times=True, return_angles=True)

    ts=np.squeeze(np.hstack(sl['times']))
    sort=np.argsort(ts)
    ts=2010+ts[sort]/365.25                         # [jyr]
    phis=np.squeeze(np.hstack(sl['angles']))[sort]  # [deg]

which we can have a look at
::

    ax=plt.gca()
    ax.scatter(ts,phis)
    ax.set_xlabel(r'observation time')
    ax.set_ylabel(r'scan angle')
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
    params.parallax=5

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

::

    mag=18
    al_error=astromet.sigma_ast(mag) # about 1.1 mas at this magnitude
    errs=al_error*np.random.randn(phis.size)

    t_obs,x_obs,phi_obs,rac_obs,dec_obs=astromet.mock_obs(ts,phis,racs,decs,err=x_err)
    radphis=np.deg2rad(phi_obs)

    plotts=np.linspace(np.min(t_obs),np.max(t_obs),1000)
    plotracs,plotdecs=astromet.track(plotts,params)

    ax=plt.gca()
    for i in range(t_obs.size):
        ax.plot([rac_obs-al_error*np.sin(radphis),rac_obs+al_error*np.sin(radphis)],
                [dec_obs-al_error*np.cos(radphis),dec_obs+al_error*np.cos(radphis)],c='b')
    ax.plot(plotracs,plotdecs,c='k')
    ax.set_xlabel(r'$RA \cos(Dec)$ [mas]')
    ax.set_ylabel(r'$Dec$ [mas]')
    plt.show()

which gives the true c.o.l. track in black, and the 1D observations in orange.


.. image:: plots/twoBodyScans.png
  :width: 400
  :alt: two body orbit scanned at particular angles

There's quite a lot going on in mock_obs() so let's examine the outputs a little
more closely - to replicate gaia it creates 9 observations for each observation period
(corresponding to Gaia's 9 rows of CCDs), generates a random error for each and applies
this to the rac and dec measurements, then projects the whole thing along the scan angles
to give the xs.

If we don't want 9 scans we can use the optional argument nmeasure. For example,
setting nmeasure=1 will just apply random errors to the positions we've already generated
and project along scan directions.

Let's look at the projected positions over time
::
    ax=plt.gca()
    ax.errorbar(t_obs,x_obs,yerr=al_error,fmt='x')
    ax.set_xlabel(r'observation time')
    ax.set_ylabel(r'$x_i = \alpha^*_i\ \sin(\phi) + \delta_i\ \cos(\phi)$')
    plt.show()

this isn't the most illuminating plot, but this is the space Gaia actually fits in:

.. image:: plots/scanXs.png
  :width: 400
  :alt: projected distance vs time

fitting
-------

We've done all the hard work so now let's actually fit the system
::

    bresults=astromet.fit(t_obs,x_obs,phi_obs,al_error,ra,dec)

this will give a similar set of results to simple_fit() from the quickstart,
but using a close emulation of the full Gaia astrometric pipeline
'AGIS <https://ui.adsabs.harvard.edu/abs/2012A%26A...538A..78L/abstract>'_.

In short this pipeline iteratively performs fits, inflating (if needed) an extra
error term (the 'excess_noise') until the residuals between the observations and best
fitting single-body model are consistent with this enlarged error.

Finally we might want an *exact* analog to the Gaia results, so we can transform
the output from fit() into the specific astrometric fields in the Gaia data model
using
::

    gaia_results=astromet.gaia_results(bresults)

.. _scanning-law: https://github.com/gaiaverse/scanninglaw
