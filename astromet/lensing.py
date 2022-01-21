import numpy as np
import warnings

# ----------------------
# -Lensing
# ----------------------

def onsky_lens(dracs, ddecs, dracs_blend, ddecs_blend, thetaE, blendl):
    """
    Returns the lensed tracks as they are seen on the sky, using tracks of the light source and the lens. Takes into account the possibility of blending with the lens.

    Args:
        - dracs     ndarray - RAcosDec positions of the light source, mas
        - ddecs     ndarray - Dec positions of the light source, mas
        - dracs_blend    ndarray - RAcosDec positions of the light source, mas
        - ddecs_blend     ndarray - Dec positions of the light source, mas
        - thetaE    float - angular Einstein radius, mas
        - blendl       float - blend flux (units of source flux)
    Returns:
        - dracs_lensed      ndarray - RAcosDec positions on the lensed/blended track, mas
        - ddecs_lensed      ndarray - Dec positions on the lensed/blended track, mas
        - mag_diff   ndarray - difference in magnitude with respect to the baseline
    """

    # track & amplification of the light center
    dracs_img_rel, ddecs_img_rel, ampl = ulens(
        dracs - dracs_blend, ddecs - ddecs_blend, thetaE)
    dracs_img, ddecs_img = dracs_img_rel + dracs_blend, ddecs_img_rel + ddecs_blend
    mag_diff = -2.5*np.log10(ampl)

    # blending - if lens light is significant
    if(blendl > 0):
        light_ratio = blendl/ampl # blend light / source light (amplified)
        dracs_blended, ddecs_blended = blend(
            dracs_img, ddecs_img, dracs_blend, ddecs_blend, light_ratio)
        mag_diff = -2.5*np.log10(blendl+ampl)
        return dracs_blended, ddecs_blended, mag_diff
    else:
        return dracs_img, ddecs_img, mag_diff


def ulens(ddec, drac, thetaE):
    """
    Gets the light centre of lensing images - as simple as possible. Defined in the reference frame of the lens.
    Args:
        - ddec      ndarray - Dec relative positions of the light source, mas
        - drac      ndarray - RAcosDec relative positions of the light source, mas
        - thetaE    float - angular Einstein radius, mas
    Returns:
        - ddec_centroid     ndarray - Dec relative positions of the light centre of images, mas
        - drac_centroid     ndarray - RAcosDec relative positions of the light centre of images, mas
        - ampl      ndarray - amplification for each position (flux / flux at baseline)
    """

    x = ddec/thetaE
    y = drac/thetaE
    u = np.sqrt(x**2+y**2)

    ampl = (u**2 + 2)/(u*np.sqrt(u**2+4))

    th_plus = 0.5 * (u + (u**2 + 4)**0.5)
    th_minus = 0.5 * (u - (u**2 + 4)**0.5)

    A_plus = (u**2+2)/(2*u*(u**2+4)**0.5) + 0.5
    A_minus = A_plus - 1

    ddec_plus, drac_plus = th_plus * ddec / u, th_plus * drac / u
    ddec_minus, drac_minus = th_minus * ddec / u, th_minus * drac / u

    ddec_centroid, drac_centroid = (ddec_plus*A_plus + ddec_minus*A_minus) / \
        (A_plus + A_minus), (drac_plus*A_plus + drac_minus*A_minus)/(A_plus + A_minus)

    return ddec_centroid, drac_centroid, ampl


def get_offset(params, u0, t0):
    """
    Calculates the difference in positions of the lens and the source at the centred epoch, using the standard microlensing parameters u0, t0. Corrects params to include the difference.

    Args:
        - params        astrometric and lensing parameters
        - u0            float - impact parameter - lens-source separation at closest approach in units of the angular Einstein radius
        - t0            float - time of closest lens-source approach, decimalyear
    Returns:
        - params        params, corrected for the lens offset
    """

    mu_rel = np.array([params.pmrac - params.blendpmrac,
                      params.pmdec - params.blendpmdec])
    offset_t0 = mu_rel*(t0-params.epoch)
    offset_u0_dir = [mu_rel[1], -mu_rel[0]]
    offset_u0 = offset_u0_dir/np.linalg.norm(offset_u0_dir) * \
        u0*params.thetaE  # separation at t0
    offset_mas = offset_t0 - offset_u0
    params.blenddrac, params.blendddec = offset_mas[0], offset_mas[1]
    return params

def define_lens(u0, t0, tE, piEN, piEE, m0, fbl, pmrac_source, pmdec_source, d_source, thetaE):
    """
    Defines astromet parameters using standard microlensing parameters (u0, t0, tE, piEN, piEE, m0, fbl), kinematics of the source and thetaE.

    Args:
        (returned from a photometric model)
        - u0            float - impact parameter - lens-source separation at closest approach in units of the angular Einstein radius
        - t0            float - time of closest lens-source approach, reduced JD
        - tE            float - timescale of the event, days
        - piEN          float - local north component of the microlensing parallax vector (..)
        - piEE          float - local east component of the microlensing parallax vector
        - m0            float - magnitude at baseline, mag
        - fbl           float - blending parameter (flux from the source / combined flux)
        (assumed)
        - pmrac_source          float - proper motion of the source in RAcosDec, mas/yr
        - pmdec_source          float - proper motion of the source in DEC, mas/yr
        - d_source            float - distance to the source, kpc
        - thetaE        float - angular Einstein radius, mas
    Returns:
        - params        astromet parameters
    """

    # conversion to years
    y = (1.0*u.year).to(u.day).value
    t0 = Time(t0+2450000, format='jd').decimalyear
    tE = tE/y

    # relative motion
    piE = np.sqrt(piEN**2 + piEE**2)
    pi_rel = piE*thetaE

    mu_rel = thetaE/tE
    mu_rel_N = mu_rel*piEN/piE
    mu_rel_E = mu_rel*piEE/piE

    # lens motion
    pmdec_lens = mu_rel_N + pmdec_source
    pmrac_lens = mu_rel_E + pmrac_source

    # parallaxes
    pi_source = 1/d_source
    pi_lens = pi_source + pi_rel

    # astromet parameters
    params=astromet.params()

    params.ra=ra_event
    params.dec=dec_event

    # source motion
    params.parallax=pi_source
    params.pmrac=pmrac_source
    params.pmdec=pmdec_source

    # lens motion
    params.blendparallax=pi_lens
    params.blendpmrac=pmrac_lens
    params.blendpmdec=pmdec_lens

    # lensing event
    params.thetaE=thetaE
    params.blendl=(1 - fbl)/fbl

    # correct params to include source-lens offset
    params = astromet.get_offset(params, u0, t0)

    return params

# ----------------------
# -Blending
# ----------------------

def blend(drac_firstlight, ddec_firstlight, drac_blend, ddec_blend, lr):
    """
    Returns a blended position of two light sources as a simple weighted average.
    (Not applicable to Gaia data for separations beyond 200 mas.)

    Args:
        - drac_firstlight     ndarray - RAcosDec positions of the primary light source, mas
        - ddec_firstlight     ndarray - Dec positions of the primary light source, mas
        - drac_blend    ndarray - RAcosDec positions of the blend, mas
        - ddec_blend    ndarray - Dec positions of the blend, mas
        - lr            ndarray or float - light ratio (flux from the primary source / combined flux)
    Returns:
        - drac_blended      ndarray - RAcosDec positions after blending, mas
        - ddec_blended      ndarray - Dec positions after blending, mas
    """

    if np.max(np.sqrt((drac_firstlight-drac_blend)**2 + (ddec_firstlight-ddec_blend)**2)) > 200:
        warnings.warn("You are using separations > 200 mas in the blending function - those sources will not be blended in Gaia data!")

    drac_blended, ddec_blended = drac_firstlight/(1+lr) + drac_blend * lr/(1+lr), ddec_firstlight/(1+lr) + ddec_blend * lr/(1+lr)
    return drac_blended, ddec_blended
