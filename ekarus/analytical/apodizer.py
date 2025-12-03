
import xupy as xp

def generate_app_keller(pupil, target_contrast, max_iterations:int, beta:float=0, oversampling:int=4):
    """
    Function taken from HCIpy (Por et al. 2018):
    https://github.com/ehpor/hcipy/blob/master/hcipy/coronagraphy/apodizing_phase_plate.py

    Accelerated Gerchberg-Saxton-like algorithm for APP design by
    Christoph Keller [Keller2016]_ and based on Douglas-Rachford operator splitting.
    The acceleration was inspired by the paper by Jim Fienup [Fienup1976]_. The
    acceleration can provide speed-ups of up to two orders of magnitude and
    produce better APPs.

    .. [Keller2016] Keller C.U., 2016, "Novel instrument concepts for
        characterizing directly imaged exoplanets", Proc. SPIE 9908,
        Ground-based and Airborne Instrumentation for Astronomy VI, 99089V
        doi: 10.1117/12.2232633; https://doi.org/10.1117/12.2232633

    .. [Fienup1976] J. R. Fienup, 1976, "Reconstruction of an object from the modulus
        of its Fourier transform," Opt. Lett. 3, 27-29

    Parameters
    ----------
    pupil : ndarray(bool)
        Boolean of the pupil aperture mask.
    target_contrast : ndarray(float)
        The required contrast in the focal plane: float mask that is 1.0
        everywhere except for the dark zone where it is the contrast value (e.g. 1e-5).
    max_iterations : int
        The maximum number of iterations.
    beta : float (optional)
        The acceleration parameter. The default is 0 (no acceleration).
        Good values for beta are typically between 0.3 and 0.9. Values larger
        than 1.0 will not work.
    oversampling : int (optional)
        The oversampling to use for the PSF computation.
        Should be greater than 3 to avoid issues, default is 4.

    Returns
    -------
    Wavefront
        The APP as a wavefront.

    Raises
    ------
    ValueError
        If beta is not between 0 and 1.
        If oversampling is less than 3.
    """
    if beta < 0 or beta > 1:
        raise ValueError('Beta should be between 0 and 1.')
    if oversampling < 3:
        raise ValueError('Oversampling should be at least 3 to avoid numerical issues.')

    # initialize APP with pupil
    app = pupil * xp.exp(1j*xp.zeros(pupil.shape),dtype=xp.complex64)

    # define dark zone as location where contrast is < 1e-1
    dark_zone = target_contrast < 0.1

    old_image = None
    for i in range(max_iterations):
        image = xp.fft.fftshift(xp.fft.fft2(app)) # calculate image plane electric field

        if not xp.any(xp.abs(image)**2 / xp.max(xp.abs(image)**2) > target_contrast):
            break

        new_image = image.copy()
        if beta != 0 and old_image is not None:
            new_image[dark_zone] = old_image[dark_zone] * beta - new_image[dark_zone] * (1 + beta)
        else:
            new_image[dark_zone] = 0
        old_image = new_image.copy()

        app = xp.fft.ifft2(xp.fft.ifftshift(new_image)) # determine pupil electric field
        app[~pupil.astype(bool)] = 0 # enforce pupil
        # app[pupil] /= xp.abs(app[pupil]) # enforce unity transmission within pupil
    
    if i == max_iterations-1:
        contrast = xp.abs(image)**2 / xp.max(xp.abs(image)**2)
        raise Warning(f'Maximum number of iterations ({max_iterations:1.0f}) reached, worst contrast in dark hole is: {xp.log10(xp.max(contrast[dark_zone])):1.1f}')

    return app