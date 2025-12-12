import matplotlib.pyplot as plt
import xupy as xp

from ekarus.e2e.utils.image_utils import reshape_on_mask, image_grid


def generate_app_keller(pupil, target_contrast, max_iterations:int, 
                        beta:float=0, oversampling:int=4, 
                        phi_guess=None, phi_residual=None):
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
    phi_guess : ndarray(float)
        Initial guess for the APP phase in [rad].
        Default is zero phase across the pupil.
    phi_residual : ndarray(float)
        Residual pupil phase in [rad].
        Default is zero (diffraction limited).

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

    if phi_guess is None:
        # initialize APP with pupil
        app = pupil * xp.exp(1j*xp.zeros(pupil.shape),dtype=xp.complex64)
    else:
        app = pupil * xp.exp(1j*phi_guess,dtype=xp.complex64)

    # define dark zone as location where contrast is < 1e-1
    dark_zone = target_contrast < 0.1

    old_image = None
    for i in range(max_iterations):
        # calculate image plane electric field
        if phi_residual is not None:
            image = xp.fft.fftshift(xp.fft.fft2(app * xp.exp(1j*phi_residual,dtype=xp.complex64))) 
        else:
            image = xp.fft.fftshift(xp.fft.fft2(app)) 

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
        # app[pupil.astype(bool)] /= xp.abs(app[pupil.astype(bool)]) # enforce unity transmission within pupil
        app = xp.asarray(pupil) * xp.exp(1j*xp.angle(app),dtype=xp.complex64)
    
    psf = xp.abs(image)**2
    contrast =  psf / xp.max(psf)
    ref_psf = xp.abs(xp.fft.fftshift(xp.fft.fft2(pupil)))**2

    flag = 0
    if i == max_iterations-1:
        print(f'Maximum number of iterations ({max_iterations:1.0f}) reached, worst contrast in dark hole is: {xp.log10(xp.max(contrast[dark_zone])):1.1f}')
        flag = 1
    else:
        print(f'Apodizer computed after {i:1.0f} iterations: average contrast in dark hole is {xp.mean(xp.log10(contrast[dark_zone])):1.1f}, Strehl is {xp.max(psf)/xp.max(ref_psf)*1e+2:1.2f}%')

    return app, flag


def define_apodizing_phase(pupil, contrast, iwa, owa, beta:float, 
                           oversampling:int, symmetric_dark_hole:bool=False, 
                           max_its:int=1000, show:bool=True, 
                           phi_guess_in_rad=None, phi_offset_in_rad=None):
    mask_shape = max(pupil.shape)
    if phi_guess_in_rad is not None:
        guess_phase = reshape_on_mask(phi_guess_in_rad,pupil)
        phi_guess = xp.pad(guess_phase.copy(), pad_width=int((mask_shape*(oversampling-1)//2)), mode='constant', constant_values=0.0)
    else:
        phi_guess = None
    if phi_offset_in_rad is not None:
        offset_phase = reshape_on_mask(phi_offset_in_rad,pupil)
        phi_residual = xp.pad(offset_phase.copy(), pad_width=int((mask_shape*(oversampling-1)//2)), mode='constant', constant_values=0.0)
    else:
        phi_residual = None
    padded_pupil = xp.pad(1-pupil.copy(), pad_width=int((mask_shape*(oversampling-1)//2)), mode='constant', constant_values=0.0)
    X,Y = image_grid(padded_pupil.shape,recenter=True)
    rho = xp.sqrt(X**2+Y**2)
    target_contrast = xp.ones_like(rho)
    if symmetric_dark_hole is True:
        where = (rho <= owa*oversampling) * (rho >= iwa*oversampling)
    else:
        where = (rho <= owa*oversampling) * (X >= iwa*oversampling) 
    target_contrast[where] = contrast
    app, max_its_flag = generate_app_keller(padded_pupil, target_contrast, max_iterations=max_its, beta=beta, 
                              phi_guess=phi_guess, phi_residual=phi_residual)
    phase = xp.angle(app)[padded_pupil>0.0]
    apodizer_phase = reshape_on_mask(phase, pupil)
    if show:
        focal_field = xp.fft.fftshift(xp.fft.fft2(app))
        app_psf = xp.abs(focal_field)**2
        app_psf /= xp.max(app_psf)
        sz = app_psf.shape
        plt.figure(figsize=(18,4))
        plt.subplot(1,3,1)
        plt.imshow(xp.asnumpy(xp.log10(target_contrast)),origin='lower',cmap='RdGy')
        plt.colorbar()
        plt.title('Target contrast')
        plt.subplot(1,3,2)
        plt.imshow(xp.asnumpy(apodizer_phase),origin='lower',cmap='RdBu')
        plt.colorbar()
        plt.title('APP phase')
        plt.subplot(1,3,3)
        plt.imshow(xp.asnumpy(xp.log10(app_psf)),cmap='inferno',vmin=xp.log10(xp.min(target_contrast)))
        plt.colorbar()
        pp = 20
        plt.xlim([sz[0]//2-pp*oversampling,sz[0]//2+pp*oversampling])
        plt.ylim([sz[1]//2-pp*oversampling,sz[1]//2+pp*oversampling])
        plt.title('Apodized PSF')
    return apodizer_phase, max_its_flag