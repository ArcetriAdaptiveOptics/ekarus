
import numpy as np
import matplotlib.pyplot as plt
from ekarus.e2e.pyramid_wfs import PyramidWFS
from arte.types.mask import CircularMask
# from ekarus.e2e.utils.zernike_coefficients import create_field_from_zernike_coefficients
from ekarus.e2e.utils.image_utils import image_grid

from arte.utils.zernike_generator import ZernikeGenerator

def create_field_from_zernike_coefficients(mask, noll_ids:tuple, amplitudes:tuple):
    """
    Create an electric field input corresponding to a Zernike aberration.
    
    :param mask: CircularMask object defining the pupil
    :param noll_ids: tuple of Zernike noll number
    :param amplitudes: Amplitude or tuple of amplitudes
                       of the Zernike aberration in radians
    
    :return: input electric field as a numpy complex array
    """
    phase_mask = project_zernike_on_mask(mask, noll_ids, amplitudes)
    return mask.asTransmissionValue() * np.exp(1j * phase_mask)

def project_zernike_on_mask(mask, noll_ids:tuple, amplitudes:tuple):
    """
    Create a linear combination of Zernikes on a mask.
    
    :param mask: CircularMask object defining the pupil
    :param noll_ids: tuple of Zernike noll number
    :param amplitudes: Amplitude or tuple of amplitudes
                       of the Zernike aberration in radians
    
    :return: zernike combination
    """
    zg = ZernikeGenerator(mask)

    if isinstance(noll_ids,int):
        amp = amplitudes
        noll = noll_ids
        zern = amp * zg.getZernike(noll)
    else:
        amplitudes *= np.ones_like(noll_ids)
        zern = np.zeros(mask.mask().shape)
        for amp,noll in zip(amplitudes, noll_ids):
            zern += amp * zg.getZernike(noll)

    return zern


def imageShow(image2d, pixelSize=1, title='', xlabel='', ylabel='', zlabel='', shrink=1.0):
    sz=image2d.shape
    plt.imshow(image2d, extent=[-sz[0]/2*pixelSize, sz[0]/2*pixelSize,
                                -sz[1]/2*pixelSize, sz[1]/2*pixelSize],origin='lower')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    cbar= plt.colorbar(shrink=shrink)
    cbar.ax.set_ylabel(zlabel)

def showZoomCenter(image, pixelSize, **kwargs):
    '''show log(image) zoomed around center'''
    imageHalfSizeInPoints= image.shape[0]/2
    roi= [int(imageHalfSizeInPoints*0.8), int(imageHalfSizeInPoints*1.2)]
    imageZoomedLog= np.log(image[roi[0]: roi[1], roi[0]:roi[1]])
    imageShow(imageZoomedLog, pixelSize=pixelSize, **kwargs)


def main():
    nx = 128
    oversampling = 4

    # Create pupil mask
    mask = CircularMask((nx,nx), maskRadius=nx // 2)

    lambdaInM = 1000e-9
    pupilSizeInM = 32e-3

    lambdaOverD = lambdaInM/pupilSizeInM

    pix2rad = lambdaOverD/oversampling
    rad2arcsec = 180/np.pi*3600
    pix2arcsec = pix2rad*rad2arcsec

    # Create the input electric field for flat wavefront (a piston of 1 radians)
    input_field = create_field_from_zernike_coefficients(mask, 1, 1)

    apex_angle = 112*(2*np.pi)*lambdaOverD
    wfs = PyramidWFS(apex_angle, oversampling)
    padded_field = np.pad(input_field, int((oversampling-1)/2*nx), mode='constant', constant_values=0.0)
    output_field = wfs.propagate(padded_field, lambdaOverD)

    psf = np.abs(wfs.field_on_focal_plane*np.conj(wfs.field_on_focal_plane))

    plt.figure(1, figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.title("Input Electric Field (Phase)")
    plt.imshow(np.angle(input_field), cmap='twilight')
    plt.colorbar(shrink=0.6)

    plt.subplot(1, 3, 2)
    showZoomCenter(psf, pix2arcsec, title=r'Zoomed PSF (D=%g m, $\lambda$=%g $\mu m$)' % (pupilSizeInM, lambdaInM*1e6),
                xlabel=r'$\theta_{sky}$ [arcsec]', ylabel=r'$\phi_{sky}$ [arcsec]',
                zlabel=r'log PSF($\theta_{sky}$, $\phi_{sky}$)', shrink=0.6)

    plt.subplot(1, 3, 3)
    plt.title("Output Electric Field (Intensity)")
    plt.imshow(np.abs(output_field**2))
    plt.colorbar(shrink=0.6)

    plt.figure(2)
    plt.imshow(wfs.pyramid_phase_delay((nx, nx)), cmap='twilight')
    plt.title("Pyramid phase mask")
    plt.colorbar()

        #return wfs, input_field, output_field

    # Test modulation
    tiltX,tiltY = image_grid(padded_field.shape, recenter=True)
    L = max(padded_field.shape)

    # Initialize intensity and perform modulation
    intensity = np.zeros([nx*oversampling,nx*oversampling])
    pow_psf = np.zeros([nx*oversampling,nx*oversampling])

    xTilt = -50/rad2arcsec/lambdaOverD*(2*np.pi)*oversampling#-50/rad2arcsecpix2rad
    yTilt = -60/rad2arcsec/lambdaOverD*(2*np.pi)*oversampling #-60/rad2arcsec/pix2rad
    # tilt = X/D * xTilt + Y/D * yTilt
    tilt = tiltX/L * xTilt + tiltY/L * yTilt
    tilted_input = padded_field * np.exp(1j*tilt, dtype=complex)

    tilt_ef = np.fft.fftshift(np.fft.fft2(tilted_input))
    tilt_psf = np.abs(tilt_ef**2)

    plt.figure()
    showZoomCenter(tilt_psf, pix2arcsec, title=r'Tilted PSF (xTilt = %g arcsec, yTilt = %g arcsec)' % (xTilt*rad2arcsec*lambdaOverD/(2*np.pi*oversampling),
                 yTilt*rad2arcsec*lambdaOverD/(2*np.pi*oversampling)),
                xlabel=r'$\theta_{sky}$ [arcsec]', ylabel=r'$\phi_{sky}$ [arcsec]',
                zlabel=r'log PSF($\theta_{sky}$, $\phi_{sky}$)')


    # Number of steps for modulation (multiple of 4 for symmetry between subapertures)
    modulationInLambdaOverD = 6
    N_steps = int(np.ceil(modulationInLambdaOverD*2.25*np.pi)//4*4)
    phi_vec = np.linspace(0,2*np.pi,N_steps,endpoint=False)
    # alpha = lambdaOverD*modulationInLambdaOverD 
    alpha_pix = modulationInLambdaOverD*oversampling*(2*np.pi) #alpha/pix2rad*(2*np.pi)

    padded_field = np.pad(input_field, int((oversampling-1)/2*nx), mode='constant', constant_values=0.0)
    for phi in phi_vec:
        tilt = (tiltX * np.cos(phi) + tiltY * np.sin(phi))/L*alpha_pix #(X * np.cos(phi) + Y * np.sin(phi))/D*alpha_pix
        tilted_input = padded_field * np.exp(1j*tilt, dtype=complex)

        psf = np.fft.fftshift(np.fft.fft2(tilted_input))
        pow_psf += np.abs(psf)**2

        output = wfs.propagate(tilted_input, lambdaOverD)
        intensity += (np.abs(output)**2)/N_steps


    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    showZoomCenter(pow_psf, pix2arcsec, title=f'Modulated ({modulationInLambdaOverD:1.0f}) PSF',
                xlabel=r'$\theta_{sky}$ [arcsec]', ylabel=r'$\phi_{sky}$ [arcsec]',
                zlabel=r'log PSF($\theta_{sky}$, $\phi_{sky}$)', shrink=0.6)

    plt.subplot(1,2,2)
    plt.title("Modulated Electric Field (Intensity)")
    plt.imshow(intensity)
    plt.colorbar(shrink=0.6)

    plt.show()
