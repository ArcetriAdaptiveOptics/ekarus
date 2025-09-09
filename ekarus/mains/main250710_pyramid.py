
import numpy as np
import matplotlib.pyplot as plt
from ekarus.e2e.pyramid_wfs import PyramidWFS
from arte.types.mask import CircularMask
from ekarus.e2e.utils.zernike_coefficients import create_field_from_zernike_coefficients


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
    apex_angle = 0.005859393655500168*30/32/(2*np.pi)  # vertex angle in radians, can be tricky to find the right value

    # Create pupil mask
    mask = CircularMask((oversampling * nx, oversampling * nx), maskRadius=nx // 2)

    lambdaInM = 1000e-9
    pupilSizeInM = 32e-3
    pix2rad = lambdaInM/pupilSizeInM/oversampling
    rad2arcsec = 180/np.pi*3600
    pix2arcsec = pix2rad*rad2arcsec

    # Create the input electric field for flat wavefront (a piston of 1 radians)
    input_field = create_field_from_zernike_coefficients(mask, 1, 1)

    wfs = PyramidWFS(apex_angle)
    output_field = wfs.propagate(input_field, pix2rad)

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
    ny, nx = np.shape(input_field)
    cx, cy = nx // 2, ny // 2

    # Coordinates wrt center
    x = np.arange(nx) - cx
    y = np.arange(ny) - cy

    # Tilt coordinates
    X,Y = np.meshgrid(x, y)
    D = np.max((nx,ny))

    # Initialize intensity and perform modulation
    intensity = np.zeros([ny,nx])
    pow_psf = np.zeros([ny,nx])

    xTilt = -50/rad2arcsec/pix2rad
    yTilt = -60/rad2arcsec/pix2rad
    tilt = X/D * xTilt + Y/D * yTilt
    tilted_input = input_field * np.exp(1j*tilt*(2*np.pi))

    tilt_ef = np.fft.fftshift(np.fft.fft2(tilted_input))
    tilt_psf = np.abs(tilt_ef**2)

    plt.figure()
    showZoomCenter(tilt_psf, pix2arcsec, title=r'Tilted PSF (xTilt = %g arcsec, yTilt = %g arcsec)' % (xTilt*rad2arcsec*pix2rad, yTilt*rad2arcsec*pix2rad),
                xlabel=r'$\theta_{sky}$ [arcsec]', ylabel=r'$\phi_{sky}$ [arcsec]',
                zlabel=r'log PSF($\theta_{sky}$, $\phi_{sky}$)')


    # Number of steps for modulation (multiple of 4 for symmetry between subapertures)
    modulationInLambdaOverD = 6
    N_steps = int(np.ceil(modulationInLambdaOverD*2.4*np.pi)//4*4)
    phi_vec = np.linspace(0,2*np.pi,N_steps,endpoint=False)
    alpha = lambdaInM/pupilSizeInM*modulationInLambdaOverD 
    alpha_pix = alpha/pix2rad*(2*np.pi)

    for phi in phi_vec:
        tilt = (X * np.cos(phi) + Y * np.sin(phi))/D*alpha_pix
        tilted_input = input_field * np.exp(1j*tilt)

        psf = np.fft.fftshift(np.fft.fft2(tilted_input))
        pow_psf += np.abs(psf)**2

        output = wfs.propagate(tilted_input, pix2rad)
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
