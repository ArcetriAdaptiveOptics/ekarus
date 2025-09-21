import numpy as np
from ekarus.e2e.utils.create_field_from_zernike_coefficients import create_field_from_zernike_coefficients
from arte.types.mask import CircularMask

def create_psf_of_a_zernike_aberration(zernike_mode, amplitude):
    """
    Create a PSF of a Zernike aberration 
    :param zernike_mode: Zernike mode to apply (1 for piston)
    :param amplitude: Amplitude of the Zernike aberration in radians
    :return: input_field:
    """
    nx = 128
    
    # Create pupil mask
    oversampling = 4 # Oversampling factor, try 1, 2, 4
    mask = CircularMask((oversampling * nx, oversampling * nx), maskRadius=nx // 2)

    # Create the input electric field 
    input_field = create_field_from_zernike_coefficients(mask, zernike_mode, amplitude)

    
    # Propagate the field from the pupil plane to the focal plane
    # use fft2 and fftshift
    output_field =  TODO 

    # Visualize the results
    import matplotlib.pyplot as plt 
    plt.figure(1, figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Input Electric Field (Phase)")
    plt.imshow(np.angle(input_field), cmap='twilight')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.title("Output Electric Field (Intensity) - Log Scale")
    plt.imshow(np.log(np.abs(output_field**2)), cmap='inferno')
    plt.colorbar()
    plt.show()

    return input_field, output_field
