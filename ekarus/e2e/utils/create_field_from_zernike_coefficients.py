from arte.utils.zernike_generator import ZernikeGenerator
import numpy as np

def create_field_from_zernike_coefficients(mask, zernike_mode, amplitude):
    """
    Create an electric field input corresponding to a Zernike aberration.
    
    :param mask: CircularMask object defining the pupil
    :param zernike_mode: Zernike mode to apply (1 for piston)
    :param amplitude: Amplitude of the Zernike aberration in radians
    :return: input electric field as a numpy complex array
    """
    zg = ZernikeGenerator(mask)
    phase_mask = amplitude * zg.getZernike(zernike_mode)  # Zernike mode 1 is piston
    return mask.asTransmissionValue() * np.exp(1j * phase_mask)


# TODO extend this function to support a vector of Zernike modes using arte.types.zernike_coefficients