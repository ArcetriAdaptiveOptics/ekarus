from arte.utils.zernike_generator import ZernikeGenerator
# from arte.types.zernike_coefficients import ZernikeCoefficients
import numpy as np


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


# def create_field_from_zernike_coefficients(mask, zernike_mode, amplitude):
#     """
#     Create an electric field input corresponding to a Zernike aberration.
    
#     :param mask: CircularMask object defining the pupil
#     :param zernike_mode: Zernike noll number
#     :param amplitude: Amplitude of the Zernike aberration in radians
    
#     :return: input electric field as a numpy complex array
#     """
#     zg = ZernikeGenerator(mask)
#     phase_mask = amplitude * zg.getZernike(zernike_mode)  # Zernike mode 1 is piston
#     return mask.asTransmissionValue() * np.exp(1j * phase_mask)
