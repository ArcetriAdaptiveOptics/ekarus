import numpy as np

from ekarus.e2e.utils.image_utils import image_grid
from functools import lru_cache

class PyramidWFS:
    """
    Optical modeling of a pyramid wavefront sensor for adaptive optics.

    This class simulates the propagation of an electric field through a pyramid
    wavefront sensor, applying a phase mask that simulates a 4-faces pyramid,
    where the phase shift depends on the distance from the apex.
    """

    def __init__(self, apex_angle, oversampling, xp=np):
        """
        Pyramid wavefront sensor constructor.

        :param apex_angle: pyramid vertex angle in radians
        """
        self.apex_angle = apex_angle
        self.oversampling = oversampling

        self.modulationAngleInLambdaOverD = None

        self._xp = xp
        self.dtype = xp.float32 if xp.__name__ == 'cupy' else xp.float64
        self.cdtype = xp.complex64 if xp.__name__ == 'cupy' else xp.complex128


    def get_intensity(self, input_field, lambdaOverD, wedgeShift=None):

        L = max(input_field.shape) # TBI: deal with non-square input fields
        padded_field = self._xp.pad(input_field, int((self.oversampling-1)/2*L), mode='constant', constant_values=0.0)

        if self.modulationAngleInLambdaOverD*(2*self._xp.pi)*self.oversampling < 0.1:
            output_field = self.propagate(padded_field, lambdaOverD)
            intensity = self._xp.abs(output_field**2)
        else:
            intensity = self.modulate(padded_field, lambdaOverD, wedgeShift)

        return intensity


    def set_modulation_angle(self, modulationAngleInLambdaOverD, verbose:bool=True):
        self.modulationAngleInLambdaOverD = modulationAngleInLambdaOverD
        self.modulationNsteps = self._xp.ceil(modulationAngleInLambdaOverD*2.4*self._xp.pi)//4*4
        if verbose:
            print(f'Now modulating {modulationAngleInLambdaOverD:1.0f} [lambda/D] with {self.modulationNsteps:1.0f} modulation steps')
        
        
    @lru_cache(maxsize=5)
    def pyramid_phase_delay(self, shape):
        """
        Computes the phase delay introduced by the pyramid wavefront sensor
        in the focal plane.

        :param shape: tuple (ny, nx) electric field dimensions
        :return: array numpy 2D float (phase delay in pixels)
        """
        X,Y = image_grid(shape, recenter=True, xp=self._xp)
        D = max(shape)
        phi = self.apex_angle*(1 - 1/D*(abs(X)+abs(Y)))
        phi = self._xp.asarray(phi,dtype=self.dtype)

        return phi
    

    def propagate(self, input_field, lambdaOverD):
        """
        Propagate the electric field through the pyramid:
        1. From the pupil plane to the focal plane (FFT)
        2. Apply the pyramid phase delay (shift each point by the phase delay)
        3. From the focal plane to the output pupil plane (IFFT)

        :param input_field: complex array numpy 2D representing the input
        electric field
        :param lambdaOverD: float, ratio of wavelength to pupil size

        :return: complex array numpy 2D representing the output electric field
        """
        self.field_on_focal_plane = self._xp.fft.fftshift(self._xp.fft.fft2(input_field))

        phase_delay = self.pyramid_phase_delay(input_field.shape) / lambdaOverD
        self._ef_focal_plane_delayed = self.field_on_focal_plane * self._xp.exp(1j*phase_delay, dtype = self.cdtype)

        output_field = self._xp.fft.ifft2(self._xp.fft.ifftshift(self._ef_focal_plane_delayed))

        return output_field
    

    def modulate(self, input_field, lambdaOverD, wedgeShift=None):
        """
        Modulates the input electric field by tilting it in different directions
        and averaging the resulting intensities.

        :param input_field: complex array numpy 2D representing the input
                            electric field
        :param lambdaOverD: float, ratio of wavelength to pupil size

        :return: numpy 2D array representing the average intensity after
                 modulation
        """
        
        tiltX,tiltY = self._get_XY_tilt_planes(input_field.shape)

        alpha_pix = self.modulationAngleInLambdaOverD*self.oversampling*(2*self._xp.pi)
        phi_vec = (2*self._xp.pi)*self._xp.arange(self.modulationNsteps)/self.modulationNsteps

        if wedgeShift is not None:
            wedgeX, wedgeY= wedgeShift
            wedge_tilt = (tiltX*wedgeX + tiltY*wedgeY)*self.oversampling*(2*self._xp.pi)
            input_field *= self._xp.exp(1j*wedge_tilt, dtype = self.cdtype)

        intensity = self._xp.zeros(input_field.shape, dtype = self.dtype)

        for phi in phi_vec:
            tilt = tiltX * self._xp.cos(phi) + tiltY * self._xp.sin(phi)
            tilted_input = input_field * self._xp.exp(1j*tilt*alpha_pix, dtype = self.cdtype)

            output = self.propagate(tilted_input, lambdaOverD)
            intensity += (abs(output**2))/self.modulationNsteps

        return intensity
    

    @lru_cache(maxsize=5)
    def _get_XY_tilt_planes(self, input_shape):
        tiltX,tiltY = image_grid(input_shape, recenter=True, xp=self._xp)
        L = max(input_shape)
        return tiltX/L,tiltY/L

