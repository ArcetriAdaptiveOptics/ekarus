import numpy as np

from ekarus.e2e.utils.image_utils import image_grid

class PyramidWFS:
    """
    Optical modeling of a pyramid wavefront sensor for adaptive optics.

    This class simulates the propagation of an electric field through a pyramid
    wavefront sensor, applying a phase mask that simulates a 4-faces pyramid,
    where the phase shift depends on the distance from the apex.
    """

    def __init__(self, apex_angle, xp=np):
        """
        Pyramid wavefront sensor constructor.

        :param apex_angle: pyramid vertex angle in radians
        """
        self.apex_angle = apex_angle
        self._xp = xp
        

    def pyramid_phase_delay(self, shape):
        """
        Computes the phase delay introduced by the pyramid wavefront sensor
        in the focal plane.

        :param shape: tuple (ny, nx) electric field dimensions
        :return: array numpy 2D float (phase delay in pixels)
        """
        X,Y = image_grid(shape, recenter=True)
        D = self._xp.max(shape)
        phi = 2*self._xp.pi*self.apex_angle*(1 - 1/D*(self._xp.abs(X)+self._xp.abs(Y)))

        return phi
    

    def propagate(self, input_field, pix2rad):
        """
        Propagate the electric field through the pyramid:
        1. From the pupil plane to the focal plane (FFT)
        2. Apply the pyramid phase delay (shift each point by the phase delay)
        3. From the focal plane to the output pupil plane (IFFT)

        :param input_field: complex array numpy 2D representing the input electric field
        :param pix2rad: float indicating the number of pixels per radian

        :return: complex array numpy 2D representing the output electric field
        """
        self.field_on_focal_plane = self._xp.fft.fftshift(self._xp.fft.fft2(input_field))

        phase_delay = self.pyramid_phase_delay(self.field_on_focal_plane.shape) / pix2rad
        self._ef_focal_plane_delayed = self.field_on_focal_plane * self._xp.exp(1j*phase_delay)

        output_field = self._xp.fft.ifft2(self._xp.fft.ifftshift(self._ef_focal_plane_delayed))

        return output_field
    

    def modulate(self, input_field, alpha, pix2rad):
        """
        Modulates the input electric field by tilting it in different directions
        and averaging the resulting intensities.

        :param input_field: complex array numpy 2D representing the input electric field
        :param alpha: float, modulation amplitude in radians
        :param pix2rad: float indicating the number of pixels per radian

        :return: array numpy 2D representing the average intensity after modulation
        """

        tiltX,tiltY = image_grid(input_field.shape, recenter=True)
        L = max(input_field.shape)

        alpha_pix = alpha/pix2rad*(2*self._xp.pi)
        N_steps = int((alpha_pix//20+1)*4)
        phi_vec = 2*self._xp.pi*self._xp.arange(N_steps)/N_steps

        intensity = self._xp.zeros(input_field.shape, dtype = self._xp.float32)

        for phi in phi_vec:
            tilt = (tiltX * self._xp.cos(phi) + tiltY * self._xp.sin(phi))/L
            tilted_input = input_field * self._xp.exp(1j*tilt*alpha_pix)

            output = self.propagate(tilted_input, pix2rad)
            intensity += (self._xp.abs(output**2))/N_steps

        return intensity