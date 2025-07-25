import numpy as xp
# import cupy as xp

from ekarus.e2e.utils.image_utils import image_grid

class PyramidWFS:
    """
    Optical modeling of a pyramid wavefront sensor for adaptive optics.

    This class simulates the propagation of an electric field through a pyramid
    wavefront sensor, applying a phase mask that simulates a 4-faces pyramid,
    where the phase shift depends on the distance from the apex.
    """

    def __init__(self, apex_angle):
        """
        Pyramid wavefront sensor constructor.

        :param apex_angle: pyramid vertex angle in radians
        """
        self.apex_angle = apex_angle
        

    def pyramid_phase_delay(self, shape):
        """
        Computes the phase delay introduced by the pyramid wavefront sensor
        in the focal plane.

        :param shape: tuple (ny, nx) electric field dimensions
        :return: array numpy 2D float (phase delay in pixels)
        """
        X,Y = image_grid(shape, recenter=True)
        D = xp.max(shape)
        phi = 2*xp.pi*self.apex_angle*(1 - 1/D*(xp.abs(X)+xp.abs(Y)))

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
        self.field_on_focal_plane = xp.fft.fftshift(xp.fft.fft2(input_field))

        phase_delay = self.pyramid_phase_delay(self.field_on_focal_plane.shape) / pix2rad
        self._ef_focal_plane_delayed = self.field_on_focal_plane * xp.exp(1j*phase_delay)

        output_field = xp.fft.ifft2(xp.fft.ifftshift(self._ef_focal_plane_delayed))

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
        L = xp.max(input_field.shape)

        alpha_pix = alpha/pix2rad
        N_steps = int((alpha_pix//4+1)*4)
        phi_vec = 2*xp.pi*xp.arange(N_steps)/N_steps

        intensity = xp.zeros(input_field.shape, dtype = float)

        for phi in phi_vec:
            tilt = (tiltX * xp.cos(phi) + tiltY * xp.sin(phi))/L
            tilted_input = input_field * xp.exp(1j*tilt*alpha_pix)

            output = self.propagate(tilted_input, pix2rad)
            intensity += (xp.abs(output**2))/N_steps

        return intensity