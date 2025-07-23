import numpy as xp
# import cupy as xp

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

        :param apex_angle: pyramid vertex angle (in pixels)
        """
        self.apex_angle = apex_angle
        

    def pyramid_phase_delay(self, shape):
        """
        Computes the phase delay introduced by the pyramid wavefront sensor
        in the focal plane.

        :param shape: tuple (ny, nx) electric field dimensions
        :return: array numpy 2D float (phase delay in pixels)
        """
        ny, nx = shape
        cx, cy = nx // 2, ny // 2

        # Coordinates wrt center
        x = xp.arange(nx) - cx
        y = xp.arange(ny) - cy
        X, Y = xp.meshgrid(x, y)

        # Phase delay calculation
        D = xp.min((nx,ny))#//2
        phi = self.apex_angle*(1 - 1/D*(xp.abs(X)+xp.abs(Y)))

        return phi
    

    def propagate(self, input_field):
        """
        Propagate the electric field through the pyramid:
        1. From the pupil plane to the focal plane (FFT)
        2. Apply the pyramid phase delay (shift each point by the phase delay)
        3. From the focal plane to the output pupil plane (IFFT)

        :param input_field: complex array numpy 2D representing the input electric field
        :return: complex array numpy 2D representing the output electric field
        """
        # 1. Propagation from the pupil plane to the focal plane (Fourier transform)
        _ef_on_focal_plane = xp.fft.fft2(input_field)#, norm = 'ortho')
        self.field_on_focal_plane = xp.fft.fftshift(_ef_on_focal_plane)

        # 2. Calcola il ritardo di fase della piramide
        phase_delay = self.pyramid_phase_delay(self.field_on_focal_plane.shape)

        # 2b. Applica il ritardo di fase: ogni punto viene ritardato di phase_delay (fase)
        self._ef_focal_plane_delayed = self.field_on_focal_plane * xp.exp(1j*phase_delay)

        # 3. Inverse propagation from the focal plane to the output pupil plane (inverse transform)
        self._ef_focal_plane_delayed = xp.fft.ifftshift(self._ef_focal_plane_delayed)
        output_field = xp.fft.ifft2(self._ef_focal_plane_delayed)#, norm = 'ortho')

        return output_field
    

    def modulate(self, input_field, alpha_pix, N_steps=16):
        """
        Modulates the input electric field by tilting it in different directions
        and averaging the resulting intensities.

        :param input_field: complex array numpy 2D representing the input electric field
        :param alpha_pix: float, modulation amplitude in pixels

        :return: array numpy 2D representing the average intensity after modulation
        """
        ny, nx = xp.shape(input_field)
        cx, cy = nx // 2, ny // 2

        # Coordinates wrt center
        x = xp.arange(nx) - cx
        y = xp.arange(ny) - cy

        # Tilt coordinates
        tiltX,tiltY = xp.meshgrid(x, y)
        maxX = xp.max(xp.abs(tiltX))
        maxY = xp.max(xp.abs(tiltY))

        # Number of steps for modulation (multiple of 4 for symmetry between subapertures)
        phi_vec = 2*xp.pi*xp.arange(N_steps)/N_steps

        # Initialize intensity and perform modulation
        intensity = xp.zeros([ny,nx])

        for phi in phi_vec:
            tilt = tiltX * xp.cos(phi) + tiltY * xp.sin(phi)
            tilt *= 1/xp.sqrt((maxX*xp.cos(phi))**2 + (maxY*xp.sin(phi))**2)
            tilted_input = input_field * xp.exp(1j*tilt*alpha_pix)

            output = self.propagate(tilted_input)
            intensity += (xp.abs(output)**2)/N_steps

        return intensity