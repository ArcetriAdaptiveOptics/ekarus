import xupy as xp
# import numpy as np

from ekarus.e2e.utils.image_utils import image_grid
# from functools import lru_cache

class Pyr3WFS:
    """
    Optical modeling of a pyramid wavefront sensor for adaptive optics.

    This class simulates the propagation of an electric field through a pyramid
    wavefront sensor, applying a phase mask that simulates a 4-faces pyramid,
    where the phase shift depends on the distance from the apex.
    """

    def __init__(self, vertex_angle, oversampling, sensorLambda, sensorBandwidth=0.0):
        """
        Pyramid wavefront sensor constructor.

        :param vertex3_angle: pyramid vertex angle in radians
        """
        self.vertex3_angle = vertex_angle
        self.oversampling = oversampling
        self.lambdaInM = sensorLambda

        self._lambdaRange = None
        if sensorBandwidth >= 100e-9:
            nLambdas = sensorBandwidth//50e-9
            if nLambdas % 2: # make sure nLambdas is odd to include lambda0InM
                nLambdas += 1
            self._lambdaRange = xp.linspace(sensorLambda-sensorBandwidth/2, 
                                            sensorLambda+sensorBandwidth/2, 
                                            int(nLambdas))

        self.modulationAngleInLambdaOverD = None

        self.dtype = xp.float
        self.cdtype = xp.cfloat


    def get_intensity(self, input_field, lambda0OverD):
        """
        Computes the intensity on the detector of the pyramid wavefront sensor
        given an input electric field.
        """
        L = max(input_field.shape) # TBI: deal with non-square input fields
        padded_field = xp.pad(input_field, int((self.oversampling-1)/2*L), mode='constant', constant_values=0.0)

        if self._lambdaRange is None:
            intensity = self._intensity_from_field(padded_field, lambda0OverD)
        else:
            intensity = xp.zeros(padded_field.shape)
            lambdasOverD = lambda0OverD/self.lambdaInM*self._lambdaRange
            for lambdaOverD in lambdasOverD:
                rescaled_field = padded_field * (self.lambdaInM/lambdaOverD)
                intensity += self._intensity_from_field(rescaled_field, lambda0OverD)/len(self._lambdaRange) 

        return intensity
    
    def _intensity_from_field(self, padded_field, lambdaOverD):
        if self.modulationAngleInLambdaOverD*(2*xp.pi)*self.oversampling < 0.1:
            output_field = self.propagate(padded_field, lambdaOverD)
            intensity = xp.abs(output_field)**2
        else:
            intensity = self.modulate(padded_field, lambdaOverD)
        return intensity
    

    def set_modulation_angle(self, modulationAngleInLambdaOverD,verbose:bool=True):
        self.modulationAngleInLambdaOverD = modulationAngleInLambdaOverD
        self._modNsteps = xp.ceil(modulationAngleInLambdaOverD*2.25*xp.pi)//4*4
        if verbose:
            print(f'Modulating {modulationAngleInLambdaOverD:1.0f} [lambda/D] with {self._modNsteps:1.0f} modulation steps')
    

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
        self.field_on_focal_plane = xp.fft.fftshift(xp.fft.fft2(input_field))

        phase_delay = self.pyr3_phase_delay(input_field.shape) / lambdaOverD
        self._ef_focal_plane_delayed = self.field_on_focal_plane * xp.exp(1j*phase_delay, dtype=self.cdtype)

        output_field = xp.fft.ifft2(xp.fft.ifftshift(self._ef_focal_plane_delayed))

        return output_field
    

    def modulate(self, input_field, lambdaOverD):
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

        alpha_pix = self.modulationAngleInLambdaOverD*self.oversampling*(2*xp.pi)
        phi_vec = (2*xp.pi)*xp.arange(self._modNsteps)/self._modNsteps

        intensity = xp.zeros(input_field.shape,dtype = self.dtype)

        for phi in phi_vec:
            tilt = tiltX * xp.cos(phi) + tiltY * xp.sin(phi)
            tilted_input = input_field * xp.exp(1j*tilt*alpha_pix, dtype=self.cdtype)

            output = self.propagate(tilted_input, lambdaOverD)
            intensity += (abs(output**2))/self._modNsteps

        return intensity
    
    
    # @lru_cache(maxsize=5)
    def pyr3_phase_delay(self, shape):
        """
        Computes the phase delay introduced by the pyramid wavefront sensor
        in the focal plane.

        :param shape: tuple (ny, nx) electric field dimensions
        :return: array numpy 2D float (phase delay in pixels)
        """
        X,Y = image_grid(shape, recenter=True)
        p0 = X
        p1 = -0.5 * X + xp.sqrt(3)/2 * Y
        p2 = -0.5 * X - xp.sqrt(3)/2 * Y
        proj = xp.maximum(xp.maximum(p0, p1), p2)
        phi = self.vertex3_angle * (1.0 - proj / float(xp.max(proj)))
        phi = xp.asarray(phi, dtype=self.dtype)
        return phi
    

    # @lru_cache(maxsize=5)
    def _get_XY_tilt_planes(self, input_shape):
        tiltX,tiltY = image_grid(input_shape, recenter=True)
        L = max(input_shape)
        return tiltX/L,tiltY/L

