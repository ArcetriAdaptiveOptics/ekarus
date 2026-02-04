import xupy as xp
# from functools import lru_cache
from ekarus.e2e.utils.image_utils import get_circular_mask

class ZernikeWFS:
    """
    Optical modeling of a Zernike wavefront sensor for adaptive optics.

    This class simulates the propagation of an electric field through a Zernike
    wavefront sensor, applying a phase shift with a circular focal plane mask.
    """

    def __init__(self, lambdaInM:float, sizeInLambdaOverD:float, delay:float, oversampling:int, lambdaOverD:float):
        """
        Zernike wavefront sensor constructor.
        """
        self.lambdaInM = lambdaInM
        self.dot_radius = sizeInLambdaOverD
        self.phase_delay = delay
        self.oversampling = oversampling
        self.lambdaOverD = lambdaOverD

        self.dtype = xp.float
        self.cdtype = xp.cfloat


    def get_intensity(self, input_field, lambdaOverD):
        """
        Computes the intensity on the detector of the Zernike wavefront sensor
        given an input electric field.
        """
        L = max(input_field.shape) # TBI: deal with non-square input fields
        padded_field = xp.pad(input_field, int((self.oversampling-1)/2*L), mode='constant', constant_values=0.0)
        field_on_focal_plane = xp.fft.fftshift(xp.fft.fft2(padded_field))
        transmitted_field = field_on_focal_plane * self.zwfs_complex_amplitude(padded_field.shape,lambdaOverD)
        output_field = xp.fft.ifft2(xp.fft.ifftshift(transmitted_field))
        # cpix = L*self.oversampling//2
        # N = self.cropSize/2
        # cropped_field = output_field[cpix-N*L:cpix+N*L,cpix-N*L:cpix+N*L]
        # intensity = xp.abs(cropped_field)**2
        intensity = xp.abs(output_field)**2
        return intensity

    # @lru_cache(maxsize=5)
    def zwfs_complex_amplitude(self,shape,lambdaOverD):
        ratio = self.lambdaOverD/lambdaOverD
        alpha = self.dot_radius * ratio
        delta = self.phase_delay * ratio
        amp = get_circular_mask(shape,alpha*self.oversampling/2,mask_center=(shape[0]/2,shape[1]/2))
        phase = delta * (1-amp).astype(self.dtype)
        field = xp.exp(1j*phase,dtype=self.cdtype)#1.0 - (1.0 - xp.exp(1j*phase,dtype=self.cdtype))*amp
        return field
    

