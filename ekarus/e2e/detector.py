import numpy as xp

from arte.math.toccd import toccd
from ekarus.e2e.utils.image_utils import image_grid, get_photocenter, get_circular_mask


class Detector:

    def __init__(self, detector_shape = None, RON: float = 0.0, max_bits:int = 12):
        """
        Detector constructor.
        
        :param pix_size: number of pixels in the detector
        :param RON: readout noise in electrons
        """

        self.RON = RON
        self.subapertures = None
        self.detector_shape = detector_shape
        self.max_bits = max_bits

    
    def compute_slopes(self, intensity, rebin:int = 0, photon_flux = None, use_diagonal:bool=False):
        """
        Compute slopes from the intensity image using subaperture masks.

        :param intensity: 2D array of intensity values.
        :param Npix: pixel diameter of the subaperture.        :
        :param rebin: (optional) int indicating the rebinning factor
        with 0 meaning no rebiining, 1 reducing the image size to 1/4
        Default is 0
        :param use_diagonal: if True, include diagonal terms in the slopes.

        :return: array of slopes for each subaperture
        """

        ccd_intensity = self.resize_on_detector(intensity, rebin, photon_flux)

        if self.subapertures is None:
            raise ValueError('Supaperture masks have not been defined!')

        A = ccd_intensity[~self.subapertures[0]]
        B = ccd_intensity[~self.subapertures[1]]
        C = ccd_intensity[~self.subapertures[2]]
        D = ccd_intensity[~self.subapertures[3]]

        up_down = (A+B) - (C+D)
        left_right = (A+C) - (B+D)

        slope = xp.hstack((up_down, left_right))

        if use_diagonal:
            ccd_lr = xp.fliplr(ccd_intensity)
            maskAlr = xp.fliplr(self.subapertures[0])
            maskClr = xp.fliplr(self.subapertures[2])
            Alr = ccd_lr[~maskAlr]
            Clr = ccd_lr[~maskClr]
            diag = (B+Clr) - (Alr+D)
            slope = xp.hstack((up_down, left_right, diag))

        # Normalize slopes by the mean intensity
        mean_intensity = xp.mean(xp.hstack((A,B,C,D)))
        slope *= 1/mean_intensity

        return slope
    
    
    def define_subaperture_masks(self, intensity, Npix, rebin:int = 0, photon_flux = None):
        """
        Create subaperture masks for the given shape and pixel size.

        :param intensity: 2D array of intensity values
        :param Npix: pixel diameter of the subaperture
        :param rebin: (optional) int indicating the rebinning factor
        with 0 meaning no rebiining, 1 reducing the image size to 1/4
        Default is 0

        :return: array of 4 boolean masks for each subaperture
        """

        ccd_intensity = self.resize_on_detector(intensity, rebin, photon_flux)

        ny,nx = ccd_intensity.shape

        subaperture_masks = xp.zeros((4, ny, nx), dtype=bool)

        for i in range(4):
            qy,qx = self.find_subaperture_center(ccd_intensity,quad_n=i+1)
            subaperture_masks[i] = get_circular_mask(ccd_intensity.shape, radius=Npix//2, center=(qy,qx))

        self.subapertures = subaperture_masks


    def add_electron_noise(self, intensity, flux):
        """
        Simulate detector noise based on the given intensity and real photon flux.
        
        Parameters:
        - intensity: 2D array of intensity values.
        - flux: Real photon flux in the image.
        
        Returns:
        - Noisy intensity image.
        """

        # Re-scale the intensity based on the flux
        norm_intensity = intensity*flux/xp.sum(intensity)

        # Noise
        poisson_noise = xp.random.poisson(norm_intensity, xp.shape(intensity)) # Possion noise
        readout_noise = xp.random.normal(0, self.RON, size=xp.shape(intensity)) # readout noise
        
        noisy_intensity = xp.round(norm_intensity + poisson_noise + readout_noise)

        # Saturation
        noisy_intensity = xp.minimum(2**self.max_bits,noisy_intensity)
        
        return xp.maximum(0,noisy_intensity)
    
    

    def resize_on_detector(self, image, rebin_fact:int = 0, photon_flux = None):

        ccd_size =self.detector_shape
        if rebin_fact > 0:
            rebin = 4*rebin_fact
            ccd_size = (self.detector_shape[0]//rebin, self.detector_shape[1]//rebin)
        ccd_intensity = toccd(image, ccd_size)

        if photon_flux is not None:
            ccd_intensity = self.add_electron_noise(ccd_intensity, photon_flux)

        return ccd_intensity
    


    def find_subaperture_center(self, ccd_intensity, quad_n:int = 1):

        X,Y = image_grid(ccd_intensity.shape, recenter=True)
        quadrant_mask = xp.zeros_like(ccd_intensity)

        match quad_n:
            case 1:
                quadrant_mask[xp.logical_and(X < 0, Y >= 0)] = 1
            case 2:
                quadrant_mask[xp.logical_and(X >= 0, Y >= 0)] = 1
            case 3:
                quadrant_mask[xp.logical_and(X < 0, Y < 0)] = 1
            case 4:
                quadrant_mask[xp.logical_and(X >= 0, Y < 0)] = 1
            case _:
                raise ValueError('Possible quadrant numbers are 1,2,3,4 (numbered left-to-right top-to-bottom starting from the top left)')

        quadrant_mask = xp.reshape(quadrant_mask, ccd_intensity.shape)
        intensity = ccd_intensity * quadrant_mask
        qy,qx = get_photocenter(intensity)

        # return qy,qx
        return xp.round(qy),xp.round(qx)
    

    
