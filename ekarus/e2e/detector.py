import numpy as xp
# import cupy as xp

from arte.math.toccd import toccd
from ekarus.e2e.utils.image_utils import image_grid, get_photocenter, get_circular_mask


class Detector:

    def __init__(self, detector_shape = None, RON: float = 0.0):
        """
        Detector constructor.
        
        :param pix_size: number of pixels in the detector
        :param RON: readout noise in electrons
        """

        self.RON = RON
        self.subapertures = None
        self.detector_shape = detector_shape

    
    def compute_slopes(self, intensity, use_diagonal:bool=False):
        """
        Compute slopes from the intensity image using subaperture masks.

        :param intensity: 2D array of intensity values.
        :param Npix: pixel diameter of the subaperture.
        :param use_diagonal: if True, include diagonal terms in the slopes.

        :return: array of slopes for each subaperture
        """

        ccd_intensity = xp.array(toccd(intensity, self.detector_shape),dtype=float)

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
            diag = (B+C) - (A+D)
            slope = xp.hstack((up_down, left_right, diag))

        # Normalize slopes by the mean intensity
        mean_intensity = xp.mean(xp.hstack((A,B,C,D)))
        slope *= 1/mean_intensity

        return slope
    
    
    def define_subaperture_masks(self, intensity, Npix):
        """
        Create subaperture masks for the given shape and pixel size.

        :param intensity: 2D array of intensity values.
        :param Npix: pixel diameter of the subaperture

        :return: array of 4 boolean masks for each subaperture
        """

        ccd_intensity = xp.array(toccd(intensity, self.detector_shape),dtype=float)
        ny,nx = ccd_intensity.shape

        subaperture_centers = xp.zeros([4,2])
        subaperture_masks = xp.zeros((4, ny, nx), dtype=bool)

        for i in range(4):
            qy,qx = self.find_subaperture_center(ccd_intensity,quad_n=i+1)
            subaperture_centers[i,:] = xp.array([qy,qx])
            subaperture_masks[i] = get_circular_mask(ccd_intensity.shape, radius=Npix//2, center=(qy,qx))

        self.subapertures = subaperture_masks
        self.subaperture_centers = subaperture_centers


    def electron_noise(self, intensity, flux=None):
        """
        Simulate detector noise based on the given intensity and real photon flux.
        
        Parameters:
        - intensity: 2D array of intensity values.
        - flux: Real photon flux in the image.
        
        Returns:
        - Noisy intensity image.
        """
        if flux is None:
            flux = xp.sum(intensity)

        # Re-scale the intensity based on the flux
        norm_intensity = intensity*flux/xp.sum(intensity)

        # Noise
        poisson_noise = xp.random.poisson(norm_intensity, xp.shape(intensity)) # Possion noise
        readout_noise = xp.random.normal(0, self.RON, size=xp.shape(intensity)) # readout noise
        
        noisy_intensity = xp.round(norm_intensity + poisson_noise + readout_noise)
        
        return xp.maximum(0,noisy_intensity)
    


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
    

    # @ staticmethod
    # def _find_quadrant(ccd_intensity, quad_n:int = 1):

    #     ny, nx = xp.shape(ccd_intensity)
    #     x = xp.arange(nx)
    #     y = xp.arange(ny)
    #     X, Y = xp.meshgrid(x, y)

    #     match quad_n:
    #         case 1:
    #             quadrant = ccd_intensity[xp.logical_and(X < nx//2, Y >= ny//2)]
    #         case 2:
    #             quadrant = ccd_intensity[xp.logical_and(X >= nx//2, Y >= ny//2)]
    #         case 3:
    #             quadrant = ccd_intensity[xp.logical_and(X < nx//2, Y < ny//2)]
    #         case 4:
    #             quadrant = ccd_intensity[xp.logical_and(X >= nx//2, Y < ny//2)]
    #         case _:
    #             raise ValueError('Possible quadrant numbers are 1,2,3,4 (numbered left-to-right top-to-bottom starting from the top left)')
        
    #     quadrant = quadrant.reshape((ny // 2, nx // 2))

    #     return quadrant
    
    # @ staticmethod
    # def _find_first_quadrant(ccd_intensity, quad_n:int = 1):
    #     # Find image center
    #     ny, nx = xp.shape(ccd_intensity)
    #     cx, cy = nx // 2, ny // 2

    #     # Coordinates wrt center
    #     x = xp.arange(nx) - cx
    #     y = xp.arange(ny) - cy
    #     X, Y = xp.meshgrid(x, y)

    #     quadrant = ccd_intensity[xp.logical_and(X < 0, Y >= 0)]
    #     quadrant = quadrant.reshape((ny // 2, nx // 2))

    #     return quadrant
    
    

    
