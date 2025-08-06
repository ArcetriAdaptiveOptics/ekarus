import numpy as np

from ekarus.e2e.utils.image_utils import image_grid, get_photocenter, get_circular_mask


class SlopeComputer():

    def __init__(self, sensor, **kwargs):

        if hasattr(sensor,'apex_angle'):
            self.wfs_type = 'PyrWFS'
            self._define_subaperture_masks(**kwargs)
        else:
            raise NotImplementedError('Unrecognized sensor type. Available types are: PyrWFS')
        


    def compute_slopes(self, detector_image, **kwargs):
        """
        Compute slopes from the detector image
        """

        slopes = None

        match self.wfs_type:
            case 'PyrWFS':
                slopes = self._compute_pyramid_slopes(detector_image, **kwargs)

        return slopes


    def _compute_pyramid_slopes(self, detector_image, use_diagonal:bool=False):
        """
        Compute slopes from the detector image
        """
        
        # TODO add rebin factor computation from detector image and self.subapertures shape comparison

        A = detector_image[~self.subapertures[0]]
        B = detector_image[~self.subapertures[1]]
        C = detector_image[~self.subapertures[2]]
        D = detector_image[~self.subapertures[3]]

        up_down = (A+B) - (C+D)
        left_right = (A+C) - (B+D)

        slopes = np.hstack((up_down, left_right))

        if use_diagonal:
            ccd_lr = np.fliplr(detector_image)
            maskAlr = np.fliplr(self.subapertures[0])
            maskClr = np.fliplr(self.subapertures[2])
            Alr = ccd_lr[~maskAlr]
            Clr = ccd_lr[~maskClr]
            diag = (B+Clr) - (Alr+D)
            slopes = np.hstack((up_down, left_right, diag))

        # Normalize slopes by the mean intensity
        mean_intensity = np.mean(np.hstack((A,B,C,D)))
        slopes *= 1/mean_intensity

        return slopes
    
    
    def _define_subaperture_masks(self, subaperture_image, Npix):
        """
        Create subaperture masks for the given shape and pixel size.

        :return: array of 4 boolean masks for each subaperture
        """

        ny,nx = subaperture_image.shape
        subaperture_masks = np.zeros((4, ny, nx), dtype=bool)

        for i in range(4):
            qy,qx = self.find_subaperture_center(subaperture_image,quad_n=i+1)
            subaperture_masks[i] = get_circular_mask(subaperture_image.shape, radius=Npix//2, center=(qy,qx))

        self.subapertures = subaperture_masks

    
    @staticmethod
    def find_subaperture_center(detector_image, quad_n:int = 1):

        X,Y = image_grid(detector_image.shape, recenter=True)
        quadrant_mask = np.zeros_like(detector_image)

        match quad_n:
            case 1:
                quadrant_mask[np.logical_and(X < 0, Y >= 0)] = 1
            case 2:
                quadrant_mask[np.logical_and(X >= 0, Y >= 0)] = 1
            case 3:
                quadrant_mask[np.logical_and(X < 0, Y < 0)] = 1
            case 4:
                quadrant_mask[np.logical_and(X >= 0, Y < 0)] = 1
            case _:
                raise ValueError('Possible quadrant numbers are 1,2,3,4 (numbered left-to-right top-to-bottom starting from the top left)')

        quadrant_mask = np.reshape(quadrant_mask, detector_image.shape)
        intensity = detector_image * quadrant_mask
        qy,qx = get_photocenter(intensity)

        # return qy,qx
        return np.round(qy),np.round(qx)