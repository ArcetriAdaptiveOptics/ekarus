import numpy as xp
# import cupy as xp

from arte.types.mask import CircularMask

from arte.math.toccd import toccd


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

        ccd_intensity = self.resize_intensity_on_detector(intensity)

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
        ccd_intensity = self.resize_intensity_on_detector(intensity)

        # Find image center
        ny, nx = xp.shape(ccd_intensity)
        cx, cy = nx // 2, ny // 2

        # Coordinates wrt center
        x = xp.arange(nx) - cx
        y = xp.arange(ny) - cy
        X, Y = xp.meshgrid(x, y)

        # Use only a single quadrant
        quadrant = ccd_intensity[xp.logical_and(X < 0, Y >= 0)]
        quadrant = quadrant.reshape((ny // 2, nx // 2))

        # Find photocenter
        xquad = xp.arange(nx//2)
        yquad = xp.arange(ny//2)
        Xquad, Yquad = xp.meshgrid(xquad, yquad)
        qy = xp.sum(Yquad * quadrant) / xp.sum(quadrant)
        qx = xp.sum(Xquad * quadrant) / xp.sum(quadrant)

        # Define mask in single quadrant
        cmask = CircularMask(xp.shape(quadrant), maskRadius=Npix//2, maskCenter=(qy,qx))
        quad_mask = cmask.mask()

        # Pad and flip mask to 4 quadrants
        padded_mask = xp.ones(xp.size(ccd_intensity))
        padded_mask[xp.logical_and(X.flatten() < 0, Y.flatten() >= 0)] = quad_mask.flatten()
        padded_mask = (padded_mask.reshape(ccd_intensity.shape)).astype(bool)    

        mask_A = padded_mask.copy()
        mask_B = xp.roll(mask_A, shift = 2*(cx-qx), axis = 1)
        mask_C = xp.roll(mask_A, shift = 2*(cy-qy), axis = 0)
        mask_aux = xp.roll(mask_A, shift = 2*(cy-qy), axis = 0)
        mask_D = xp.roll(mask_aux, shift = 2*(cx-qx), axis = 1)
        # mask_B = xp.flip(mask_A, axis=1)
        # mask_C = xp.flip(mask_A, axis=0)
        # mask_D = xp.flip(mask_B, axis=0)

        subaperture_masks = xp.zeros((4, ny, nx), dtype=bool)
        subaperture_masks[0] = mask_A
        subaperture_masks[1] = mask_B
        subaperture_masks[2] = mask_C
        subaperture_masks[3] = mask_D

        self.subapertures = subaperture_masks

        return subaperture_masks


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
    
    
    def resize_intensity_on_detector(self, intensity, flux=None):
        """ Rescales to detector shape and adjusts the flux """
        return toccd(intensity, self.detector_shape, set_total=flux)


    def _get_circular_mask(self, shape, radius, center=None):
        """
        Create a circular mask for the given shape.
        
        :param shape: tuple (ny, nx) dimensions of the mask
        :param radius: radius of the circular mask
        :param center: tuple (cy, cx) center of the circular mask
        :return: boolean numpy array with the mask
        """
        mask = CircularMask(shape, maskRadius=radius, maskCenter=center)
        return (mask.mask()).astype(bool)
    

    # @staticmethod
    # def _toccd(a, newshape, set_total=None):
    #     '''
    #     Clone of oaalib's toccd() function, using least common multiple
    #     to rebin an array similar to opencv's INTER_AREA interpolation.
    #     '''
    #     from arte.math.factors import lcm
    #     from arte.utils.rebin import rebin

    #     if (a.shape == newshape).all():
    #         return a

    #     if len(a.shape) != 2:
    #         raise ValueError('Input array shape is %s instead of 2d, cannot continue:' % str(a.shape))

    #     if len(newshape) != 2:
    #         raise ValueError('Output shape is %s instead of 2d, cannot continue' % str(newshape))

    #     if set_total is None:
    #         set_total = a.sum()

    #     mcmx = lcm(a.shape[0], newshape[0])
    #     mcmy = lcm(a.shape[1], newshape[1])

    #     temp = rebin(a, (mcmx, a.shape[1]), sample=True)
    #     temp = rebin(temp, (newshape[0], a.shape[1]))
    #     temp = rebin(temp, (newshape[0], mcmy), sample=True)
    #     rebinned = rebin(temp, newshape)

    #     return rebinned / rebinned.sum() * set_total
    

    
