from os.path import join
import xupy as xp
np = xp.np

from ekarus.e2e.utils.image_utils import image_grid, get_photocenter, get_circular_mask
from .utils.my_fits_package import save_fits, read_fits
from .utils.root import resultspath


class SlopeComputer():

    def __init__(self, wfs, detector):

        self._wfs = wfs
        self._detector = detector

        if hasattr(wfs,'apex_angle'):
            self.wfs_type = 'PyrWFS'
        else:
            raise NotImplementedError('Unrecognized sensor type. Available types are: PyrWFS')

        self.dtype = xp.float


    def calibrate_sensor(self, tn: str, prefix_str: str, *args):
        """
        Calibrates the sensor.
        
        Cases
        -----
        - PyrWFS: defines the subaperture masks
        """
        match self.wfs_type:
            case 'PyrWFS':
                subap_path = join(resultspath, tn, f'{prefix_str}_SubapertureMasks.fits')
                piston, lambdaOverD, subaperturePixelSize = args
                try:
                    subaperture_masks = read_fits(subap_path).astype(bool)
                    self._subaperture_masks = xp.asarray(subaperture_masks)
                except FileNotFoundError:
                    subapertureSizeInPixels = self._get_subaperture_pixel_size(subaperturePixelSize)
                    self.calibrate_sensor(piston, lambdaOverD, subapertureSizeInPixels)
                    hdr_dict = {'APEX_ANG': self._wfs.apex_angle, 'RAD2PIX': lambdaOverD, 'OVERSAMP': self._wfs.oversampling,  'SUBAPPIX': subapertureSizeInPixels}
                    save_fits(subap_path, (self._subaperture_masks).astype(xp.uint8), hdr_dict)
                oldAngle = self._wfs.modulationAngleInLambdaOverD
                self._wfs.set_modulation_angle(modulationAngleInLambdaOverD=10,verbose=False) # modulate a lot during subaperture definition
                modulated_intensity = self._wfs.get_intensity(piston, lambdaOverD)
                self._wfs.set_modulation_angle(modulationAngleInLambdaOverD=oldAngle,verbose=False)
                detector_image = self._detector.image_on_detector(modulated_intensity)
                self._define_subaperture_masks(detector_image, subaperturePixelSize)
            case _:
                raise NotImplementedError('Unrecognized sensor type. Available types are: PyrWFS')
        

    def _get_subaperture_pixel_size(self, pupilSizeInPixels): 
        image_size = pupilSizeInPixels*self._wfs.oversampling
        rebin_factor = min(self._detector.detector_shape)/image_size
        pupilPixelSizeOnDetector = pupilSizeInPixels * rebin_factor
        return pupilPixelSizeOnDetector-0.5 

    def compute_slopes(self, input_field, lambdaOverD, nPhotons, **kwargs):
        """
        Compute slopes from the input field
        """

        intensity = self._wfs.get_intensity(input_field, lambdaOverD)
        detector_image = self._detector.image_on_detector(intensity, photon_flux=nPhotons)

        match self.wfs_type:
            case 'PyrWFS':
                slopes = self._compute_pyramid_slopes(detector_image, **kwargs)
            case _:
                raise NotImplementedError('Unrecognized sensor type. Available types are: PyrWFS')

        return slopes


    def _compute_pyramid_slopes(self, detector_image, use_diagonal:bool=False):

        A = detector_image[~self._subaperture_masks[0]]
        B = detector_image[~self._subaperture_masks[1]]
        C = detector_image[~self._subaperture_masks[2]]
        D = detector_image[~self._subaperture_masks[3]]

        up_down = (A+B) - (C+D)
        left_right = (A+C) - (B+D)

        slopes = self._xp.hstack((up_down, left_right))

        if use_diagonal:
            ccd_lr = self._xp.fliplr(detector_image)
            maskAlr = self._xp.fliplr(self._subaperture_masks[0])
            maskClr = self._xp.fliplr(self._subaperture_masks[2])
            Alr = ccd_lr[~maskAlr]
            Clr = ccd_lr[~maskClr]
            diag = (B+Clr) - (Alr+D)
            slopes = self._xp.hstack((up_down, left_right, diag))

        mean_intensity = self._xp.mean(self._xp.hstack((A,B,C,D)))
        slopes *= 1/mean_intensity

        return slopes
    
    
    def _define_subaperture_masks(self, subaperture_image, Npix):
        """
        Create subaperture masks for the given shape and pixel size.

        :return: array of 4 boolean masks for each subaperture
        """

        ny,nx = subaperture_image.shape
        subaperture_masks = self._xp.zeros((4, ny, nx), dtype=bool)

        for i in range(4):
            qy,qx = self.find_subaperture_center(subaperture_image, quad_n=i+1, xp=self._xp, dtype=self.dtype)
            subaperture_masks[i] = get_circular_mask(subaperture_image.shape, mask_radius=Npix//2, mask_center=(qy,qx), xp=self._xp)

        self._subaperture_masks = subaperture_masks

    
    @staticmethod
    def find_subaperture_center(detector_image, quad_n:int = 1, xp=np, dtype=np.float32):

        X,Y = image_grid(detector_image.shape, recenter=True, xp=xp)
        quadrant_mask = xp.zeros_like(detector_image, dtype=dtype)

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

        quadrant_mask = xp.reshape(quadrant_mask, detector_image.shape)
        intensity = detector_image * quadrant_mask
        qy,qx = get_photocenter(intensity, xp=xp)

        # return qy,qx
        return xp.round(qy), xp.round(qx)