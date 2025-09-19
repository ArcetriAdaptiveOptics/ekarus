from os.path import join
import xupy as xp
np = xp.np

from ekarus.e2e.utils.image_utils import image_grid, get_photocenter, get_circular_mask
from .utils.my_fits_package import save_fits, read_fits
from .utils.root import calibpath #resultspath


class SlopeComputer():

    def __init__(self, wfs, detector, sc_pars):
        """ The constructor """

        self._wfs = wfs
        self._detector = detector

        (
            self.dt,
            self.intGain,
            self.delay,
            self.nModes,
            self.ttOffloadFreqHz,
        ) = (
            1 / sc_pars["loopFrequencyInHz"],
            sc_pars["integratorGain"],
            sc_pars["delay"],
            sc_pars["nModes2Correct"],
            sc_pars["ttOffloadFrequencyInHz"],
        )

        if hasattr(wfs,'apex_angle'):
            self.wfs_type = 'PyrWFS'
            self.modulationAngleInLambdaOverD = sc_pars["modulationInLambdaOverD"]
        else:
            raise NotImplementedError('Unrecognized sensor type. Available types are: PyrWFS')
        

        self.dtype = xp.float


    def calibrate_sensor(self, tn:str, prefix_str:str, **kwargs):
        """
        Calibrates the sensor.
        
        Cases
        -----
        - PyrWFS: defines the subaperture masks
        """
        match self.wfs_type:
            case 'PyrWFS':
                subap_path = join(calibpath, tn, prefix_str+'SubapertureMasks.fits')
                piston, lambdaOverD, subaperturePixelSize = kwargs['piston'], kwargs['lambdaOverD'], kwargs['Npix']
                try:
                    subaperture_masks = read_fits(subap_path).astype(bool)
                    self._subaperture_masks = xp.asarray(subaperture_masks)
                except FileNotFoundError:
                    print('Defining the detector subaperture masks ...')
                    self._wfs.set_modulation_angle(modulationAngleInLambdaOverD=10) # modulate a lot during subaperture definition
                    modulated_intensity = self._wfs.get_intensity(piston, lambdaOverD)
                    detector_image = self._detector.image_on_detector(modulated_intensity)
                    self._define_subaperture_masks(detector_image, subaperturePixelSize)
                    hdr_dict = {'APEX_ANG': self._wfs.apex_angle, 'RAD2PIX': lambdaOverD, 'OVERSAMP': self._wfs.oversampling,  'SUBAPPIX': subaperturePixelSize}
                    save_fits(subap_path, (self._subaperture_masks).astype(xp.uint8), hdr_dict)
            case _:
                raise NotImplementedError('Unrecognized sensor type. Available types are: PyrWFS')
    

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


    def load_reconstructor(self, Rec, m2c):
        """
        Load the reconstructor and the mode-to-command matrix
        """
        self.Rec = Rec
        self.m2c = m2c
        modal_gains = xp.zeros(xp.shape(Rec)[0])
        modal_gains[:self.nModes] = 1
        self.modal_gains = modal_gains
        

    def _compute_pyramid_slopes(self, detector_image, use_diagonal:bool=False):

        A = detector_image[~self._subaperture_masks[0]]
        B = detector_image[~self._subaperture_masks[1]]
        C = detector_image[~self._subaperture_masks[2]]
        D = detector_image[~self._subaperture_masks[3]]

        up_down = (A+B) - (C+D)
        left_right = (A+C) - (B+D)

        slopes = xp.hstack((up_down, left_right))

        if use_diagonal:
            ccd_lr = xp.fliplr(detector_image)
            maskAlr = xp.fliplr(self._subaperture_masks[0])
            maskClr = xp.fliplr(self._subaperture_masks[2])
            Alr = ccd_lr[~maskAlr]
            Clr = ccd_lr[~maskClr]
            diag = (B+Clr) - (Alr+D)
            slopes = xp.hstack((up_down, left_right, diag))

        mean_intensity = xp.mean(xp.hstack((A,B,C,D)))
        slopes *= 1/mean_intensity

        return slopes
    
    
    def _define_subaperture_masks(self, subaperture_image, Npix):
        """
        Create subaperture masks for the given shape and pixel size.

        :return: array of 4 boolean masks for each subaperture
        """

        ny,nx = subaperture_image.shape
        subaperture_masks = xp.zeros((4, ny, nx), dtype=bool)

        for i in range(4):
            # qy,qx = self.find_subaperture_center(subaperture_image, quad_n=i+1, xp=self._xp, dtype=self.dtype)
            qx,qy = self.find_subaperture_center(subaperture_image, quad_n=i+1)
            subaperture_masks[i] = get_circular_mask(subaperture_image.shape, mask_radius=Npix//2, mask_center=(qx,qy))

        self._subaperture_masks = subaperture_masks

    
    @staticmethod
    def find_subaperture_center(detector_image, quad_n:int = 1):

        X,Y = image_grid(detector_image.shape, recenter=True)
        quadrant_mask = xp.zeros_like(detector_image, dtype=xp.float)

        match quad_n:
            # case 1:
            #     quadrant_mask[xp.logical_and(X < 0, Y >= 0)] = 1
            # case 2:
            #     quadrant_mask[xp.logical_and(X >= 0, Y >= 0)] = 1
            # case 3:
            #     quadrant_mask[xp.logical_and(X < 0, Y < 0)] = 1
            # case 4:
            #     quadrant_mask[xp.logical_and(X >= 0, Y < 0)] = 1
            case 1:
                quadrant_mask[xp.logical_and(X >= 0, Y < 0)] = 1
            case 2:
                quadrant_mask[xp.logical_and(X >= 0, Y >= 0)] = 1
            case 3:
                quadrant_mask[xp.logical_and(X < 0, Y < 0)] = 1
            case 4:
                quadrant_mask[xp.logical_and(X < 0, Y >= 0)] = 1
            case _:
                raise ValueError('Possible quadrant numbers are 1,2,3,4 (numbered left-to-right top-to-bottom starting from the top left)')

        quadrant_mask = xp.reshape(quadrant_mask, detector_image.shape)
        intensity = detector_image * quadrant_mask

        qx,qy = get_photocenter(intensity)

        # return qy,qx
        return xp.round(qy), xp.round(qx)