from os.path import join
import xupy as xp

from ekarus.e2e.utils.image_utils import image_grid, get_photocenter, get_circular_mask
from ..utils.my_fits_package import save_fits, read_fits
from ..utils.root import calibpath #resultspath


def get_iir_from_zeros_and_poles(zeros:list,poles:list):
    import sympy as sp
    import numpy as np
    zeros = np.array(zeros)
    poles = np.array(poles)
    z = sp.symbols('z')
    num = np.prod((z-zeros))
    num = sp.Poly(sp.expand(num))
    den = np.prod((z-poles))
    den = sp.Poly(sp.expand(den))
    return num.all_coeffs(), den.all_coeffs()


class SlopeComputer():

    def __init__(self, wfs, detector, sc_pars=None):
        """ The constructor """

        self._wfs = wfs
        self._detector = detector

        try:
            (
                self.dt,
                self.gains,
                self.delay,
                self.nModes,
            ) = (
                1 / sc_pars["loopFrequencyInHz"],
                sc_pars["integratorGain"],
                sc_pars["delay"],
                sc_pars["nModes2Correct"],
            )
        except KeyError:
            pass

        try:
            zeros, poles = sc_pars['zeros'], sc_pars['poles']
            n,d = get_iir_from_zeros_and_poles(zeros, poles)
            self.iir_num = xp.hstack([float(n[i]) for i in range(len(n))])
            self.iir_den = xp.hstack([float(d[i]) for i in range(len(d))])
        except KeyError:
            self.iir_num, self.iir_den = xp.array([1.0]), xp.array([1.0,-1.0])

        if len(self.gains) != len(self.nModes):
            raise ValueError(f'Integrator gains {self.gains} are not compatible with length of number of modes to correct {self.nModes}')
        
        self.modalGains = xp.hstack([xp.repeat(self.gains[i],int(self.nModes[i])) for i in range(len(self.nModes))])
        self.nModes = int(xp.max(xp.cumsum(self.nModes)))

        if len(self.gains) == 1:
            self.intGain = xp.max(self.gains) # intGain is the single integrator gain
        else:
            self.intGain = 1.0 # intGain is a scalar rescaling fro modal gains

        self.slope_null = None

        if hasattr(wfs,'apex_angle'):
            self.wfs_type = 'PWFS'
            self.modulationAngleInLambdaOverD = sc_pars["modulationInLambdaOverD"]
            self._slope_method = 'slopes'
        elif hasattr(wfs,'vertex3_angle'):
            self.wfs_type = '3PWFS'
            self.modulationAngleInLambdaOverD = sc_pars["modulationInLambdaOverD"]
            self._slope_method = 'slopes'
        elif hasattr(wfs,'dot_radius'):
            self.wfs_type = 'ZWFS'
        else:
            raise NotImplementedError('Unrecognized sensor type. Available types are: PWFS, 3PWFS, ZWFS')
        

    def iir_filter(self, x, y):
        L = xp.shape(x)[0]
        fb = xp.stack([self.iir_num[i]*x[-1-i] for i in range(min(len(self.iir_num),L))])
        iir = xp.sum(fb,axis=0)
        if self.iir_den is not None and L > 1:
            ff = xp.stack([self.iir_den[i]*y[-1-i] for i in range(1,min(len(self.iir_den),L))])
            iir -= xp.sum(ff,axis=0)
        return iir


    def calibrate_sensor(self, tn:str, prefix_str:str, recompute:bool, **kwargs):
        """
        Calibrates the sensor.
        
        Cases
        -----
        - PWFS/3PWFS: defines the subaperture masks
        """
        roi_path = join(calibpath, tn, prefix_str+'ROI-Masks.fits')    
        match self.wfs_type:
            case 'PWFS':
                self._piston, self._lambdaOverD, subaperturePixelSize, centerObscPixelSize = kwargs['piston'], kwargs['lambdaOverD'], kwargs['Npix'], kwargs['centerObscurationInPixels']
                try:
                    if recompute is True:
                        raise FileNotFoundError('Recompute is True')
                    subaperture_masks = read_fits(roi_path).astype(bool)
                    self._roi_masks = xp.asarray(subaperture_masks)
                except FileNotFoundError:
                    print('Defining the detector subaperture masks ...')
                    self._wfs.set_modulation_angle(modulationAngleInLambdaOverD=10) # modulate a lot during subaperture definition
                    modulated_intensity = self._wfs.get_intensity(self._piston, self._lambdaOverD)
                    detector_image = self._detector.image_on_detector(modulated_intensity)
                    self._define_pyr_subaperture_masks(detector_image, subaperturePixelSize, centerObscPixelSize)
                    hdr_dict = {'APEX_ANG': self._wfs.apex_angle, 'RAD2PIX': self._lambdaOverD, 'OVERSAMP': self._wfs.oversampling,  'SUBAPPIX': subaperturePixelSize}
                    save_fits(roi_path, (self._roi_masks).astype(xp.uint8), hdr_dict)
            case '3PWFS':
                self._piston, self._lambdaOverD, subaperturePixelSize, centerObscPixelSize = kwargs['piston'], kwargs['lambdaOverD'], kwargs['Npix'], kwargs['centerObscurationInPixels']
                try:
                    if recompute is True:
                        raise FileNotFoundError('Recompute is True')
                    subaperture_masks = read_fits(roi_path).astype(bool)
                    self._roi_masks = xp.asarray(subaperture_masks)
                except FileNotFoundError:
                    print('Defining the detector subaperture masks ...')
                    self._wfs.set_modulation_angle(modulationAngleInLambdaOverD=10) # modulate a lot during subaperture definition
                    modulated_intensity = self._wfs.get_intensity(self._piston, self._lambdaOverD)
                    detector_image = self._detector.image_on_detector(modulated_intensity)
                    self._define_3pyr_subaperture_masks(detector_image, subaperturePixelSize, centerObscPixelSize)
                    hdr_dict = {'APEX_ANG': self._wfs.vertex3_angle, 'RAD2PIX': self._lambdaOverD, 'OVERSAMP': self._wfs.oversampling,  'SUBAPPIX': subaperturePixelSize}
                    save_fits(roi_path, (self._roi_masks).astype(xp.uint8), hdr_dict)
            case 'ZWFS':
                try:
                    if recompute is True:
                        raise FileNotFoundError('Recompute is True')
                    subaperture_masks = read_fits(roi_path).astype(bool)
                    self._roi_masks = xp.asarray(subaperture_masks)
                except FileNotFoundError:
                    camera_shape = self._detector.detector_shape
                    roiSizeInPix = max(camera_shape)/self._wfs.cropSize
                    roi_mask = get_circular_mask(camera_shape, mask_radius=roiSizeInPix/2)
                    self._roi_masks = roi_mask
                    save_fits(roi_path, (self._roi_masks).astype(xp.uint8))
            case _:
                raise NotImplementedError('Unrecognized sensor type. Available types are: PWFS, 3PWFS, ZWFS')
    

    def compute_slopes(self, input_field, lambdaOverD, nPhotons):
        """
        Compute slopes from the input field
        """
        intensity = self._wfs.get_intensity(input_field, lambdaOverD)
        detector_image = self._detector.image_on_detector(intensity, photon_flux=nPhotons)

        match self.wfs_type:
            case 'PWFS':
                slopes = self._compute_pyr_signal(detector_image)
            case '3PWFS':
                slopes = self._compute_3pyr_signal(detector_image)
            case 'ZWFS':
                slopes = detector_image[~self._roi_masks]
            case _:
                raise NotImplementedError('Unrecognized sensor type. Available types are: PWFS, 3PWFS, ZWFS')
            
        if self.slope_null is not None:
            slopes -= self.slope_null

        return slopes
    
    
    def load_reconstructor(self, IM, m2c, method:str=None, remove_slope_null:bool=True):
        """
        Load the reconstructor and the mode-to-command matrix
        """
        Rec = xp.linalg.pinv(IM[:,:self.nModes])
        self.Rec = Rec
        self.m2c = m2c[:,:self.nModes]

        if method is not None:
            self._slope_method = method

        if remove_slope_null is True and self.slope_null is None and self.wfs_type != 'ZWFS':
            print(f'Computing slope null...')
            self.slope_null = self.compute_slopes(self._piston, self._lambdaOverD, None)


    def set_new_gain(self, intGain:float):
        """ Scale all gains by a single scalar coefficient """
        self.modalGains *= intGain/self.intGain
        self.intGain = intGain


    def set_slope_null(self, slope_null):
        self.slope_null = slope_null


    def _compute_pyr_signal(self, detector_image):
        A = detector_image[~self._roi_masks[0]]
        B = detector_image[~self._roi_masks[1]]
        C = detector_image[~self._roi_masks[2]]
        D = detector_image[~self._roi_masks[3]]

        match self._slope_method:
            case 'slopes':
                up_down = (A+B) - (C+D)
                left_right = (A+C) - (B+D)
                slopes = xp.hstack((up_down, left_right))

            case 'diagonal_slopes':
                up_down = (A+B) - (C+D)
                left_right = (A+C) - (B+D)
                # diag = (A+D) - (B+C)
                diag = xp.sqrt(2)*(A+D) - xp.sqrt(2)*(B+C)
                slopes = xp.hstack((up_down, left_right, diag))
                
            case 'raw_intensity':
                slopes = xp.hstack((A,B,C,D))

            case _:
                raise KeyError("Unrecongised method: available methods are 'slopes', 'raw_intensity', 'diagonal_slopes'")
            
        mean_intensity = xp.mean(xp.hstack((A,B,C,D)))#/4
        slopes *= 1/mean_intensity
        return slopes
    

    def _compute_3pyr_signal(self, detector_image):
        A = detector_image[~self._roi_masks[0]]
        B = detector_image[~self._roi_masks[1]]
        C = detector_image[~self._roi_masks[2]]

        match self._slope_method:
            case 'slopes':
                up_down = xp.sqrt(3)/2*(B-C)
                left_right = A - (B+C)/2
                slopes = xp.hstack((up_down, left_right))

            case 'all_slopes':
                ab = (A+B)/2-C
                ac = (A+C)/2-B
                bc = (C+B)/2-A
                slopes = xp.hstack((ab,ac,bc))
                
            case 'raw_intensity':
                slopes = xp.hstack((A,B,C))

            case _:
                raise KeyError("Unrecongised method: available methods are 'slopes', 'raw_intensity', 'all_slopes'")
            
        mean_intensity = xp.mean(xp.hstack((A,B,C)))
        slopes *= 1/mean_intensity
        return slopes
    
    
    def _define_pyr_subaperture_masks(self, subaperture_image, Npix, centerObscPixDiam:float = 0.0):
        """
        Create subaperture masks for the given shape and pixel size.

        :return: array of 4 boolean masks for each subaperture
        """
        ny,nx = subaperture_image.shape
        subaperture_masks = xp.zeros((4, ny, nx), dtype=bool)
        for i in range(4):
            # qy,qx = self.find_subaperture_center(subaperture_image, quad_n=i+1, xp=self._xp, dtype=xp.float)
            qx,qy = self.find_subaperture_center(subaperture_image, index=i+1, mode='quadrant')
            subaperture_masks[i] = get_circular_mask(subaperture_image.shape, mask_radius=Npix/2, mask_center=(qx,qy))
            if centerObscPixDiam > 0.0:
                obsc_mask = get_circular_mask(subaperture_image.shape, mask_radius=centerObscPixDiam//2, mask_center=(qx,qy))
                subaperture_masks[i] = (subaperture_masks[i] + (1-obsc_mask)).astype(bool)
        self._roi_masks = subaperture_masks


    def _define_3pyr_subaperture_masks(self, subaperture_image, Npix, centerObscPixDiam:float = 0.0):
        """
        Create subaperture masks for the given shape and pixel size.

        :return: array of 3 boolean masks for each subaperture
        """
        ny,nx = subaperture_image.shape
        subaperture_masks = xp.zeros((3, ny, nx), dtype=bool)
        for i in range(3):
            qx,qy = self.find_subaperture_center(subaperture_image, index=i+1, mode='triangle')
            subaperture_masks[i] = get_circular_mask(subaperture_image.shape, mask_radius=Npix/2, mask_center=(qx,qy))
            if centerObscPixDiam > 0.0:
                obsc_mask = get_circular_mask(subaperture_image.shape, mask_radius=centerObscPixDiam//2, mask_center=(qx,qy))
                subaperture_masks[i] = (subaperture_masks[i] + (1-obsc_mask)).astype(bool)
        self._roi_masks = subaperture_masks

    
    @staticmethod
    def find_subaperture_center(detector_image, index:int=1, mode:str='quadrant'):
        X,Y = image_grid(detector_image.shape, recenter=True)
        roi_mask = xp.zeros_like(detector_image, dtype=xp.float)
        if mode == 'quadrant':
            match index:
                case 1:
                    roi_mask[xp.logical_and(X >= 0, Y < 0)] = 1
                case 2:
                    roi_mask[xp.logical_and(X >= 0, Y >= 0)] = 1
                case 3:
                    roi_mask[xp.logical_and(X < 0, Y < 0)] = 1
                case 4:
                    roi_mask[xp.logical_and(X < 0, Y >= 0)] = 1
                case _:
                    raise ValueError('Possible quadrant numbers are 1,2,3,4 (numbered left-to-right top-to-bottom starting from the top left)')
        elif mode == 'triangle':
            match index:
                case 1:
                    roi_mask[X > abs(Y)/xp.tan(xp.pi/3)] = 1
                case 2:
                    roi_mask[xp.logical_and(Y >= 0, X <= abs(Y)/xp.tan(xp.pi/3))] = 1
                case 3:
                    roi_mask[xp.logical_and(Y < 0, X <= abs(Y)/xp.tan(xp.pi/3))] = 1
                case _:
                    raise ValueError('Possible quadrant numbers are 1,2,3 (right, top-left, bottom-left)')
        roi_mask = xp.reshape(roi_mask, detector_image.shape)
        intensity = detector_image * roi_mask
        qx,qy = get_photocenter(intensity)
        
        if mode == 'quadrant':
            return xp.round(qy), xp.round(qx)
        else:
            return xp.round(qx-0.5), xp.round(qy-0.5)