import numpy as np
import os

from ekarus.e2e.utils import my_fits_package as myfits
from ekarus.e2e.utils.read_configuration import ConfigReader
from ekarus.analytical.turbulence_layers import TurbulenceLayers

# from ekarus.e2e.alpao_deformable_mirror import ALPAODM
# from ekarus.e2e.pyramid_wfs import PyramidWFS
# from ekarus.e2e.detector import Detector
# from ekarus.e2e.slope_computer import SlopeComputer

from ekarus.e2e.utils.image_utils import get_circular_mask, reshape_on_mask
from ekarus.analytical.kl_modes import make_modal_base_from_ifs_fft


class HighLevelAO():

    def __init__(self, tn, xp=np):

        self.basepath = os.getcwd()

        self.resultpath = os.path.join(self.basepath,'ekarus','simulations','Results')
        if not os.path.exists(self.resultpath):
            os.mkdir(self.resultpath)

        self.savepath = os.path.join(self.resultpath,str(tn))
        if not os.path.exists(self.savepath):
            os.mkdir(self.savepath)

        self._xp = xp
        self.dtype = xp.float32 if xp.__name__ == 'cupy' else xp.float64

        self._read_configuration(tn)
        self._read_loop_parameters()


    def get_photons_per_second(self, starMagnitude, B0 = 1e+10):
        collected_flux = None
        if starMagnitude is not None:
            total_flux = B0 * 10**(-starMagnitude/2.5)
            collecting_area = self._xp.pi/4*self.pupilSizeInM**2
            collected_flux = self.throughput * total_flux * collecting_area
        return collected_flux
    

    def _read_configuration(self, tn):
        config_path = os.path.join(self.basepath,'ekarus','simulations','Config',str(tn)+'.ini')
        self._config = ConfigReader(config_path, self._xp)
        self.pupilSizeInM, self.pupilSizeInPixels, self.throughput = self._config.read_telescope_pars()
        mask_shape = (self.pupilSizeInPixels, self.pupilSizeInPixels)
        self.cmask = get_circular_mask(mask_shape, mask_radius=self.pupilSizeInPixels//2, xp=self._xp)

    def _read_loop_parameters(self):
        nIterations, loopFrequencyInHz, integratorGain, delay, nModes2Correct = self._config.read_loop_pars()
        self.dt = 1/loopFrequencyInHz
        self.Nits = nIterations
        self.integratorGain = integratorGain
        self.delaySteps = delay
        self.nModes = nModes2Correct


    def define_subaperture_masks(self, slope_computer, lambdaInM):
        subap_path = os.path.join(self.savepath,'SubapertureMasks.fits')
        try:
            subaperture_masks = myfits.read_fits(subap_path, isBool=True)
            if self._xp.__name__ == 'cupy':
                subaperture_masks = self._xp.asarray(subaperture_masks)
            slope_computer._subaperture_masks = subaperture_masks
        except FileNotFoundError:
            subapertureSizeInPixels = self._get_subaperture_pixel_size(slope_computer)
            piston = 1-self.cmask
            lambdaOverD = lambdaInM/self.pupilSizeInM
            slope_computer.calibrate_sensor(piston, lambdaOverD, subapertureSizeInPixels)
            subaperture_masks = slope_computer._subaperture_masks
            hdr_dict = {'APEX_ANG': slope_computer._wfs.apex_angle, 'RAD2PIX': lambdaOverD, 'OVERSAMP': slope_computer._wfs.oversampling,  'SUBAPPIX': subapertureSizeInPixels}
            myfits.save_fits(subap_path, (subaperture_masks).astype(self._xp.uint8), hdr_dict)
        return slope_computer

    def _get_subaperture_pixel_size(self, slope_computer): 
        image_size = self.pupilSizeInPixels*slope_computer._wfs.oversampling
        rebin_factor = min(slope_computer._detector.detector_shape)/image_size
        pupilPixelSizeOnDetector = self.pupilSizeInPixels * rebin_factor
        return pupilPixelSizeOnDetector-0.5 
    

    def define_KL_modes(self, dm, oversampling, zern_modes:int = 5):
        KL_path = os.path.join(self.savepath,'KLmodes.fits')
        m2c_path = os.path.join(self.savepath,'m2c.fits')
        try:
            KL = myfits.read_fits(KL_path)
            m2c = myfits.read_fits(m2c_path)
            if self._xp.__name__ == 'cupy':
                KL = self._xp.asarray(KL, dtype=self._xp.float32)
                m2c = self._xp.asarray(m2c, dtype=self._xp.float32)
        except FileNotFoundError:
            r0s, L0, _, _ = self._config.read_atmo_pars()
            if isinstance(r0s, float):
                r0 = r0s
            else:
                r0 = self._xp.sqrt(self._xp.sum(r0s**2))
            KL, m2c, _ = make_modal_base_from_ifs_fft(1-dm.mask, self.pupilSizeInPixels, \
                self.pupilSizeInM, dm.IFF.T, r0, L0, zern_modes=zern_modes,\
                oversampling=oversampling, verbose = True, xp=self._xp, dtype=self.dtype)
            hdr_dict = {'r0': r0, 'L0': L0, 'N_ZERN': zern_modes}
            myfits.save_fits(KL_path, KL, hdr_dict)
            myfits.save_fits(m2c_path, m2c, hdr_dict)
        return KL, m2c
    

    def calibrate_modes(self, slope_computer, MM, lambdaInM, amps):
        IM_path = os.path.join(self.savepath,'IM.fits')
        Rec_path = os.path.join(self.savepath,'Rec.fits')
        try:
            IM = myfits.read_fits(IM_path)
            Rec = myfits.read_fits(Rec_path)
            if self._xp.__name__ == 'cupy':
                IM = self._xp.asarray(IM, dtype=self._xp.float32)
                Rec = self._xp.asarray(Rec, dtype=self._xp.float32)
        except FileNotFoundError:
            Rec, IM, _ = self._compute_reconstructor(slope_computer, MM, lambdaInM, amps)
            myfits.save_fits(IM_path, IM)
            myfits.save_fits(Rec_path, Rec)
        return Rec, IM
    
    
    def _compute_reconstructor(self, slope_computer, MM, lambdaInM, amps):
        Nmodes = self._xp.shape(MM)[0]
        slopes = None
        electric_field_amp = 1-self.cmask

        lambdaOverD = lambdaInM/self.pupilSizeInM
        Nphotons = None # perfect calibration: no noise

        if isinstance(amps, float):
            amps *= self._xp.ones(Nmodes)

        for i in range(Nmodes):
            print(f'\rMode {i+1}/{Nmodes}', end='')
            amp = amps[i]
            mode_phase = reshape_on_mask(MM[i,:]*amp, self.cmask, xp=self._xp)
            input_field = self._xp.exp(1j*mode_phase) * electric_field_amp
            push_slope = slope_computer.compute_slopes(input_field, lambdaOverD, Nphotons)/amp #self.get_slopes(input_field, Nphotons)/amp

            input_field = self._xp.conj(input_field)
            pull_slope = slope_computer.compute_slopes(input_field, lambdaOverD, Nphotons)/amp #self.get_slopes(input_field, Nphotons)/amp

            if slopes is None:
                slopes = (push_slope-pull_slope)/2
            else:
                slopes = self._xp.vstack((slopes,(push_slope-pull_slope)/2))
        print(' ')
        IM = slopes.T
        U,S,Vt = self._xp.linalg.svd(IM, full_matrices=False)
        Rec = (Vt.T*1/S) @ U.T

        return Rec, IM, 1/S
    
    
    def initialize_turbulence(self, N:int=None):
        r0s, L0, windSpeeds, windAngles = self._config.read_atmo_pars()

        if N is None:
            maxTime = self.dt * self.Nits
            if isinstance(windSpeeds,float) or isinstance(windSpeeds,int):
                maxSpeed = windSpeeds
            else:
                maxSpeed = windSpeeds.max()
            maxLen = maxSpeed*maxTime
            N = int(self._xp.ceil(maxLen/self.pupilSizeInM))
        else:
            N = input.copy()
        N = int(self._xp.max(self._xp.array([10,N]))) # set minimum N to 10

        screenPixels = N*self.pupilSizeInPixels
        screenMeters = N*self.pupilSizeInM 
        atmo_path = os.path.join(self.savepath, 'AtmospherePhaseScreens.fits')
        self.layers = TurbulenceLayers(r0s, L0, windSpeeds, windAngles, atmo_path, xp=self._xp)
        self.layers.generate_phase_screens(screenPixels, screenMeters)
        self.layers.rescale_phasescreens() # rescale in meters
        self.layers.update_mask(self.cmask)


    def get_phasescreen_at_time(self, time):
        masked_phases = self.layers.move_mask_on_phasescreens(time)
        masked_phase = self._xp.sum(masked_phases,axis=0)
        return masked_phase
    
    
    def save_telemetry_data(self, data_dict):
        for key in data_dict:
            file_path = os.path.join(self.savepath,str(key)+'.fits')
            myfits.save_fits(file_path, data_dict[key])








        