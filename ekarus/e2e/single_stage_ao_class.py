import numpy as np
import os

from ekarus.e2e.utils import my_fits_package as myfits
from ekarus.e2e.utils.read_configuration import ConfigReader
from ekarus.analytical.turbulence_layers import TurbulenceLayers

from ekarus.e2e.alpao_deformable_mirror import ALPAODM
from ekarus.e2e.pyramid_wfs import PyramidWFS
from ekarus.e2e.detector import Detector
from ekarus.e2e.slope_computer import SlopeComputer

from ekarus.e2e.utils.image_utils import get_circular_mask, reshape_on_mask
from ekarus.analytical.kl_modes import make_modal_base_from_ifs_fft


class SingleStageAO():

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
        self._initialize_devices()

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
        # self.pupilSizeInM, self.pupilSizeInPixels, self.oversampling, self.throughput = self._config.read_telescope_pars()
        # mask_shape = (self.oversampling * self.pupilSizeInPixels, self.oversampling * self.pupilSizeInPixels)
        self.pupilSizeInM, self.pupilSizeInPixels, self.throughput = self._config.read_telescope_pars()
        mask_shape = (self.pupilSizeInPixels, self.pupilSizeInPixels)
        self.cmask = get_circular_mask(mask_shape, mask_radius=self.pupilSizeInPixels//2, xp=self._xp)
        
    def _initialize_devices(self):
        apex_angle, oversampling, modulationAngleInLambdaOverD = self._config.read_sensor_pars()
        detector_shape, RON, quantum_efficiency = self._config.read_detector_pars()
        Nacts = self._config.read_dm_pars()

        self.pyr = PyramidWFS(apex_angle, oversampling, xp=self._xp)
        self.pyr.set_modulation_angle(modulationAngleInLambdaOverD)
        self.ccd = Detector(detector_shape=detector_shape, RON=RON, quantum_efficiency=quantum_efficiency, xp=self._xp)
        self.dm = ALPAODM(Nacts, Npix=self.pupilSizeInPixels, xp=self._xp)

        # self.slope_computer = SlopeComputer(wfs_type='PyrWFS', xp=self._xp) # OLD
        self.slope_computer = SlopeComputer(self.pyr, self.ccd, xp=self._xp)
    
    def _read_loop_parameters(self):
        nIterations, loopFrequencyInHz, integratorGain, delay, nModes2Correct = self._config.read_loop_pars()
        self.dt = 1/loopFrequencyInHz
        self.Nits = nIterations
        self.integratorGain = integratorGain
        self.delaySteps = delay
        self.nModes = nModes2Correct

    def define_subaperture_masks(self, lambdaInM):
        subap_path = os.path.join(self.savepath,'SubapertureMasks.fits')
        try:
            subaperture_masks = myfits.read_fits(subap_path, isBool=True)
            if self._xp.__name__ == 'cupy':
                subaperture_masks = self._xp.asarray(subaperture_masks)
            self.slope_computer._subaperture_masks = subaperture_masks
        except FileNotFoundError:

            subapertureSizeInPixels = self._get_subaperture_pixel_size()
            piston = 1-self.cmask
            lambdaOverD = lambdaInM/self.pupilSizeInM
            
            self.slope_computer.calibrate_sensor(piston, lambdaOverD, subapertureSizeInPixels)

            subaperture_masks = self.slope_computer._subaperture_masks
            hdr_dict = {'APEX_ANG': self.pyr.apex_angle, 'RAD2PIX': lambdaOverD, 'OVERSAMP': self.pyr.oversampling,  'SUBAPPIX': subapertureSizeInPixels}
            myfits.save_fits(subap_path, (subaperture_masks).astype(self._xp.uint8), hdr_dict)

    def _get_subaperture_pixel_size(self): # this should go in the slope_computer or should be an input (?)
        image_size = self.pupilSizeInPixels*self.pyr.oversampling
        rebin_factor = min(self.ccd.detector_shape)/image_size
        pupilPixelSizeOnDetector = self.pupilSizeInPixels * rebin_factor
        return pupilPixelSizeOnDetector-0.5 
    

    def define_KL_modes(self, zern_modes:int = 5):
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

            KL, m2c, _ = make_modal_base_from_ifs_fft(1-self.dm.mask, self.pupilSizeInPixels, \
                self.pupilSizeInM, self.dm.IFF.T, r0, L0, zern_modes=zern_modes,\
                oversampling=self.pyr.oversampling, verbose = True, xp=self._xp, dtype=self.dtype)
            hdr_dict = {'r0': r0, 'L0': L0, 'N_ZERN': zern_modes}
            myfits.save_fits(KL_path, KL, hdr_dict)
            myfits.save_fits(m2c_path, m2c, hdr_dict)
        return KL, m2c
    

    def calibrate_modes(self, MM, lambdaInM, amps):
        IM_path = os.path.join(self.savepath,'IM.fits')
        Rec_path = os.path.join(self.savepath,'Rec.fits')
        try:
            IM = myfits.read_fits(IM_path)
            Rec = myfits.read_fits(Rec_path)
            if self._xp.__name__ == 'cupy':
                IM = self._xp.asarray(IM, dtype=self._xp.float32)
                Rec = self._xp.asarray(Rec, dtype=self._xp.float32)
        except FileNotFoundError:
            Rec, IM, _ = self._compute_reconstructor(MM, lambdaInM, amps)
            myfits.save_fits(IM_path, IM)
            myfits.save_fits(Rec_path, Rec)
        return Rec, IM
    
    
    def _compute_reconstructor(self, MM, lambdaInM, amps):

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
            push_slope = self.slope_computer.compute_slopes(input_field, lambdaOverD, Nphotons)/amp #self.get_slopes(input_field, Nphotons)/amp

            input_field = self._xp.conj(input_field)
            pull_slope = self.slope_computer.compute_slopes(input_field, lambdaOverD, Nphotons)/amp #self.get_slopes(input_field, Nphotons)/amp

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


    def run_loop(self, lambdaInM, starMagnitude, Rec, m2c):
        # self.pyr.set_modulation_angle(self.modulationAngleInLambdaOverD)
        electric_field_amp = 1-self.cmask

        modal_gains = self._xp.zeros(Rec.shape[0])
        modal_gains[:self.nModes] = 1

        lambdaOverD = lambdaInM/self.pupilSizeInM
        Nphotons = self.get_photons_per_second(starMagnitude) * self.dt

        # Define variables
        mask_len = int(self._xp.sum(1-self.dm.mask))
        dm_cmd = self._xp.zeros(self.dm.Nacts, dtype=self.dtype)
        self.dm.set_position(dm_cmd, absolute=True)

        # Save telemetry
        dm_cmds = self._xp.zeros([self.Nits,self.dm.Nacts])
        dm_phases = self._xp.zeros([self.Nits,mask_len])
        residual_phases = self._xp.zeros([self.Nits,mask_len])
        input_phases = self._xp.zeros([self.Nits,mask_len])
        detector_images = self._xp.zeros([self.Nits,self.ccd.detector_shape[0],self.ccd.detector_shape[1]])

        # X,Y = image_grid(self.cmask.shape, recenter=True, xp=self._xp)
        # tiltX = X[~self.cmask]/(self.cmask.shape[0]//2)
        # tiltY = Y[~self.cmask]/(self.cmask.shape[1]//2)
        # TTmat = self._xp.stack((tiltX.T,tiltY.T),axis=1)

        for i in range(self.Nits):
            print(f'\rIteration {i+1}/{self.Nits}', end='')
            sim_time = self.dt*i

            atmo_phase = self.get_phasescreen_at_time(sim_time)
            input_phase = atmo_phase[~self.cmask]
            input_phase -= self._xp.mean(input_phase) # remove piston

            # # Tilt offloading
            # if i>0 and ttOffloadFrequency > 0 and i % int(loopFrequencyInHz/ttOffloadFrequency) <= 1e-6:
            #     tt_coeffs = modes[:2]
            #     input_phase -= TTmat @ tt_coeffs

            if i >= self.delaySteps:
                self.dm.set_position(dm_cmds[i-self.delaySteps,:], absolute=True)
            residual_phase = input_phase - self.dm.surface
            delta_phase_in_rad = reshape_on_mask(residual_phase*(2*self._xp.pi)/lambdaInM, self.cmask, xp=self._xp)

            input_field = electric_field_amp * self._xp.exp(1j*delta_phase_in_rad)
            slopes = self.slope_computer.compute_slopes(input_field, lambdaOverD, Nphotons)
            modes = Rec @ slopes
            modes *= modal_gains
            cmd = m2c @ modes
            dm_cmd += cmd*self.integratorGain
            dm_cmds[i,:] = dm_cmd*lambdaInM/(2*self._xp.pi) # convert to meters
            
            # Save telemetry
            dm_phases[i,:] = self.dm.surface
            input_phases[i,:] = input_phase
            detector_images[i,:,:] = self.ccd.last_frame
            residual_phases[i,:] = residual_phase
        print('')

        return input_phases, residual_phases, dm_phases, dm_cmds, detector_images 



    def get_phasescreen_at_time(self, time):
        masked_phases = self.layers.move_mask_on_phasescreens(time)
        masked_phase = self._xp.sum(masked_phases,axis=0)
        return masked_phase







        