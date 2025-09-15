import numpy as np
import os

from ekarus.e2e.alpao_deformable_mirror import ALPAODM
from ekarus.e2e.pyramid_wfs import PyramidWFS
from ekarus.e2e.detector import Detector
from ekarus.e2e.slope_computer import SlopeComputer

from ekarus.e2e.high_level_ao_class import HighLevelAO
from ekarus.e2e.utils.image_utils import reshape_on_mask #, get_masked_array

from ekarus.e2e.utils.my_fits_package import read_fits


class SingleStageAO(HighLevelAO):

    def __init__(self, tn, xp=np):

        super().__init__(tn, xp)

        self._initialize_devices()
    
        
    def _initialize_devices(self):
        apex_angle, oversampling, modulationAngleInLambdaOverD = self._config.read_sensor_pars()
        detector_shape, RON, quantum_efficiency = self._config.read_detector_pars()
        Nacts = self._config.read_dm_pars()

        self.pyr = PyramidWFS(apex_angle, oversampling, xp=self._xp)
        self.pyr.set_modulation_angle(modulationAngleInLambdaOverD)
        self.ccd = Detector(detector_shape=detector_shape, RON=RON, quantum_efficiency=quantum_efficiency, xp=self._xp)
        self.dm = ALPAODM(Nacts, Npix=self.pupilSizeInPixels, xp=self._xp)

        self.slope_computer = SlopeComputer(self.pyr, self.ccd, xp=self._xp)


    def run_loop(self, lambdaInM, starMagnitude, Rec, m2c, save_telemetry:bool=False):
        electric_field_amp = 1-self.cmask

        modal_gains = self._xp.zeros(Rec.shape[0])
        modal_gains[:self.nModes] = 1

        lambdaOverD = lambdaInM/self.pupilSizeInM
        Nphotons = self.get_photons_per_second(starMagnitude) * self.dt

        # Define variables
        mask_len = int(self._xp.sum(1-self.dm.mask))
        dm_cmd = self._xp.zeros(self.dm.Nacts, dtype=self.dtype)
        self.dm.set_position(dm_cmd, absolute=True)
        dm_cmds = self._xp.zeros([self.Nits,self.dm.Nacts])

        if save_telemetry:
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

            residual_phases[i,:] = residual_phase
            input_phases[i,:] = input_phase
            if save_telemetry:
                dm_phases[i,:] = self.dm.surface
                detector_images[i,:,:] = self.ccd.last_frame
        print('')

        errRad2 = self._xp.std(residual_phases*(2*self._xp.pi)/lambdaInM,axis=-1)**2
        inputErrRad2 = self._xp.std(input_phases*(2*self._xp.pi)/lambdaInM,axis=-1)**2 

        if save_telemetry:
            print('Saving telemetry to .fits ...')
            ma_input_phases = np.stack([reshape_on_mask(input_phases[i,:], self.cmask, self._xp) for i in range(self.Nits)])
            ma_dm_phases = np.stack([reshape_on_mask(dm_phases[i,:], self.cmask, self._xp) for i in range(self.Nits)])
            ma_res_phases = np.stack([reshape_on_mask(residual_phases[i,:], self.cmask, self._xp) for i in range(self.Nits)])

            data_dict = {'AtmoPhases': ma_input_phases, 'DMphases': ma_dm_phases, 'ResPhases': ma_res_phases, \
                         'DetectorFrames': detector_images, 'DMcommands': dm_cmds}
            self.save_telemetry_data(data_dict)

        return errRad2, inputErrRad2
    

    def load_telemetry_data(self):
        atmo_phases = read_fits(os.path.join(self.savepath,'AtmoPhases.fits'))
        dm_phases = read_fits(os.path.join(self.savepath,'DMphases.fits'))
        res_phases = read_fits(os.path.join(self.savepath,'ResPhases.fits'))
        det_frames = read_fits(os.path.join(self.savepath,'DetectorFrames.fits'))
        dm_cmds =  read_fits(os.path.join(self.savepath,'DMcommands.fits'))
        return atmo_phases, dm_phases, res_phases, det_frames, dm_cmds











        