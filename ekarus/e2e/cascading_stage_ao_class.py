import xupy as xp
import numpy as np
from numpy.ma import masked_array

from ekarus.e2e.devices.alpao_deformable_mirror import ALPAODM
# from ekarus.e2e.pyramid_wfs import PyramidWFS
# from ekarus.e2e.detector import Detector
# from ekarus.e2e.slope_computer import SlopeComputer

# masked_array = xp.masked_array
import matplotlib.pyplot as plt
from ekarus.e2e.utils.image_utils import showZoomCenter, myimshow 

from ekarus.e2e.high_level_ao_class import HighLevelAO
from ekarus.e2e.utils.image_utils import reshape_on_mask #, get_masked_array



class CascadingAO(HighLevelAO):

    def __init__(self, tn: str):
        """The constructor"""

        super().__init__(tn)
        
        self.telemetry_keys = [
            "atmo_phases",
            "dm1_phases",
            "res1_phases",
            "ccd1_frames",
            "rec1_modes",
            "dm1_commands",
            "dm2_phases",
            "res2_phases",
            "ccd2_frames",
            "rec2_modes",
            "dm2_commands",
        ]

        self._initialize_devices()


    def _initialize_devices(self):
        """
        Initializes the devices used in the AO system.

        - WFS1, WFS2
        - Detector 1,2
        - DM1, DM2
        - Slope computer 1,2
        """

        self.pyr1, self.ccd1, self.sc1 = self._initialize_pyr_slope_computer('PYR1','CCD1','SLOPE.COMPUTER1')

        dm_pars = self._config.read_dm_pars('DM1')
        # self.dm1 = ALPAODM(dm_pars["Nacts"], Npix=self.pupilSizeInPixels, max_stroke=dm_pars['max_stroke_in_m'])
        self.dm1 = ALPAODM(dm_pars["Nacts"], pupil_mask = self.cmask.copy(), max_stroke=dm_pars['max_stroke_in_m'])

        self.pyr2, self.ccd2, self.sc2 = self._initialize_pyr_slope_computer('PYR2','CCD2','SLOPE.COMPUTER2')

        dm_pars = self._config.read_dm_pars('DM2')
        # self.dm2 = ALPAODM(dm_pars["Nacts"], Npix=self.pupilSizeInPixels, max_stroke=dm_pars['max_stroke_in_m'])
        self.dm2 = ALPAODM(dm_pars["Nacts"], pupil_mask = self.cmask.copy(), max_stroke=dm_pars['max_stroke_in_m'])


    def get_photons_per_subap(self, starMagnitude):
        collected_photons = self.get_photons_per_second(starMagnitude=starMagnitude)

        Nsubaps1 = xp.sum(1-self.sc1._subaperture_masks)
        ph1 = collected_photons * self.ccd1.quantum_efficiency * self.ccd1.beam_split_ratio
        ph_per_subap1 = ph1 / Nsubaps1 * self.sc1.dt

        Nsubaps2 = xp.sum(1-self.sc2._subaperture_masks)
        ph2 = collected_photons * self.ccd2.quantum_efficiency * self.ccd2.beam_split_ratio
        ph_per_subap2 = ph2 / Nsubaps2 * self.sc2.dt

        print(f'First stage: {ph_per_subap1:1.1f}e-/frame/subap, Second stage: {ph_per_subap2:1.1f}e-/frame/subap')

        return ph_per_subap1, ph_per_subap2

        

    def run_loop(self, lambdaInM:float, starMagnitude:float, save_prefix:str=None):
        """
        Main loop for the single stage AO system.

        Parameters
        ----------
        starMagnitude : float
            The magnitude of the star being observed.
        save_telemetry_prefix : str, optional
            String prefix to save telemetry data (default is None: telemetry is not saved).
        """
        m2rad = 2 * xp.pi / lambdaInM 

        # self.pyr1.set_modulation_angle(self.sc1.modulationAngleInLambdaOverD)
        # self.pyr2.set_modulation_angle(self.sc2.modulationAngleInLambdaOverD)

        dm1_cmd = xp.zeros(self.dm1.Nacts, dtype=self.dtype)
        self.dm1.set_position(dm1_cmd, absolute=True)
        self.dm1.surface -= self.dm1.surface  # make sure DM is flat

        dm2_cmd = xp.zeros(self.dm2.Nacts, dtype=self.dtype)
        self.dm2.set_position(dm2_cmd, absolute=True)
        self.dm2.surface -= self.dm2.surface  # make sure DM is flat

        # Define variables
        mask_len = int(xp.sum(1 - self.cmask))
        dm1_mask_len = int(xp.sum(1 - self.dm1.mask))
        dm1_cmds = xp.zeros([self.Nits, self.dm1.Nacts])

        dm2_mask_len = int(xp.sum(1 - self.dm2.mask))
        dm2_cmds = xp.zeros([self.Nits, self.dm2.Nacts])

        res2_phase_rad2 = xp.zeros(self.Nits)
        res1_phase_rad2 = xp.zeros(self.Nits)
        atmo_phase_rad2 = xp.zeros(self.Nits)

        if save_prefix is not None:
            input_phases = xp.zeros([self.Nits, mask_len])
            dm1_phases = xp.zeros([self.Nits, dm1_mask_len])
            residual1_phases = xp.zeros([self.Nits, mask_len])
            ccd1_images = xp.zeros([self.Nits, self.ccd1.detector_shape[0], self.ccd1.detector_shape[1]])
            rec1_modes = xp.zeros([self.Nits,self.sc1.Rec.shape[0]])
            dm2_phases = xp.zeros([self.Nits, dm2_mask_len])
            residual2_phases = xp.zeros([self.Nits, mask_len])
            ccd2_images = xp.zeros([self.Nits, self.ccd2.detector_shape[0], self.ccd2.detector_shape[1]])
            rec2_modes = xp.zeros([self.Nits,self.sc2.Rec.shape[0]])


        for i in range(self.Nits):
            print(f"\rIteration {i+1}/{self.Nits}", end="\r", flush=True)
            sim_time = self.dt * i

            atmo_phase = self.get_phasescreen_at_time(sim_time)
            input_phase = atmo_phase[~self.cmask]
            input_phase -= xp.mean(input_phase)  # remove piston

            if i >= self.sc1.delay:
                self.dm1.set_position(dm1_cmds[i - self.sc1.delay, :], absolute=True)

            residual1_phase = input_phase - self.dm1.get_surface()

            if i % int(self.sc1.dt/self.dt) == 0:
                dm1_cmds[i,:], modes1 = self.perform_loop_iteration(residual1_phase, dm1_cmd, self.sc1, 
                                                                    starMagnitude=starMagnitude, slaving=self.dm1.slaving)
            else:
                dm1_cmds[i,:] = dm1_cmds[i-1,:].copy()


            if i >= self.sc2.delay:
                self.dm2.set_position(dm2_cmds[i - self.sc2.delay, :], absolute=True)

            residual2_phase = residual1_phase - self.dm2.get_surface()

            if i % int(self.sc2.dt/self.dt) == 0:
                dm2_cmds[i,:], modes2 = self.perform_loop_iteration(residual2_phase, dm2_cmd, self.sc2, 
                                                                    starMagnitude=starMagnitude, slaving=self.dm2.slaving)
            else:
                dm2_cmds[i,:] = dm2_cmds[i-1,:].copy()

            res2_phase_rad2[i] = self.phase_rms(residual2_phase*m2rad)**2
            res1_phase_rad2[i] = self.phase_rms(residual1_phase*m2rad)**2
            atmo_phase_rad2[i] = self.phase_rms(input_phase*m2rad)**2

            if save_prefix is not None:  
                input_phases[i, :] = input_phase          
                residual1_phases[i, :] = residual1_phase
                dm1_phases[i, :] = self.dm1.surface
                ccd1_images[i, :, :] = self.ccd1.last_frame
                rec1_modes[i, :] = modes1

                residual2_phases[i, :] = residual2_phase
                dm2_phases[i, :] = self.dm2.surface
                ccd2_images[i, :, :] = self.ccd2.last_frame
                rec2_modes[i, :] = modes2
        

        if save_prefix is not None:
            print("Saving telemetry to .fits ...")
            dm1_mask_cube = xp.asnumpy(xp.stack([self.dm1.mask for _ in range(self.Nits)]))
            dm2_mask_cube = xp.asnumpy(xp.stack([self.dm2.mask for _ in range(self.Nits)]))
            mask_cube = xp.asnumpy(xp.stack([self.cmask for _ in range(self.Nits)]))
            input_phases = xp.stack([reshape_on_mask(input_phases[i, :], self.cmask) for i in range(self.Nits)])
            dm1_phases = xp.stack([reshape_on_mask(dm1_phases[i, :], self.dm1.mask)for i in range(self.Nits)])
            res1_phases = xp.stack([reshape_on_mask(residual1_phases[i, :], self.cmask)for i in range(self.Nits)])
            dm2_phases = xp.stack([reshape_on_mask(dm2_phases[i, :], self.dm2.mask)for i in range(self.Nits)])
            res2_phases = xp.stack([reshape_on_mask(residual2_phases[i, :], self.cmask)for i in range(self.Nits)])

            ma_input_phases = masked_array(xp.asnumpy(input_phases), mask=mask_cube)
            ma_dm1_phases = masked_array(xp.asnumpy(dm1_phases), mask=dm1_mask_cube)
            ma_res1_phases = masked_array(xp.asnumpy(res1_phases), mask=mask_cube)
            ma_dm2_phases = masked_array(xp.asnumpy(dm2_phases), mask=dm2_mask_cube)
            ma_res2_phases = masked_array(xp.asnumpy(res2_phases), mask=mask_cube)

            data_dict = {}
            for key, value in zip(
                self.telemetry_keys,
                [
                    ma_input_phases,
                    ma_dm1_phases,
                    ma_res1_phases,
                    ccd1_images,
                    rec1_modes,
                    dm1_cmds,
                    ma_dm2_phases,
                    ma_res2_phases,
                    ccd2_images,
                    rec2_modes,
                    dm2_cmds,
                ],
            ):
                data_dict[key] = value

            self.save_telemetry_data(data_dict, save_prefix)

        return res2_phase_rad2, res1_phase_rad2, atmo_phase_rad2
    

    def plot_iteration(self, lambdaRef, frame_id:int=-1, save_prefix:str=None):
        """
        Plots the telemetry data for a specific iteration/frame.
        
        Parameters
        ----------
        lambdaRef : float
            The reference wavelength in meters.
        frame_id : int, optional
            The frame/iteration index to plot, by default -1 (last frame).
        save_prefix : str, optional
            The prefix used when saving telemetry data, by default None.
        """
        if save_prefix is None:
            save_prefix = self.save_prefix

        ma_atmo_phases, _, res1_phases, det1_frames, rec1_modes, dm1_cmds, _, res2_phases, det2_frames, rec2_modes, dm2_cmds = self.load_telemetry_data(save_prefix=save_prefix)

        atmo_phase_in_rad = ma_atmo_phases[frame_id].data[~ma_atmo_phases[frame_id].mask]*(2*xp.pi/lambdaRef)
        res1_phase_in_rad = res1_phases[frame_id].data[~res1_phases[frame_id].mask]*(2*xp.pi/lambdaRef)
        res2_phase_in_rad = res2_phases[frame_id].data[~res2_phases[frame_id].mask]*(2*xp.pi/lambdaRef)

        in_err_rad2 = xp.asnumpy(self.phase_rms(atmo_phase_in_rad)**2)
        res1_err_rad2 = xp.asnumpy(self.phase_rms(res1_phase_in_rad)**2)
        res2_err_rad2 = xp.asnumpy(self.phase_rms(res2_phase_in_rad)**2)

        psf1, pixelSize = self._psf_from_frame(xp.array(res1_phases[frame_id]), lambdaRef)
        psf2, pixelSize = self._psf_from_frame(xp.array(res2_phases[frame_id]), lambdaRef)
        cmask = self.cmask.get() if xp.on_gpu else self.cmask.copy()

        plt.figure()#figsize=(9,9))
        plt.subplot(2,4,1)
        myimshow(masked_array(ma_atmo_phases[frame_id],cmask), \
        title=f'Atmosphere phase [m]\nSR = {xp.exp(-in_err_rad2):1.3f} @{lambdaRef*1e+9:1.0f}[nm]',\
        cmap='RdBu',shrink=0.8)
        plt.axis('off')
        plt.subplot(2,4,5)
        myimshow(masked_array(res1_phases[frame_id],cmask), \
        title=f'Residual phase [m] after DM1\nSR = {xp.exp(-res1_err_rad2):1.3f} @{lambdaRef*1e+9:1.0f}[nm]',\
        cmap='RdBu',shrink=0.8)
        plt.axis('off')
        plt.subplot(2,4,3)
        showZoomCenter(psf1, pixelSize, shrink=0.8,
        title = f'Corrected PSF 1\nSR = {xp.exp(-res1_err_rad2):1.3f} @{lambdaRef*1e+9:1.0f}[nm]'
            , cmap='inferno', xlabel=r'$\lambda/D$'
            , ylabel=r'$\lambda/D$') 
        plt.subplot(2,4,2)
        myimshow(det1_frames[frame_id], title = 'Detector 1 frame', shrink=0.8)
        plt.subplot(2,4,4)
        self.dm1.plot_position(dm1_cmds[frame_id])
        plt.title('DM1 command [m]')
        plt.axis('off')
        plt.subplot(2,4,7)
        showZoomCenter(psf2, pixelSize, shrink=0.8,
        title = f'Corrected PSF 2\nSR = {xp.exp(-res2_err_rad2):1.3f} @{lambdaRef*1e+9:1.0f}[nm]'
            , cmap='inferno', xlabel=r'$\lambda/D$'
            , ylabel=r'$\lambda/D$') 
        plt.subplot(2,4,6)
        myimshow(det2_frames[frame_id], title = 'Detector 2 frame', shrink=0.8)
        plt.subplot(2,4,8)
        self.dm2.plot_position(dm2_cmds[frame_id])
        plt.title('DM2 command [m]')
        plt.axis('off')

        res1_phase = xp.asarray(res1_phases[frame_id].data[~res1_phases[frame_id].mask])
        KL1_modes = (xp.linalg.pinv(self.KL1)).T @ res1_phase
        res2_phase = xp.asarray(res2_phases[frame_id].data[~res2_phases[frame_id].mask])
        KL2_modes = (xp.linalg.pinv(self.KL1)).T @ res2_phase

        plt.figure()
        plt.plot(xp.asnumpy(xp.abs(KL1_modes))*1e+9,label='after 1st stage')
        plt.plot(xp.asnumpy(xp.abs(KL2_modes))*1e+9,label='after 2nd stage')
        plt.legend()
        plt.xlabel('mode index')
        plt.ylabel('mode RMS amp [nm]')
        plt.title('KL modes')
        plt.grid()
        plt.xscale('log')
        plt.yscale('log')


    def plot_contrast(self, lambdaRef, frame_id:int=-1, save_prefix:str='',oversampling:int=12):
        """
        Plots the telemetry data for a specific iteration/frame.
        
        Parameters
        ----------
        lambdaRef : float
            The reference wavelength in meters.
        frame_id : int, optional
            The frame/iteration index to plot, by default -1 (last frame).
        save_prefix : str, optional
            The prefix used when saving telemetry data, by default None.
        """
        if save_prefix is None:
            save_prefix = self.save_prefix

        _, _, res1_phases, _, _, _, _, res2_phases, _, _, _ = self.load_telemetry_data(save_prefix=save_prefix)

        res1_phase_in_rad = res1_phases[frame_id].data[~res1_phases[frame_id].mask]*(2*xp.pi/lambdaRef)
        psf1,psd1,pix_dist=self.get_contrast(residual_phase_in_rad=res1_phase_in_rad,oversampling=oversampling)
        res2_phase_in_rad = res2_phases[frame_id].data[~res2_phases[frame_id].mask]*(2*xp.pi/lambdaRef)
        psf2,psd2,pix_dist=self.get_contrast(residual_phase_in_rad=res2_phase_in_rad,oversampling=oversampling)


        # plt.figure(figsize=(8,4))
        # plt.subplot(1,2,1)
        # showZoomCenter(xp.asnumpy(psf1), 1/oversampling, shrink=0.7,
        # title = f'Coronographic PSF after DM1', cmap='inferno', xlabel=r'$\lambda/D$', ylabel=r'$\lambda/D$') 
        # plt.subplot(1,2,2)
        # showZoomCenter(xp.asnumpy(psf2), 1/oversampling, shrink=0.7,
        # title = f'Coronographic PSF after DM2', cmap='inferno', xlabel=r'$\lambda/D$', ylabel=r'$\lambda/D$') 

        plt.figure()
        plt.plot(xp.asnumpy(pix_dist),xp.asnumpy(psd1),label='First stage')
        plt.plot(xp.asnumpy(pix_dist),xp.asnumpy(psd2),label='Second stage')
        plt.legend()
        plt.grid()
        plt.yscale('log')
        plt.xlabel(r'$\lambda/D$')
        plt.xlim([0,30])
        plt.title('Contrast')

        return psd1, psd2, pix_dist

    # def __init__(self, tn, xp=np):

    #     super().__init__(tn, xp)

    #     self._initialize_devices()


    # def _initialize_devices(self):
    #     apex_angle, oversampling,modulationAngleInLambdaOverD = self._config.read_sensor_pars('WFS1')
    #     detector_shape, RON, quantum_efficiency = self._config.read_detector_pars('DETECTOR1')
    #     Nacts = self._config.read_dm_pars('DM1')

    #     self.pyr1 = PyramidWFS(apex_angle, oversampling, xp=self._xp)
    #     self.pyr1.set_modulation_angle(modulationAngleInLambdaOverD)
    #     self.ccd1 = Detector(detector_shape=detector_shape, RON=RON, quantum_efficiency=quantum_efficiency, xp=self._xp)
    #     self.dm1 = ALPAODM(Nacts, Npix=self.pupilSizeInPixels, xp=self._xp)
    #     self.slope_computer1 = SlopeComputer(self.pyr1, self.ccd1, xp=self._xp)

    #     apex_angle, oversampling, modulationAngleInLambdaOverD = self._config.read_sensor_pars('WFS2')
    #     detector_shape, RON, quantum_efficiency = self._config.read_detector_pars('DETECTOR2')
    #     Nacts = self._config.read_dm_pars('DM2')

    #     self.pyr2 = PyramidWFS(apex_angle, oversampling, xp=self._xp)
    #     self.pyr2.set_modulation_angle(modulationAngleInLambdaOverD)
    #     self.ccd2 = Detector(detector_shape=detector_shape, RON=RON, quantum_efficiency=quantum_efficiency, xp=self._xp)
    #     self.dm2 = ALPAODM(Nacts, Npix=self.pupilSizeInPixels, xp=self._xp)
    #     self.slope_computer2 = SlopeComputer(self.pyr2, self.ccd2, xp=self._xp)

    
    # def run_loop(self, lambdaInM, starMagnitude, Rec1, Rec2, m2c_dm1, m2c_dm2, save_telemetry:bool=False):
    #     electric_field_amp = 1-self.cmask

    #     # modal_gains = self._xp.zeros(Rec.shape[0])
    #     # modal_gains[:self.nModes] = 1

    #     lambdaOverD = lambdaInM/self.pupilSizeInM
    #     Nphotons = self.get_photons_per_second(starMagnitude) * self.dt

    #     # Define variables
    #     mask_len = int(self._xp.sum(1-self.cmask))
    #     dm1_cmd = self._xp.zeros(self.dm1.Nacts, dtype=self.dtype)
    #     self.dm1.set_position(dm1_cmd, absolute=True)
    #     dm1_cmds = self._xp.zeros([self.Nits,self.dm1.Nacts])

    #     dm2_cmd = self._xp.zeros(self.dm2.Nacts, dtype=self.dtype)
    #     self.dm2.set_position(dm2_cmd, absolute=True)
    #     dm2_cmds = self._xp.zeros([self.Nits,self.dm2.Nacts])

    #     if save_telemetry:
    #         dm1_phases = self._xp.zeros([self.Nits,mask_len])
    #         dm2_phases = self._xp.zeros([self.Nits,mask_len])
    #         after_dm1_phases = self._xp.zeros([self.Nits,mask_len])
    #         after_dm2_phases = self._xp.zeros([self.Nits,mask_len])
    #         input_phases = self._xp.zeros([self.Nits,mask_len])
    #         detector1_images = self._xp.zeros([self.Nits,self.ccd1.detector_shape[0],self.ccd1.detector_shape[1]])
    #         detector2_images = self._xp.zeros([self.Nits,self.ccd2.detector_shape[0],self.ccd2.detector_shape[1]])

    #     for i in range(self.Nits):
    #         print(f'\rIteration {i+1}/{self.Nits}', end='')
    #         sim_time = self.dt*i

    #         atmo_phase = self.get_phasescreen_at_time(sim_time)
    #         input_phase = atmo_phase[~self.cmask]
    #         input_phase -= self._xp.mean(input_phase) # remove piston

    #         if i >= self.delaySteps:
    #             self.dm1.set_position(dm1_cmds[i-self.delaySteps,:], absolute=True)
    #         after_dm1_phase = input_phase - self.dm1.surface
    #         delta_phase_in_rad = reshape_on_mask(after_dm1_phase*(2*self._xp.pi)/lambdaInM, self.cmask, xp=self._xp)
    #         input_field = electric_field_amp * self._xp.exp(1j*delta_phase_in_rad)
    #         slopes = self.slope_computer1.compute_slopes(input_field, lambdaOverD, Nphotons)
    #         modes = Rec1 @ slopes
    #         # modes *= modal_gains
    #         cmd = m2c_dm1 @ modes
    #         dm1_cmd += cmd*self.integratorGain
    #         dm1_cmds[i,:] = dm1_cmd*lambdaInM/(2*self._xp.pi) # convert to meters

    #         if i >= self.delaySteps:
    #             self.dm2.set_position(dm2_cmds[i-self.delaySteps,:], absolute=True)
    #         after_dm2_phase = after_dm1_phase - self.dm2.surface
    #         delta_phase_in_rad = reshape_on_mask(after_dm2_phase*(2*self._xp.pi)/lambdaInM, self.cmask, xp=self._xp)
    #         input_field = electric_field_amp * self._xp.exp(1j*delta_phase_in_rad)
    #         slopes = self.slope_computer2.compute_slopes(input_field, lambdaOverD, Nphotons)
    #         modes = Rec2 @ slopes
    #         # modes *= modal_gains
    #         cmd = m2c_dm2 @ modes
    #         dm2_cmd += cmd*self.integratorGain
    #         dm2_cmds[i,:] = dm2_cmd*lambdaInM/(2*self._xp.pi) # convert to meters

    #         after_dm2_phases[i,:] = after_dm2_phase
    #         after_dm1_phases[i,:] = after_dm1_phase
    #         input_phases[i,:] = input_phase
    #         if save_telemetry:
    #             dm1_phases[i,:] = self.dm1.surface
    #             detector1_images[i,:,:] = self.ccd1.last_frame                
    #             dm2_phases[i,:] = self.dm2.surface
    #             detector2_images[i,:,:] = self.ccd2.last_frame
    #     print('')

    #     afterDM2Rad2 = self._xp.std(after_dm2_phases*(2*self._xp.pi)/lambdaInM,axis=-1)**2
    #     afterDM1Rad2 = self._xp.std(after_dm1_phases*(2*self._xp.pi)/lambdaInM,axis=-1)**2
    #     inputErrRad2 = self._xp.std(input_phases*(2*self._xp.pi)/lambdaInM,axis=-1)**2 

    #     if save_telemetry:
    #         print('Saving telemetry to .fits ...')
    #         ma_input_phases = np.stack([reshape_on_mask(input_phases[i,:], self.cmask, self._xp) for i in range(self.Nits)])
    #         ma_dm1_phases = np.stack([reshape_on_mask(dm1_phases[i,:], self.cmask, self._xp) for i in range(self.Nits)])
    #         ma_res1_phases = np.stack([reshape_on_mask(after_dm1_phases[i,:], self.cmask, self._xp) for i in range(self.Nits)])
    #         ma_dm2_phases = np.stack([reshape_on_mask(dm2_phases[i,:], self.cmask, self._xp) for i in range(self.Nits)])
    #         ma_res2_phases = np.stack([reshape_on_mask(after_dm2_phases[i,:], self.cmask, self._xp) for i in range(self.Nits)])

    #         data_dict = {'DM1phases': ma_dm1_phases, 'Res1Phases': ma_res1_phases, \
    #                      'Detector1Frames': detector1_images, 'DM1commands': dm1_cmds, \
    #                      'DM2phases': ma_dm2_phases, 'Res2Phases': ma_res2_phases, \
    #                      'Detector2Frames': detector2_images, 'DM2commands': dm2_cmds, \
    #                      'AtmoPhases': ma_input_phases}
    #         self.save_telemetry_data(data_dict)

    #     return afterDM2Rad2, afterDM1Rad2, inputErrRad2

    # def load_telemetry_data(self):
    #     atmo_phases = read_fits(os.path.join(self.savepath,'AtmoPhases.fits'))
    #     dm1_phases = read_fits(os.path.join(self.savepath,'DM1phases.fits'))
    #     res1_phases = read_fits(os.path.join(self.savepath,'Res1Phases.fits'))
    #     det1_frames = read_fits(os.path.join(self.savepath,'Detector1Frames.fits'))
    #     dm1_cmds =  read_fits(os.path.join(self.savepath,'DM1commands.fits'))
    #     dm2_phases = read_fits(os.path.join(self.savepath,'DM2phases.fits'))
    #     res2_phases = read_fits(os.path.join(self.savepath,'Res2Phases.fits'))
    #     det2_frames = read_fits(os.path.join(self.savepath,'Detector2Frames.fits'))
    #     dm2_cmds =  read_fits(os.path.join(self.savepath,'DM2commands.fits'))
    #     return atmo_phases, dm1_phases, res1_phases, det1_frames, dm1_cmds, dm2_phases, res2_phases, det2_frames, dm2_cmds






        