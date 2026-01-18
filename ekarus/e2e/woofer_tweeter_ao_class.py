import xupy as xp
from numpy.ma import masked_array

from ekarus.e2e.devices.deformable_secondary_mirror import DSM

import matplotlib.pyplot as plt
from ekarus.e2e.utils.image_utils import showZoomCenter, myimshow 

from ekarus.e2e.high_level_ao_class import HighLevelAO
from ekarus.e2e.utils.image_utils import reshape_on_mask #, get_masked_array



class WooferTweeterAO(HighLevelAO):

    def __init__(self, tn: str):
        """The constructor"""

        super().__init__(tn)
        
        self.telemetry_keys = [
            "atmo_phases",
            "dm_phases",
            "res_phases",
            "ccd1_frames",
            "rec1_modes",
            "ccd2_frames",
            "rec2_modes",
            "dm_commands",
        ]

        self._initialize_devices()


    def _initialize_devices(self):
        """
        Initializes the devices used in the AO system.

        - WFS for woofer and tweeter loops
        - Detector for woofer and tweeter loops
        - DM1, DM2 for woofer and tweeter loops, respectively
        - Slope computer for woofer and tweeter loops
        """

        self.wfs1, self.ccd1, self.sc1 = self._initialize_slope_computer('WFS1','CCD1','SLOPE.COMPUTER1')
        self.wfs2, self.ccd2, self.sc2 = self._initialize_slope_computer('WFS2','CCD2','SLOPE.COMPUTER2')
        dm_pars = self._config.read_dm_pars('DM')
        self.dm = DSM(dm_pars["Nacts"], pupil_mask = self.cmask.copy(), geom=dm_pars['geom'], max_stroke=dm_pars['max_stroke_in_m'])

  
    def get_photons_per_subap(self, starMagnitude):
        collected_photons = self.get_photons_per_second(starMagnitude=starMagnitude)
        Nsubaps1 = xp.sum(1-self.sc1._roi_masks)
        ph1 = collected_photons * self.ccd1.quantum_efficiency * self.ccd1.beam_split_ratio
        ph_per_subap1 = ph1 / Nsubaps1 * self.sc1.dt
        Nsubaps2 = xp.sum(1-self.sc2._roi_masks)
        ph2 = collected_photons * self.ccd2.quantum_efficiency * self.ccd2.beam_split_ratio
        ph_per_subap2 = ph2 / Nsubaps2 * self.sc2.dt
        print(f'First WFS: {ph_per_subap1:1.1f}e-/frame/pix, second WFS: {ph_per_subap2:1.1f}e-/frame/pix')
        return ph_per_subap1, ph_per_subap2
    

    def run_loop(self, lambdaInM:float, starMagnitude:float, 
                enable_tweeter:bool=True, save_prefix:str=None):
        """
        Main loop for the single stage AO system.

        Parameters
        ----------
        starMagnitude : float
            The magnitude of the star being observed.
        save_telemetry_prefix : str, optional
            String prefix to save telemetry data (default is None: telemetry is not saved).
        enable_tweeter : bool, optional
            Wheter to enable the tweeter stage or not. Defaults to True (enabled)
        """
        m2rad = 2 * xp.pi / lambdaInM 

        IF = self.dm.IFF.copy()
        dm_surf = xp.zeros(IF.shape[0])

        # Define variables
        mask_len = int(xp.sum(1 - self.cmask))
        int_cmds = xp.zeros([self.Nits,self.dm.Nacts], dtype=self.dtype)
        dm_cmds = xp.zeros([self.Nits,self.dm.Nacts], dtype=self.dtype)
        dm1_cmds = xp.zeros([self.Nits,self.dm.Nacts], dtype=self.dtype)
        dm2_cmds = xp.zeros([self.Nits,self.dm.Nacts], dtype=self.dtype)

        res_phase_rad2 = xp.zeros(self.Nits)
        atmo_phase_rad2 = xp.zeros(self.Nits)

        if save_prefix is not None:
            input_phases = xp.zeros([self.Nits, mask_len], dtype=self.dtype)
            residual_phases = xp.zeros([self.Nits, mask_len], dtype=self.dtype)
            ccd1_images = xp.zeros([self.Nits, self.ccd1.detector_shape[0], self.ccd1.detector_shape[1]], dtype=self.dtype)
            rec1_modes = xp.zeros([self.Nits,self.sc1.Rec.shape[0]], dtype=self.dtype)
            ccd2_images = xp.zeros([self.Nits, self.ccd2.detector_shape[0], self.ccd2.detector_shape[1]], dtype=self.dtype)
            rec2_modes = xp.zeros([self.Nits,self.sc2.Rec.shape[0]], dtype=self.dtype)

        delay = int(xp.minimum(self.sc2.delay,self.sc1.delay))
        for i in range(self.Nits):
            print(f"\rIteration {i+1}/{self.Nits}", end="\r", flush=True)
            sim_time = self.dt * i

            atmo_phase = self.get_phasescreen_at_time(sim_time)
            input_phase = atmo_phase[~self.cmask]
            input_phase -= xp.mean(input_phase)  # remove piston

            if i >= self.sc1.delay:
                dm_surf = IF @ dm1_cmds[i - self.sc1.delay, :]

            if i >= self.sc2.delay:
                dm_surf += IF @ dm2_cmds[i - self.sc2.delay, :]

            residual_phase = input_phase - dm_surf[self.dm.visible_pix_ids]

            if i % int(self.sc1.dt/self.dt) == 0:
                int_cmds[i,:], modes1 = self.perform_loop_iteration(residual_phase, self.sc1, 
                                                                    starMagnitude=starMagnitude, 
                                                                    slaving=self.dm.slaving)
                dm1_cmds[i,:] = self.sc1.iir_filter(int_cmds[:i+1,:],dm1_cmds[:i+1,:])
            else:
                dm1_cmds[i,:] = dm1_cmds[i-1,:].copy()

            if i % int(self.sc2.dt/self.dt) == 0:
                int_cmds[i,:], modes2 = self.perform_loop_iteration(residual_phase, self.sc2, 
                                                                    starMagnitude=starMagnitude, 
                                                                    slaving=self.dm.slaving)
                dm2_cmds[i,:] = self.sc2.iir_filter(int_cmds[:i+1,:], dm2_cmds[:i+1,:])
            else:
                dm2_cmds[i,:] = dm2_cmds[i-1,:].copy()
            
            dm_cmds[i,:] = dm1_cmds[i,:]+dm2_cmds[i,:]

            res_phase_rad2[i] = self.phase_rms(residual_phase*m2rad)**2
            atmo_phase_rad2[i] = self.phase_rms(input_phase*m2rad)**2

            if save_prefix is not None:  
                input_phases[i, :] = input_phase          
                residual_phases[i, :] = residual_phase
                ccd1_images[i, :, :] = self.ccd1.last_frame
                rec1_modes[i, :] = modes1

                ccd2_images[i, :, :] = self.ccd2.last_frame
                rec2_modes[i, :] = modes2
        

        if save_prefix is not None:
            print("Saving telemetry to .fits ...")
            dm_mask_cube = xp.asnumpy(xp.stack([self.dm.mask for _ in range(self.Nits)]))
            mask_cube = xp.asnumpy(xp.stack([self.cmask for _ in range(self.Nits)]))
            input_phases = xp.stack([reshape_on_mask(input_phases[i, :], self.cmask) for i in range(self.Nits)])
            dm_phases = xp.stack([reshape_on_mask(IF @ dm_cmds[i, :], self.dm.mask)for i in range(self.Nits)])
            res_phases = xp.stack([reshape_on_mask(residual_phases[i, :], self.cmask)for i in range(self.Nits)])

            ma_input_phases = masked_array(xp.asnumpy(input_phases), mask=mask_cube)
            ma_dm_phases = masked_array(xp.asnumpy(dm_phases), mask=dm_mask_cube)
            ma_res_phases = masked_array(xp.asnumpy(res_phases), mask=mask_cube)

            data_dict = {}
            for key, value in zip(
                self.telemetry_keys,
                [
                    ma_input_phases,
                    ma_dm_phases,
                    ma_res_phases,
                    ccd1_images,
                    rec1_modes,
                    ccd2_images,
                    rec2_modes,
                    dm_cmds,
                ],
            ):
                data_dict[key] = value

            self.save_telemetry_data(data_dict, save_prefix)

        return res_phase_rad2, atmo_phase_rad2
    
    
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

        ma_atmo_phases, _, res_phases, det1_frames, rec1_modes, det2_frames, rec2_modes, dm_cmds = self.load_telemetry_data(save_prefix=save_prefix)

        atmo_phase_rad = ma_atmo_phases[frame_id].data[~ma_atmo_phases[frame_id].mask]*(2*xp.pi/lambdaRef)
        res_phase_rad = res_phases[frame_id].data[~res_phases[frame_id].mask]*(2*xp.pi/lambdaRef)

        atmo_err_rad2 = xp.asnumpy(self.phase_rms(atmo_phase_rad)**2)
        res_err_rad2 = xp.asnumpy(self.phase_rms(res_phase_rad)**2)

        psf, pixelSize = self._psf_from_frame(xp.array(res_phases[frame_id]), lambdaRef)
        cmask = self.cmask.get() if xp.on_gpu else self.cmask.copy()

        plt.figure()#figsize=(9,9))
        plt.subplot(2,4,1)
        myimshow(masked_array(ma_atmo_phases[frame_id],cmask), \
        title=f'Atmosphere phase [m]\nSR = {xp.exp(-atmo_err_rad2):1.3f} @ {lambdaRef*1e+9:1.0f}[nm]',\
        cmap='RdBu',shrink=0.8)
        plt.axis('off')
        plt.subplot(2,4,2)
        myimshow(det1_frames[frame_id], title = 'LO detector frame', shrink=0.8, cmap='Blues')
        plt.subplot(2,4,3)
        showZoomCenter(psf/xp.max(psf), pixelSize, shrink=0.8,
        title = f'Corrected PSF\nSR = {xp.exp(-res_err_rad2):1.3f} @ {lambdaRef*1e+9:1.0f}[nm]'
            , cmap='inferno', xlabel=r'$\lambda/D$', ylabel=r'$\lambda/D$', vmin=-10) 
        plt.subplot(2,4,4)
        self.dm.plot_position(dm_cmds[frame_id])
        plt.title('DM command [m]')
        plt.axis('off')
        plt.subplot(2,4,6)
        myimshow(det2_frames[frame_id], title = 'HO detector frame', shrink=0.8, cmap='Blues')

        N = int(xp.maximum(self.Nits-100,self.Nits/2))
        atmo_modes = xp.zeros([N,self.KL.shape[0]])
        res2_modes = xp.zeros([N,self.sc1.nModes+self.sc2.nModes])
        phase2modes = xp.linalg.pinv(self.KL) #xp.linalg.pinv(self.KL.T)
        for frame in range(N):
            mask = ma_atmo_phases[-N+frame].mask.copy()
            atmo_phase = xp.asarray(ma_atmo_phases[-N+frame].data[~mask])
            atmo_modes[frame,:] = xp.dot(atmo_phase,phase2modes) #phase2modes @ atmo_phase
            # res1_phase = xp.asarray(res_woofer_phases[-N+frame].data[~mask])
            # res1_modes[frame,:] = xp.dot(res1_phase,phase2modes[:,:self.sc1.nModes]) #pphase2modes @ res1_phase
            res2_phase = xp.asarray(res_phases[-N+frame].data[~mask])
            res2_modes[frame,:] = xp.dot(res2_phase,phase2modes[:,:(self.sc2.nModes+self.sc1.nModes)]) #p phase2modes @ res2_phase
        atmo_mode_rms = xp.sqrt(xp.mean(atmo_modes**2,axis=0))
        res2_mode_rms = xp.sqrt(xp.mean(res2_modes**2,axis=0))
        rec1_modes_rms = xp.sqrt(xp.mean(rec1_modes[-N-1:-1,:]**2,axis=0))
        rec2_modes_rms = xp.sqrt(xp.mean(rec2_modes[-N-1:-1,:]**2,axis=0))

        plt.figure()
        plt.plot(xp.asnumpy(atmo_mode_rms)*1e+9,label='turbulence')
        plt.plot(xp.asnumpy(rec1_modes_rms)*1e+9,'--',label='LOWFS modes')
        plt.plot(xp.asnumpy(xp.arange(self.sc2.nModes)+self.sc1.nModes),
                 xp.asnumpy(rec2_modes_rms)*1e+9,'--',label='HOWFS modes')
        plt.plot(xp.asnumpy(res2_mode_rms)*1e+9,label='residual')
        plt.legend()
        plt.xlabel('mode index')
        plt.ylabel('mode RMS amp [nm]')
        plt.title('KL modes')
        plt.grid()
        plt.xscale('log')
        plt.yscale('log')




    def plot_contrast(self, lambdaRef, frame_ids:list=None, save_prefix:str='',oversampling:int=8):
        """
        Plots the telemetry data for a specific iteration/frame.
        
        Parameters
        ----------
        lambdaRef : float
            The reference wavelength in meters.
        frame_id : list, optional
            The frames over which the std is computed, by default, the bottom half.
        save_prefix : str, optional
            The prefix used when saving telemetry data, by default None.
        """
        if save_prefix is None:
            save_prefix = self.save_prefix

        if frame_ids is None:
            frame_ids = xp.arange(-self.Nits//2,0)
        else:
            frame_ids = xp.array(frame_ids)
        frame_ids = xp.asnumpy(frame_ids)

        res_phases, = self.load_telemetry_data(save_prefix=save_prefix,data_keys=['res_phases'])

        N = len(frame_ids)
        res_phases_in_rad = xp.zeros([N,int(xp.sum(1-self.cmask))])
        for j in range(N):
            res_phases_in_rad[j] = xp.asarray(res_phases[frame_ids[j]].data[~res_phases[frame_ids[j]].mask]*(2*xp.pi/lambdaRef))
        _,rms_psf,pix_dist=self.get_contrast(res_phases_in_rad,oversampling=oversampling)

        plt.figure()
        plt.plot(xp.asnumpy(pix_dist),xp.asnumpy(rms_psf))
        plt.grid()
        plt.yscale('log')
        plt.xlabel(r'$\lambda/D$')
        plt.xlim([0,30])
        plt.title(f'Contrast @ {lambdaRef*1e+9:1.0f} nm\n(assuming a perfect coronograph)')

        return rms_psf, pix_dist
    

    def plot_ristretto_contrast(self, lambdaRef, frame_ids:list=None, save_prefix:str='',oversampling:int=10):
        """
        Plots the telemetry data for a specific iteration/frame.
        
        Parameters
        ----------
        lambdaRef : float
            The reference wavelength in meters.
        frame_id : list, optional
            The frames over which the std is computed, by default, the bottom half.
        save_prefix : str, optional
            The prefix used when saving telemetry data, by default None.
        """
        if save_prefix is None:
            save_prefix = self.save_prefix

        if frame_ids is None:
            frame_ids = xp.arange(self.Nits)
        else:
            frame_ids = xp.array(frame_ids)
        frame_ids = xp.asnumpy(frame_ids)

        res2_phases, = self.load_telemetry_data(save_prefix=save_prefix, data_keys=['res_phases'])

        N = len(frame_ids)
        # res1_phases_in_rad = xp.zeros([N,int(xp.sum(1-self.cmask))])
        res2_phases_in_rad = xp.zeros([N,int(xp.sum(1-self.cmask))])
        for j in range(N):
            res2_phases_in_rad[j] = xp.asarray(res2_phases[frame_ids[j]].data[~res2_phases[frame_ids[j]].mask]*(2*xp.pi/lambdaRef))
            # res1_phases_in_rad[j] = xp.asarray(res1_phases[frame_ids[j]].data[~res1_phases[frame_ids[j]].mask]*(2*xp.pi/lambdaRef))

        # smf1_couplings=self.get_ristretto_contrast(res1_phases_in_rad,lambdaInM=lambdaRef,oversampling=oversampling)
        smf2_couplings=self.get_ristretto_contrast(res2_phases_in_rad,lambdaInM=lambdaRef,oversampling=oversampling,smfRadiusInMAS=19)

        # plt.figure()
        # plt.plot(xp.asnumpy(smf1_couplings.T))
        # plt.grid()
        # plt.yscale('log')        
        plt.figure()
        plt.plot(xp.asnumpy(smf2_couplings[1:].T),'--')
        plt.grid()
        plt.yscale('log')
        plt.xlabel('Iteration')
        plt.ylabel('Normalized flux')
        plt.title('Post-coronographic star flux\nin the 6 side spaxels\n(normalized to non-coronographic star flux)')

        return smf2_couplings

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
    #         ma_res_out_phases = np.stack([reshape_on_mask(after_dm1_phases[i,:], self.cmask, self._xp) for i in range(self.Nits)])
    #         ma_dm2_phases = np.stack([reshape_on_mask(dm2_phases[i,:], self.cmask, self._xp) for i in range(self.Nits)])
    #         ma_res_in_phases = np.stack([reshape_on_mask(after_dm2_phases[i,:], self.cmask, self._xp) for i in range(self.Nits)])

    #         data_dict = {'DM1phases': ma_dm1_phases, 'res_outPhases': ma_res_out_phases, \
    #                      'Detector1Frames': detector1_images, 'DM1commands': dm1_cmds, \
    #                      'DM2phases': ma_dm2_phases, 'res_inPhases': ma_res_in_phases, \
    #                      'Detector2Frames': detector2_images, 'DM2commands': dm2_cmds, \
    #                      'AtmoPhases': ma_input_phases}
    #         self.save_telemetry_data(data_dict)

    #     return afterDM2Rad2, afterDM1Rad2, inputErrRad2

    # def load_telemetry_data(self):
    #     atmo_phases = read_fits(os.path.join(self.savepath,'AtmoPhases.fits'))
    #     dm1_phases = read_fits(os.path.join(self.savepath,'DM1phases.fits'))
    #     res_out_phases = read_fits(os.path.join(self.savepath,'res_outPhases.fits'))
    #     det_out_frames = read_fits(os.path.join(self.savepath,'Detector1Frames.fits'))
    #     dm1_cmds =  read_fits(os.path.join(self.savepath,'DM1commands.fits'))
    #     dm2_phases = read_fits(os.path.join(self.savepath,'DM2phases.fits'))
    #     res_in_phases = read_fits(os.path.join(self.savepath,'res_inPhases.fits'))
    #     det_in_frames = read_fits(os.path.join(self.savepath,'Detector2Frames.fits'))
    #     dm2_cmds =  read_fits(os.path.join(self.savepath,'DM2commands.fits'))
    #     return atmo_phases, dm1_phases, res_out_phases, det_out_frames, dm1_cmds, dm2_phases, res_in_phases, det_in_frames, dm2_cmds






        