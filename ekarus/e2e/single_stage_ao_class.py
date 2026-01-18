import xupy as xp
import numpy as np
from numpy.ma import masked_array

from ekarus.e2e.devices.deformable_secondary_mirror import DSM
# from ekarus.e2e.devices.pyramid_wfs import PyramidWFS
# from ekarus.e2e.devices.detector import Detector
# from ekarus.e2e.devices.slope_computer import SlopeComputer

# masked_array = xp.masked_array
import matplotlib.pyplot as plt
from ekarus.e2e.utils.image_utils import showZoomCenter, myimshow 

from ekarus.e2e.high_level_ao_class import HighLevelAO
from ekarus.e2e.utils.image_utils import reshape_on_mask #, get_masked_array


class SingleStageAO(HighLevelAO):

    def __init__(self, tn: str):
        """The constructor"""

        super().__init__(tn)        
        
        self.telemetry_keys = [
            "atmo_phases",
            "dm_phases",
            "residual_phases",
            "ccd_frames",
            "reconstructor_modes",
            "dm_commands",
            "rms2_residual",
        ]

        self._initialize_devices()


    def _initialize_devices(self):
        """
        Initializes the devices used in the AO system.

        - WFS
        - Detector
        - DM
        - Slope computer
        """

        print('Initializing devices ...')

        try:
            self.pyr, self.ccd, self.sc = self._initialize_pyr_slope_computer('PYR','CCD','SLOPE.COMPUTER')
        except KeyError:
            self.pyr, self.ccd, self.sc = self._initialize_pyr_slope_computer('PYR1','CCD1','SLOPE.COMPUTER1')

        try:
            dm_pars = self._config.read_dm_pars()
        except KeyError:
            dm_pars = self._config.read_dm_pars('DM1')
        self.dm = DSM(dm_pars["Nacts"], pupil_mask = self.cmask.copy(), geom=dm_pars['geom'], max_stroke=dm_pars['max_stroke_in_m'])
    

    def run_loop(self, lambdaInM:float, starMagnitude:float, save_prefix:str=None, ncpa_cmd=None):
        """
        Main loop for the single stage AO system.

        Parameters
        ----------
        starMagnitude : float
            The magnitude of the star being observed.
        Rec : array_like
            The reconstruction matrix.
        m2c : array_like
            The modal-to-command matrix.
        save_telemetry_prefix : str, optional
            String prefix to save telemetry data (default is None: telemetry is not saved).
        """
        m2rad = 2 * xp.pi / lambdaInM

        IF = self.dm.IFF.copy()
        dm_surf = xp.zeros(IF.shape[0])

        # Define variables
        mask_len = int(xp.sum(1 - self.cmask))
        int_cmds = xp.zeros([self.Nits, self.dm.Nacts],dtype=self.dtype)
        dm_cmds = xp.zeros([self.Nits, self.dm.Nacts],dtype=self.dtype)
        if ncpa_cmd is not None:
            dm_cmds = xp.repeat(ncpa_cmd.reshape([1,self.dm.Nacts]),self.Nits,axis=0)

        res_phase_rad2 = xp.zeros(self.Nits)
        atmo_phase_rad2 = xp.zeros(self.Nits)

        if self.tt_offload is not None:
            tt_coeffs = xp.zeros([self.Nits,2],dtype=self.dtype)
            tt_offload_dt = 1/self.tt_offload['frequencyInHz']
            tt_gain = self.tt_offload['gain']
            tt_amps = xp.zeros(2)
            c2tt = xp.linalg.pinv(self.dm.act_coords.T)
            tt2surf = IF @ self.dm.act_coords.T
            

        if save_prefix is not None:
            residual_phases = xp.zeros([self.Nits, mask_len],dtype=self.dtype)
            input_phases = xp.zeros([self.Nits, mask_len],dtype=self.dtype)
            detector_images = xp.zeros([self.Nits, self.ccd.detector_shape[0], self.ccd.detector_shape[1]],dtype=self.dtype)
            rec_modes = xp.zeros([self.Nits,self.sc.Rec.shape[0]],dtype=self.dtype)

        for i in range(self.Nits):
            print(f"\rIteration {i+1}/{self.Nits}", end="\r", flush=True)
            sim_time = self.dt * i

            atmo_phase = self.get_phasescreen_at_time(sim_time)
            input_phase = atmo_phase[~self.cmask]
            input_phase -= xp.mean(input_phase)  # remove piston
                
            if i >= self.sc.delay:
                dm_surf = IF @ dm_cmds[i - self.sc.delay, :]

            if self.tt_offload is not None and  i >= self.sc.delay:
                tt_coeffs[i,:] = c2tt @ dm_cmds[i-self.sc.delay,:] #tt_coeffs[i-1,:].copy()
                N = int(max(0,i-tt_offload_dt/self.dt))
                tt = xp.mean(tt_coeffs[N:i,:],axis=0)
                tt_amps += tt_gain * tt
                dm_surf += tt2surf @ tt_amps
                
            residual_phase = input_phase - dm_surf[self.dm.visible_pix_ids]


            if i % int(self.sc.dt/self.dt) == 0:
                int_cmds[i,:], modes = self.perform_loop_iteration(residual_phase, self.sc, 
                                                                    starMagnitude=starMagnitude, 
                                                                    slaving=self.dm.slaving)
                dm_cmds[i,:] += self.sc.iir_filter(int_cmds[:i+1,:], dm_cmds[:i+1,:])
            else:
                dm_cmds[i,:] += dm_cmds[i-1,:].copy()

            res_phase_rad2[i] = self.phase_rms(residual_phase*m2rad)**2 #[xp.abs(residual_phase)>0.0]
            atmo_phase_rad2[i] = self.phase_rms(input_phase*m2rad)**2

            if save_prefix is not None:            
                residual_phases[i, :] = residual_phase
                input_phases[i, :] = input_phase
                detector_images[i, :, :] = self.ccd.last_frame
                rec_modes[i, :] = modes
        

        if save_prefix is not None:
            print("Saving telemetry to .fits ...")
            dm_mask_cube = xp.asnumpy(xp.stack([self.dm.mask for _ in range(self.Nits)]))
            mask_cube = xp.asnumpy(xp.stack([self.cmask for _ in range(self.Nits)]))
            input_phases = xp.stack([reshape_on_mask(input_phases[i, :], self.cmask) for i in range(self.Nits)])
            dm_phases = xp.stack([reshape_on_mask(IF @ dm_cmds[i, :], self.dm.mask)for i in range(self.Nits)])
            if self.tt_offload is not None:
                tt_dm_phases = xp.stack([reshape_on_mask(tt2surf @ tt_coeffs[i, :], self.dm.mask)for i in range(self.Nits)])
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
                    detector_images,
                    rec_modes,
                    dm_cmds,
                    res_phase_rad2,
                ],
            ):
                data_dict[key] = value

            self.save_telemetry_data(data_dict, save_prefix)
            if self.tt_offload is not None:
                self.save_telemetry_data({'tt_dm_phases':tt_dm_phases},save_prefix)

        return res_phase_rad2, atmo_phase_rad2
    
    
    def plot_iteration(self, lambdaRef, frame_id:int=-1, save_prefix:str=''):
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

        ma_atmo_phases, _, ma_res_phases, det_frames, rec_modes, dm_cmds, _ = self.load_telemetry_data(save_prefix=save_prefix)

        atmo_phase_in_rad = ma_atmo_phases[frame_id].data[~ma_atmo_phases[frame_id].mask]*(2*xp.pi/lambdaRef)
        res_phase_in_rad = ma_res_phases[frame_id].data[~ma_res_phases[frame_id].mask]*(2*xp.pi/lambdaRef)

        in_err_rad2 = xp.asnumpy(self.phase_rms(atmo_phase_in_rad)**2)
        res_err_rad2 = xp.asnumpy(self.phase_rms(res_phase_in_rad)**2)

        psf, pixelSize = self._psf_from_frame(xp.array(ma_res_phases[frame_id]), lambdaRef)
        cmask = self.cmask.get() if xp.on_gpu else self.cmask.copy()

        plt.figure(figsize=(9,9))
        plt.subplot(2,2,1)
        myimshow(masked_array(ma_atmo_phases[frame_id],cmask), \
        title=f'Atmosphere phase [m]\nSR = {xp.exp(-in_err_rad2):1.3f} @ {lambdaRef*1e+9:1.0f} [nm]',\
        cmap='RdBu',shrink=0.8)
        plt.axis('off')
        plt.subplot(2,2,2)
        showZoomCenter(psf/xp.max(psf), pixelSize, shrink=0.8,
        title = f'Corrected PSF\nSR = {xp.exp(-res_err_rad2):1.3f} @{lambdaRef*1e+9:1.0f}[nm]'
            , cmap='inferno', xlabel=r'$\lambda/D$'
            , ylabel=r'$\lambda/D$', vmin=-10) 
        plt.subplot(2,2,3)
        myimshow(det_frames[frame_id], title = 'Detector frame', shrink=0.8)
        plt.subplot(2,2,4)
        self.dm.plot_position(dm_cmds[frame_id])
        plt.title('Mirror command [m]')
        plt.axis('off')

        N = 100 if 100 < self.Nits//2 else self.Nits//2
        atmo_modes = xp.zeros([N,self.KL.shape[0]])
        res_modes = xp.zeros([N,self.KL.shape[0]])
        phase2modes = xp.linalg.pinv(self.KL) #xp.linalg.pinv(self.KL.T)
        for frame in range(N):
            mask = ma_atmo_phases[-N+frame].mask.copy()
            atmo_phase = xp.asarray(ma_atmo_phases[-N+frame].data[~mask])
            atmo_modes[frame,:] = xp.dot(atmo_phase,phase2modes) #phase2modes @ atmo_phase
            res_phase = xp.asarray(ma_res_phases[-N+frame].data[~mask])
            res_modes[frame,:] = xp.dot(res_phase,phase2modes) #phase2modes @ res_phase
        atmo_mode_rms = xp.sqrt(xp.mean(atmo_modes**2,axis=0))
        res_mode_rms = xp.sqrt(xp.mean(res_modes**2,axis=0))
        rec_modes_rms = xp.sqrt(xp.mean(rec_modes[-N-1:-1,:]**2,axis=0))

        plt.figure()
        plt.plot(xp.asnumpy(atmo_mode_rms)*1e+9,label='turbulence')
        plt.plot(xp.asnumpy(res_mode_rms)*1e+9,label='residual (true)')
        plt.plot(xp.asnumpy(rec_modes_rms)*1e+9,'--',label='residual (measured)')
        plt.legend()
        plt.xlabel('mode index')
        plt.ylabel('mode RMS amp [nm]')
        plt.title('KL modes')
        plt.grid()
        plt.xscale('log')
        plt.yscale('log')

        # opt_gains = rec_modes_rms/res_mode_rms[:self.sc.nModes]

        # plt.figure()
        # plt.plot(xp.asnumpy(opt_gains),label='1st stage')
        # plt.xlabel('mode #')
        # plt.title('Optical gains')
        # plt.grid()
        # plt.xscale('log')
    

    def plot_contrast(self, lambdaRef, frame_ids:list=None, save_prefix:str='', oversampling:int=12, one_sided_contrast:bool=False):
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

        data_load = self.load_telemetry_data(data_keys=['residual_phases'], save_prefix=save_prefix)
        ma_res_phases = data_load[0]

        N = len(frame_ids)
        res_phases_in_rad = xp.zeros([N,int(xp.sum(1-self.cmask))])
        for j in range(N):
            res_phases_in_rad[j] = xp.asarray(ma_res_phases[frame_ids[j]].data[~ma_res_phases[frame_ids[j]].mask]*(2*xp.pi/lambdaRef))
        _,rms_psf,pix_dist=self.get_contrast(res_phases_in_rad,oversampling=oversampling,one_sided_contrast=one_sided_contrast)

        plt.figure()
        plt.plot(xp.asnumpy(pix_dist),xp.asnumpy(rms_psf),'--')
        plt.grid()
        plt.yscale('log')
        plt.xlabel(r'$\lambda/D$')
        plt.xlim([0,30])
        plt.ylim([1e-10,1e-2])
        plt.title(f'Contrast @ {lambdaRef*1e+9:1.0f} nm\n(assuming a perfect coronograph)')

        return rms_psf, pix_dist
    

    def get_tt_spectrum(self, save_prefix:str=None, show:bool=False):

        if save_prefix is None:
            save_prefix = self.save_prefix

        data_load = self.load_telemetry_data(data_keys=['dm_commands'], save_prefix=save_prefix)
        dm_cmds = xp.array(data_load[0])

        max_cmd = xp.max(xp.abs(dm_cmds),axis=1)
        # if self.dm.slaving is not None:
        #     dm_cmds = (xp.linalg.pinv(self.dm.slaving) @ dm_cmds.T).T
        # dm_modes = xp.linalg.pinv(self.sc.m2c) @ dm_cmds.T
        # TT = dm_modes[:2,:]

        TT = xp.linalg.pinv(self.dm.act_coords.T) @ dm_cmds.T

        spe = xp.fft.rfft(TT, norm="ortho", axis=-1)
        nn = xp.sqrt(spe.shape[-1])
        spe_tt = (xp.abs(spe)) / nn
        spe_tt[:,0] = 0 # remove DC component
        freq = xp.fft.rfftfreq(TT.shape[-1], d=self.dt)

        if show:
            plt.figure()
            plt.plot(xp.asnumpy(freq), xp.asnumpy(spe_tt[0])*1e+9, label='tip')
            plt.plot(xp.asnumpy(freq), xp.asnumpy(spe_tt[1])*1e+9, label='tilt')
            plt.grid()
            plt.legend()
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('WF RMS [nm]')
            plt.xscale('log')
            plt.yscale('log')

        return TT, spe_tt, freq, max_cmd



    # def plot_rec_modes(self, save_prefix:str=''):
    #     """
    #     Plots the reconstruced modes.
        
    #     Parameters
    #     ----------
    #     save_prefix : str, optional
    #         The prefix used when saving telemetry data, by default ''.
    #     """
    #     if save_prefix is None:
    #         save_prefix = self.save_prefix

    #     _, _, _, _, rec_modes, _, _ = self.load_telemetry_data(save_prefix=save_prefix)

    #     zern_modes = rec_modes[:,:5]
    #     max_mode = xp.max(xp.abs(rec_modes[:,5:]),axis=1)
    #     m2rad = 2*xp.pi/self.pyr.lambdaInM

    #     n_its = 100

    #     plt.figure()
    #     plt.plot(xp.asnumpy(xp.abs(zern_modes)*m2rad),'-o')
    #     plt.plot(xp.asnumpy(max_mode*m2rad),'-o')
    #     # plt.yscale('log')
    #     plt.xlabel('# iteration')
    #     plt.ylabel(f'[rad] @ {self.pyr.lambdaInM*1e+9:1.0f}nm')
    #     plt.legend(('Tip','Tilt','Defocus','AstigX','AstigY','Max'))
    #     plt.xlim([0, n_its])
    #     plt.grid()
    #     plt.title('Reconstructed modes amplitude')

    #     it_vec = xp.arange(self.Nits)
    #     plt.figure()
    #     plt.plot(xp.asnumpy(it_vec[-n_its:]),xp.asnumpy(zern_modes[-n_its:,:]),'-o')
    #     plt.xlabel('# iteration')
    #     plt.ylabel('[m]')
    #     plt.legend(('Tip','Tilt','Defocus','AstigX','AstigY'))
    #     plt.grid()
    #     plt.title(f'Reconstructed Zernike modes amplitude\nLast {n_its} iterations')

    #     last_modes = rec_modes[:,-5:]
    #     plt.figure()
    #     plt.plot(xp.asnumpy(it_vec[-n_its:]),xp.asnumpy(last_modes[-n_its:,:]),'-o')
    #     plt.xlabel('# iteration')
    #     plt.ylabel('[m]')
    #     plt.grid()
    #     plt.title(f'Last 5 modes amplitude\nLast {n_its} iterations')