import xupy as xp
import numpy as np
from numpy.ma import masked_array

from ekarus.e2e.devices.alpao_deformable_mirror import ALPAODM
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
            "rms2_residual"
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

        self.pyr, self.ccd, self.sc = self._initialize_pyr_slope_computer('PYR','CCD','SLOPE.COMPUTER')

        dm_pars = self._config.read_dm_pars()
        # self.dm = ALPAODM(dm_pars["Nacts"], Npix=self.pupilSizeInPixels, max_stroke=dm_pars['max_stroke_in_m'])
        self.dm = ALPAODM(dm_pars["Nacts"], pupil_mask = self.cmask.copy(), max_stroke=dm_pars['max_stroke_in_m'])
    

    def run_loop(self, lambdaInM:float, starMagnitude:float, use_diagonal:bool=False, save_prefix:str=None):
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

        dm_cmd = xp.zeros(self.dm.Nacts, dtype=self.dtype)
        self.dm.set_position(dm_cmd, absolute=True)
        self.dm.surface -= self.dm.surface  # make sure DM is flat

        # Define variables
        mask_len = int(xp.sum(1 - self.cmask))
        dm_mask_len = int(xp.sum(1 - self.dm.mask))
        dm_cmds = xp.zeros([self.Nits, self.dm.Nacts])

        res_phase_rad2 = xp.zeros(self.Nits)
        atmo_phase_rad2 = xp.zeros(self.Nits)

        if save_prefix is not None:
            dm_phases = xp.zeros([self.Nits, dm_mask_len])
            residual_phases = xp.zeros([self.Nits, mask_len])
            input_phases = xp.zeros([self.Nits, mask_len])
            detector_images = xp.zeros([self.Nits, self.ccd.detector_shape[0], self.ccd.detector_shape[1]])
            rec_modes = xp.zeros([self.Nits,self.sc.Rec.shape[0]])

        for i in range(self.Nits):
            print(f"\rIteration {i+1}/{self.Nits}", end="\r", flush=True)
            sim_time = self.dt * i

            atmo_phase = self.get_phasescreen_at_time(sim_time)
            input_phase = atmo_phase[~self.cmask]
            input_phase -= xp.mean(input_phase)  # remove piston

            if i >= self.sc.delay:
                self.dm.set_position(dm_cmds[i - self.sc.delay, :], absolute=True)
                
            # if i>0 and int(i%200)==0 and i < 700:
            #     self.integratorGain += 0.1

            residual_phase = input_phase - self.dm.get_surface()
            
            if i % int(self.sc.dt/self.dt) == 0:
                dm_cmds[i,:], modes = self.perform_loop_iteration(residual_phase, dm_cmd, self.sc, slaving=self.dm.slaving,
                                                                  use_diagonal=use_diagonal, starMagnitude=starMagnitude)
            else:
                dm_cmds[i,:] = dm_cmds[i-1,:].copy()

            res_phase_rad2[i] = self.phase_rms(residual_phase*m2rad)**2
            atmo_phase_rad2[i] = self.phase_rms(input_phase*m2rad)**2

            if save_prefix is not None:            
                residual_phases[i, :] = residual_phase
                input_phases[i, :] = input_phase
                dm_phases[i, :] = self.dm.surface
                detector_images[i, :, :] = self.ccd.last_frame
                rec_modes[i, :] = modes
        

        if save_prefix is not None:
            print("Saving telemetry to .fits ...")
            dm_mask_cube = xp.asnumpy(xp.stack([self.dm.mask for _ in range(self.Nits)]))
            mask_cube = xp.asnumpy(xp.stack([self.cmask for _ in range(self.Nits)]))
            input_phases = xp.stack([reshape_on_mask(input_phases[i, :], self.cmask) for i in range(self.Nits)])
            dm_phases = xp.stack([reshape_on_mask(dm_phases[i, :], self.dm.mask)for i in range(self.Nits)])
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

        ma_atmo_phases, _, ma_res_phases, det_frames, _, dm_cmds, _ = self.load_telemetry_data(save_prefix=save_prefix)

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
        showZoomCenter(psf, pixelSize, shrink=0.8,
        title = f'Corrected PSF\nSR = {xp.exp(-res_err_rad2):1.3f} @{lambdaRef*1e+9:1.0f}[nm]'
            , cmap='inferno', xlabel=r'$\lambda/D$'
            , ylabel=r'$\lambda/D$') 
        plt.subplot(2,2,3)
        myimshow(det_frames[frame_id], title = 'Detector frame', shrink=0.8)
        plt.subplot(2,2,4)
        self.dm.plot_position(dm_cmds[frame_id])
        plt.title('Mirror command [m]')
        plt.axis('off')

