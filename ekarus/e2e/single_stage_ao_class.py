import xupy as xp
import numpy as np

from ekarus.e2e.devices.alpao_deformable_mirror import ALPAODM
# from ekarus.e2e.devices.pyramid_wfs import PyramidWFS
# from ekarus.e2e.devices.detector import Detector
# from ekarus.e2e.devices.slope_computer import SlopeComputer

from ekarus.e2e.high_level_ao_class import HighLevelAO
from ekarus.e2e.utils.image_utils import reshape_on_mask  # , get_masked_array


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
        self.dm = ALPAODM(dm_pars["Nacts"], Npix=self.pupilSizeInPixels, max_stroke=dm_pars['max_stroke_in_m'])
        self.dm.mask = self.cmask
    

    def run_loop(self, lambdaInM:float, starMagnitude:float, save_prefix:str=None):
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
        # electric_field_amp = 1 - self.cmask

        # lambdaOverD = lambdaInM / self.pupilSizeInM
        # Nphotons = self.get_photons_per_second(starMagnitude) * self.sc.dt
        # modal_gains = self.sc.modal_gains

        self.pyr.set_modulation_angle(self.sc.modulationAngleInLambdaOverD)

        # print(self.sc._wfs.lambdaInM, self.dm.max_stroke, self.pyr.modulationAngleInLambdaOverD)

        # Define variables
        mask_len = int(xp.sum(1 - self.dm.mask))
        dm_cmd = xp.zeros(self.dm.Nacts, dtype=self.dtype)
        self.dm.set_position(dm_cmd, absolute=True)
        dm_cmds = xp.zeros([self.Nits, self.dm.Nacts])

        res_phase_rad2 = xp.zeros(self.Nits)
        atmo_phase_rad2 = xp.zeros(self.Nits)

        if save_prefix is not None:
            dm_phases = xp.zeros([self.Nits, mask_len])
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

            residual_phase = input_phase - self.dm.surface

            # delta_phase_in_rad = reshape_on_mask(residual_phase * m2rad, self.cmask)
            # input_field = electric_field_amp * xp.exp(1j * delta_phase_in_rad)
            # slopes = self.sc.compute_slopes(input_field, lambdaOverD, Nphotons)
            # modes = self.sc.Rec @ slopes
            # gmodes = modes * modal_gains
            # cmd = self.sc.m2c @ gmodes
            # dm_cmd += cmd * self.sc.intGain
            # dm_cmds[i, :] = dm_cmd / m2rad  # convert to meters

            dm_cmds[i,:], modes = self.perform_loop_iteration(residual_phase, dm_cmd, self.sc, starMagnitude)

            res_phase_rad2[i] = xp.std(residual_phase*m2rad)**2
            atmo_phase_rad2[i] = xp.std(input_phase*m2rad)**2

            if save_prefix is not None:            
                residual_phases[i, :] = residual_phase
                input_phases[i, :] = input_phase
                dm_phases[i, :] = self.dm.surface
                detector_images[i, :, :] = self.ccd.last_frame
                rec_modes[i, :] = modes
        

        if save_prefix is not None:
            print("Saving telemetry to .fits ...")
            ma_input_phases = np.stack([reshape_on_mask(input_phases[i, :], self.cmask)for i in range(self.Nits)])
            ma_dm_phases = np.stack([reshape_on_mask(dm_phases[i, :], self.cmask)for i in range(self.Nits)])
            ma_res_phases = np.stack([reshape_on_mask(residual_phases[i, :], self.cmask)for i in range(self.Nits)])

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
                ],
            ):
                data_dict[key] = value

            self.save_telemetry_data(data_dict, save_prefix)

        return res_phase_rad2, atmo_phase_rad2

