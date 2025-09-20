import xupy as xp

np = xp.np
import os

from ekarus.e2e.alpao_deformable_mirror import ALPAODM
from ekarus.e2e.pyramid_wfs import PyramidWFS
from ekarus.e2e.detector import Detector
from ekarus.e2e.slope_computer import SlopeComputer

from ekarus.e2e.high_level_ao_class import HighLevelAO
from ekarus.e2e.utils.image_utils import reshape_on_mask  # , get_masked_array

# from ekarus.e2e.utils.my_fits_package import read_fits


class SingleStageAO(HighLevelAO):

    def __init__(self, tn: str):
        """The constructor"""
        super().__init__(tn)
        self._initialize_devices()
        self.telemetry_keys = [
            "AtmoPhases",
            "DMphases",
            "ResPhases",
            "DetectorFrames",
            "DMcommands",
        ]


    def _initialize_devices(self):
        """
        Initializes the devices used in the AO system.

        - WFS
        - Detector
        - DM
        - Slope computer
        """
        sensor_pars = self._config.read_sensor_pars()
        apex_angle, oversampling, sensorLambda = (
            sensor_pars["apex_angle"],
            sensor_pars["oversampling"],
            sensor_pars["lambdaInM"],
        )
        self.pyr = PyramidWFS(apex_angle, oversampling, sensorLambda)

        detector_pars = self._config.read_detector_pars()
        detector_shape, RON, quantum_efficiency, beam_split_ratio = (
            detector_pars["detector_shape"],
            detector_pars["RON"],
            detector_pars["quantum_efficiency"],
            detector_pars["beam_splitter_ratio"]
        )
        self.ccd = Detector(
            detector_shape=detector_shape,
            RON=RON,
            quantum_efficiency=quantum_efficiency,
            beam_split_ratio=beam_split_ratio,
        )

        sc_pars = self._config.read_slope_computer_pars()
        self.sc = SlopeComputer(self.pyr, self.ccd, sc_pars)

        dm_pars = self._config.read_dm_pars()
        Nacts = dm_pars["Nacts"]
        max_stroke = dm_pars["maxStrokeInM"]
        self.dm = ALPAODM(Nacts, Npix=self.pupilSizeInPixels, max_stroke=max_stroke)
    

    def run_loop(self, lambdaInM:float, starMagnitude:float, Rec, m2c, save_telemetry_prefix:str=None):
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
        electric_field_amp = 1 - self.cmask

        modal_gains = xp.zeros(Rec.shape[0])
        modal_gains[: self.sc.nModes] = 1

        lambdaOverD = lambdaInM / self.pupilSizeInM
        Nphotons = self.get_photons_per_second(starMagnitude) * self.sc.dt

        self.pyr.set_modulation_angle(self.modulationAngleInLambdaOverD)

        # Define variables
        mask_len = int(xp.sum(1 - self.dm.mask))
        dm_cmd = xp.zeros(self.dm.Nacts, dtype=self.dtype)
        self.dm.set_position(dm_cmd, absolute=True)
        dm_cmds = xp.zeros([self.Nits, self.dm.Nacts])

        if save_telemetry_prefix is not None:
            dm_phases = xp.zeros([self.Nits, mask_len])
            residual_phases = xp.zeros([self.Nits, mask_len])
            input_phases = xp.zeros([self.Nits, mask_len])
            detector_images = xp.zeros(
                [self.Nits, self.ccd.detector_shape[0], self.ccd.detector_shape[1]]
            )

        # X,Y = image_grid(self.cmask.shape, recenter=True, xp=self._xp)
        # tiltX = X[~self.cmask]/(self.cmask.shape[0]//2)
        # tiltY = Y[~self.cmask]/(self.cmask.shape[1]//2)
        # TTmat = xp.stack((tiltX.T,tiltY.T),axis=1)

        for i in range(self.Nits):
            print(f"\rIteration {i+1}/{self.Nits}", end="\r", flush=True)
            sim_time = self.sc.dt * i

            atmo_phase = self.get_phasescreen_at_time(sim_time)
            input_phase = atmo_phase[~self.cmask]
            input_phase -= xp.mean(input_phase)  # remove piston

            # # Tilt offloading
            # if i>0 and ttOffloadFrequency > 0 and i % int(loopFrequencyInHz/ttOffloadFrequency) <= 1e-6:
            #     tt_coeffs = modes[:2]
            #     input_phase -= TTmat @ tt_coeffs

            if i >= self.sc.delay:
                self.dm.set_position(dm_cmds[i - self.sc.delay, :], absolute=True)
                
            # if i>0 and int(i%200)==0 and i < 700:
            #     self.integratorGain += 0.1

            residual_phase = input_phase - self.dm.surface
            delta_phase_in_rad = reshape_on_mask(residual_phase * m2rad, self.cmask)

            input_field = electric_field_amp * xp.exp(1j * delta_phase_in_rad)
            slopes = self.sc.compute_slopes(input_field, lambdaOverD, Nphotons)
            modes = Rec @ slopes
            modes *= modal_gains
            cmd = m2c @ modes
            dm_cmd += cmd * self.sc.intGain
            dm_cmds[i, :] = dm_cmd / m2rad  # convert to meters

            residual_phases[i, :] = residual_phase
            input_phases[i, :] = input_phase
            if save_telemetry_prefix is not None:
                dm_phases[i, :] = self.dm.surface
                detector_images[i, :, :] = self.ccd.last_frame

        errRad2 = xp.std(residual_phases * m2rad, axis=-1) ** 2
        inputErrRad2 = xp.std(input_phases * m2rad, axis=-1) ** 2
        

        if save_telemetry_prefix is not None:
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
                    dm_cmds,
                ],
            ):
                data_dict[key] = value

            self.save_telemetry_data(data_dict)

        return errRad2, inputErrRad2

