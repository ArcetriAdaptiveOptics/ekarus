import xupy as xp
import numpy as np
from numpy.ma import masked_array
from skimage.restoration import unwrap_phase 

import matplotlib.pyplot as plt
from ekarus.e2e.utils.image_utils import showZoomCenter, myimshow, reshape_on_mask

from ekarus.e2e.devices.alpao_deformable_mirror import ALPAODM
from ekarus.e2e.high_level_ao_class import HighLevelAO

from typing import override


class PupilShift(HighLevelAO):

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
  

    @override
    def perform_loop_iteration(self, input_phase, dm_cmd, slope_computer, 
                               tilt_before_DM:tuple=(0.0,0.0), tilt_after_DM:tuple=(0.0,0.0),
                               starMagnitude:float=None, method:str='slopes'):
        """
        Performs a single iteration of the AO loop.

        Parameters
        ----------
        phase : array
            The input phase screen in meters.
        dm_cmd : array
            The current DM command in meters.
        slope_computer : SlopeComputer
            The slope computer object.
        starMagnitude : float
            The magnitude of the star being observed.
        
        Returns
        -------
        dm_cmd : xp.array
            The DM command in meters.
        modes : xp.array
            The reconstructed modes in meters.
        """

        lambda_ref = slope_computer._wfs.lambdaInM
        m2rad = 2*xp.pi/lambda_ref
        lambdaOverD = lambda_ref/self.pupilSizeInM
        Nphotons = self.get_photons_per_second(starMagnitude) * slope_computer.dt

        input_phase_in_rad = reshape_on_mask(input_phase * m2rad, self.cmask)
        padded_phase_in_rad = xp.pad(input_phase_in_rad, int((self.pyr.oversampling-1)/2*input_phase_in_rad.shape[0]), mode='constant', constant_values=0.0)
        padded_mask = xp.pad(self.cmask, int((self.pyr.oversampling-1)/2*self.cmask.shape[0]), mode='constant', constant_values=1.0)
        padded_dm_mask = xp.pad(self.dm.mask, int((self.pyr.oversampling-1)/2*self.dm.mask.shape[0]), mode='constant', constant_values=1.0)
        input_field = (1-padded_mask) * xp.exp(1j * padded_phase_in_rad)#, dtype=self.pyr.cdtype)

        if abs(max(tilt_before_DM)) > 0.0:
            input_field = self._tilt_field(input_field, tilt_before_DM[0], tilt_before_DM[1])

        dm_phase_in_rad = reshape_on_mask(self.dm.surface * m2rad, padded_dm_mask)
        dm_field = (1-padded_dm_mask) * xp.exp(1j * dm_phase_in_rad)#, dtype=self.pyr.cdtype)
        residual_field = input_field * dm_field.conj()

        if abs(max(tilt_after_DM)) > 0.0:
            residual_field = self._tilt_field(residual_field, tilt_after_DM[0], tilt_after_DM[1])
        
        intensity = self.pyr._intensity_from_field(residual_field, lambdaOverD)

        detector_image = self.ccd.image_on_detector(intensity, photon_flux=Nphotons)
        slopes = slope_computer._compute_pyr_signal(detector_image, method)
        modes = slope_computer.Rec @ slopes
        # modes = modes * slope_computer.modal_gains
        cmd = slope_computer.m2c @ modes

        if self.dm.slaving is not None:
            cmd = self.dm.slaving @ cmd
        
        dm_cmd += cmd * slope_computer.intGain / m2rad
        modes /= m2rad  # convert to meters

        # Recover the residual phase     
        if xp.max(abs(xp.angle(residual_field))) >= 0.99*xp.pi:
            residual_phase_2d = xp.array(unwrap_phase(xp.asnumpy(xp.angle(residual_field)*xp.abs(residual_field)))) # phase unwrapping
        else:
            residual_phase_2d = xp.angle(residual_field)*xp.abs(residual_field)
        residual_phase = residual_phase_2d[~padded_mask]
        residual_phase /= m2rad  # in meters

        return residual_phase, dm_cmd, modes
    

    def run_loop(self, lambdaInM:float, starMagnitude:float, 
                 tilt_before_DM:tuple=(0.0,0.0), tilt_after_DM:tuple=(0.0,0.0),
                 save_prefix:str=None):
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

        # self.pyr.set_modulation_angle(self.sc.modulationAngleInLambdaOverD) # reset modulation in case it was changed
        dm_cmd = xp.zeros(self.dm.Nacts, dtype=self.dtype) # reset DM position
        self.dm.set_position(dm_cmd, absolute=True)
        self.dm.surface -= self.dm.surface  # make sure DM is flat
        # print(xp.sum(self.dm.surface),xp.sum(self.dm.get_position()))

        # Define variables
        dm_mask_len = int(xp.sum(1 - self.dm.mask))
        mask_len = int(xp.sum(1 - self.cmask))
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

            residual_phase, dm_cmds[i,:], modes = self.perform_loop_iteration(input_phase, dm_cmd, self.sc,
                                                              tilt_before_DM, tilt_after_DM, starMagnitude)

            res_phase_rad2[i] = self.phase_rms(residual_phase[xp.abs(residual_phase)>0.0]*m2rad)**2
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
            , cmap='inferno', xlabel=r'$\lambda/D$', ylabel=r'$\lambda/D$') 
        plt.subplot(2,2,3)
        myimshow(det_frames[frame_id], title = 'Detector frame', shrink=0.8)
        plt.subplot(2,2,4)
        self.dm.plot_position(dm_cmds[frame_id])
        plt.title('Mirror command [m]')
        plt.axis('off')

    
    def _tilt_field(self, field, tiltAmpX, tiltAmpY):
        tiltX,tiltY = self.pyr._get_XY_tilt_planes(field.shape)
        wedge_tilt = (tiltX*tiltAmpX + tiltY*tiltAmpY)*(2*xp.pi)*self.pyr.oversampling
        focal_plane_field = xp.fft.fftshift(xp.fft.fft2(field))
        field = focal_plane_field * xp.exp(1j*wedge_tilt)#, dtype=self.pyr.cdtype)
        field = xp.fft.ifft2(xp.fft.ifftshift(field))
        return field

