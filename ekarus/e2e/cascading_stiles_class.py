import xupy as xp
import numpy as np
from numpy.ma import masked_array

from ekarus.e2e.devices.deformable_secondary_mirror import DSM

# masked_array = xp.masked_array
import matplotlib.pyplot as plt
from ekarus.e2e.utils.image_utils import showZoomCenter, myimshow 

from ekarus.e2e.cascading_stage_ao_class import CascadingAO
from ekarus.e2e.utils.image_utils import reshape_on_mask #, get_masked_array

import typing

class CascadingBench(CascadingAO):

    def __init__(self, tn: str):
        """The constructor"""

        super().__init__(tn)
        

    @typing.override
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

        IF1 = self.dm1.IFF.copy()
        dm1_surf = xp.zeros(IF1.shape[0])

        IF2 = self.dm2.IFF.copy()
        dm2_surf = xp.zeros(IF2.shape[0])

        # Define variables
        mask_len = int(xp.sum(1 - self.cmask))
        int1_cmds = xp.zeros([self.Nits,self.dm1.Nacts], dtype=self.dtype)
        int2_cmds = xp.zeros([self.Nits,self.dm2.Nacts], dtype=self.dtype)
        dm1_cmds = xp.zeros([self.Nits,self.dm1.Nacts], dtype=self.dtype)
        dm2_cmds = xp.zeros([self.Nits,self.dm2.Nacts], dtype=self.dtype)

        res2_phase_rad2 = xp.zeros(self.Nits)
        res1_phase_rad2 = xp.zeros(self.Nits)
        atmo_phase_rad2 = xp.zeros(self.Nits)

        if save_prefix is not None:
            input_phases = xp.zeros([self.Nits, mask_len], dtype=self.dtype)
            residual1_phases = xp.zeros([self.Nits, mask_len], dtype=self.dtype)
            ccd1_images = xp.zeros([self.Nits, self.ccd1.detector_shape[0], self.ccd1.detector_shape[1]], dtype=self.dtype)
            rec1_modes = xp.zeros([self.Nits,self.sc1.Rec.shape[0]], dtype=self.dtype)
            residual2_phases = xp.zeros([self.Nits, mask_len], dtype=self.dtype)
            ccd2_images = xp.zeros([self.Nits, self.ccd2.detector_shape[0], self.ccd2.detector_shape[1]], dtype=self.dtype)
            rec2_modes = xp.zeros([self.Nits,self.sc2.Rec.shape[0]], dtype=self.dtype)

        phase2modes = xp.linalg.pinv(self.KL.T)
        modes2dm1phase = self.KL.T.copy()


        for i in range(self.Nits):
            print(f"\rIteration {i+1}/{self.Nits}", end="\r", flush=True)
            sim_time = self.dt * i

            atmo_phase = self.get_phasescreen_at_time(sim_time)
            in_phase = atmo_phase[~self.cmask]
            in_phase -= xp.mean(in_phase)  # remove piston

            # project atmosphere on DM1
            rec_atmo = phase2modes @ in_phase
            input_phase = modes2dm1phase @ rec_atmo

            if i >= self.sc1.delay:
                dm1_surf = IF1 @ dm1_cmds[i - self.sc1.delay, :]
            residual1_phase = input_phase - dm1_surf[self.dm1.visible_pix_ids]

            if i % int(self.sc1.dt/self.dt) == 0:
                int1_cmds[i,:], modes1 = self.perform_loop_iteration(residual1_phase, self.sc1, 
                                                                    starMagnitude=starMagnitude, 
                                                                    slaving=self.dm1.slaving)
                dm1_cmds[i,:] = self.sc1.iir_filter(int1_cmds[:i+1,:], dm1_cmds[:i+1,:])
            else:
                dm1_cmds[i,:] = dm1_cmds[i-1,:].copy()


            if i >= self.sc2.delay:
                dm2_surf = IF2 @ dm2_cmds[i - self.sc2.delay, :]
            residual2_phase = residual1_phase - dm2_surf[self.dm2.visible_pix_ids]

            if i % int(self.sc2.dt/self.dt) == 0:
                int2_cmds[i,:], modes2 = self.perform_loop_iteration(residual2_phase, self.sc2, 
                                                                    starMagnitude=starMagnitude, 
                                                                    slaving=self.dm2.slaving)
                dm2_cmds[i,:] = self.sc2.iir_filter(int2_cmds[:i+1,:], dm2_cmds[:i+1,:])
            else:
                dm2_cmds[i,:] = dm2_cmds[i-1,:].copy()

            res2_phase_rad2[i] = self.phase_rms(residual2_phase*m2rad)**2
            res1_phase_rad2[i] = self.phase_rms(residual1_phase*m2rad)**2
            atmo_phase_rad2[i] = self.phase_rms(input_phase*m2rad)**2

            if save_prefix is not None:  
                input_phases[i, :] = input_phase          
                residual1_phases[i, :] = residual1_phase
                ccd1_images[i, :, :] = self.ccd1.last_frame
                rec1_modes[i, :] = modes1
                residual2_phases[i, :] = residual2_phase
                ccd2_images[i, :, :] = self.ccd2.last_frame
                rec2_modes[i, :] = modes2
        

        if save_prefix is not None:
            print("Saving telemetry to .fits ...")
            dm1_mask_cube = xp.asnumpy(xp.stack([self.dm1.mask for _ in range(self.Nits)]))
            dm2_mask_cube = xp.asnumpy(xp.stack([self.dm2.mask for _ in range(self.Nits)]))
            mask_cube = xp.asnumpy(xp.stack([self.cmask for _ in range(self.Nits)]))
            input_phases = xp.stack([reshape_on_mask(input_phases[i, :], self.cmask) for i in range(self.Nits)])
            dm1_phases = xp.stack([reshape_on_mask(IF1 @ dm1_cmds[i, :], self.dm1.mask)for i in range(self.Nits)])
            res1_phases = xp.stack([reshape_on_mask(residual1_phases[i, :], self.cmask)for i in range(self.Nits)])
            dm2_phases = xp.stack([reshape_on_mask(IF2 @ dm2_cmds[i, :], self.dm2.mask)for i in range(self.Nits)])
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
    


        