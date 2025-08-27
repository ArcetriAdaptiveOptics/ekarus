import numpy as np
import os

from ekarus.e2e.utils import my_fits_package as myfits
from ekarus.e2e.utils.read_configuration import ConfigReader

from ekarus.e2e.alpao_deformable_mirror import ALPAODM
from ekarus.e2e.pyramid_wfs import PyramidWFS
from ekarus.e2e.detector import Detector
from ekarus.e2e.slope_computer import SlopeComputer

from ekarus.e2e.utils.image_utils import get_circular_mask, reshape_on_mask
from ekarus.analytical.kl_modes import make_modal_base_from_ifs_fft


class SingleStageAO():

    def __init__(self, tn, xp=np):

        self.basepath = os.getcwd()
        dir_path = os.path.join(self.basepath,'ekarus/simulations/Results',str(tn))

        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

        self.savepath = dir_path

        self.read_configuration(tn)
        self.initialize_devices()

        self.lambdaInM = None
        self.starMagnitude = None

        self._xp = xp
        self.dtype = xp.float32 if xp.__name__ == 'cupy' else xp.float64


    def get_wavelength(self):
        return self.lambdaInM
    
    def set_wavelength(self, lambdaInM):
        self.lambdaInM = lambdaInM
        self.pixelScale = self.get_pixel_size()

    def get_star_magnitude(self):
        return self.starMagnitude
    
    def set_star_magnitude(self, starMagnitude):
        self.starMagnitude = starMagnitude
        self.photon_flux = self.get_photons_per_second()


    def get_pixel_size(self):
        return self.lambdaInM/self.pupilSizeInM/self.oversampling

    def get_photons_per_second(self, B0 = 1e+10):
        collected_flux = None
        if self.starMagnitude is not None:
            total_flux = B0 * 10**(-self.starMagnitude/2.5)
            collecting_area = self.telescopeSizeInM**2/4*self._xp.pi
            collected_flux = self.throughput * total_flux * collecting_area
        return collected_flux
    
    
    def get_slopes(self, input_field, Nphotons, **kwargs):

        modulated_intensity = self.wfs.modulate(input_field, **kwargs, pixel_scale=self.pixelScale)
        detector_image = self.ccd.image_on_detector(modulated_intensity, photon_flux = Nphotons)
        slopes = self.slope_computer.compute_slopes(detector_image)

        return slopes


    def define_KL_modal_base(self, r0, L0, zern2remove:int = 5):

        KL, m2c, _ = make_modal_base_from_ifs_fft(1-self.dm.mask, self.pupilSizeInPixels,
        self.pupilSizeInM, self.dm.IFF.T, r0, L0, zern_modes=zern2remove,
        oversampling=self.oversampling, verbose = True, xp=self._xp, dtype=self.dtype)

        return KL, m2c
    
    
    def calibrate_modes(self, MM, amps:float = 0.1, **kwargs):

        Nmodes = self._xp.shape(MM)[0]
        slopes = None
        electric_field_amp = 1-self.cmask

        if isinstance(amps, float):
            amps *= self._xp.ones(Nmodes)

        for i in range(Nmodes):
            print(f'\rMode {i+1}/{Nmodes}', end='')
            amp = amps[i]
            mode_phase = reshape_on_mask(MM[i,:]*amp, self.cmask, xp=self._xp)
            input_field = self._xp.exp(1j*mode_phase) * electric_field_amp
            push_slope = self.get_slopes(input_field, **kwargs)/amp

            input_field = self._xp.conj(input_field)
            pull_slope = self.get_slopes(input_field, **kwargs)/amp

            if slopes is None:
                slopes = (push_slope-pull_slope)/2
            else:
                slopes = self._xp.vstack((slopes,(push_slope-pull_slope)/2))

        print(' ')
        IM = slopes.T
        U,S,Vt = self._xp.linalg.svd(IM, full_matrices=False)
        Rec = (Vt.T*1/S) @ U.T

        return Rec, IM, 1/S


    def perform_loop_iteration(self, time_step, input_phase, Rec, **kwargs):

        Nphotons = self.photon_flux * time_step

        input_field = (1-self.cmask) * self._xp.exp(1j*input_phase)
        slopes = self.get_slopes(input_field, Nphotons, **kwargs)
        modes = Rec @ slopes

        return modes
    
    def read_configuration(self, tn):
        config_path = os.path.join(self.basepath,'ekarus/simulations/Config',str(tn))
        self._config = ConfigReader(config_path)
        self.pupilSizeInM, self.pupilSizeInPixels, self.oversampling = self._config.read_pupil_pars()
        mask_shape = (self.oversampling * self.pupilSizeInPixels, self.oversampling * self.pupilSizeInPixels)
        self.cmask = get_circular_mask(mask_shape, mask_radius=self.pupilSizeInPixels//2, xp=self._xp)
        self.throughput, self.telescopeSizeInM = self._config.read_telescope_pars()
    
    
    def initialize_devices(self):
        print('Initializing devices ...')
        apex_angle = self._config.read_sensor_pars()
        detector_shape, RON = self._config.read_detector_pars()
        Nacts = self._config.read_dm_pars()

        self.wfs = PyramidWFS(apex_angle, xp=self._xp)
        self.ccd = Detector(detector_shape=detector_shape, xp=self._xp)
        self.dm = ALPAODM(Nacts, Npix=self.pupilSizeInPixels, xp=self._xp)
        self.slope_computer = SlopeComputer(wfs_type = 'PyrWFS', xp=self._xp)







        