import numpy as np
import os

from ekarus.e2e.utils import my_fits_package as myfits
from ekarus.e2e.utils.read_configuration import ConfigReader
from arte.atmo.phase_screen_generator import PhaseScreenGenerator
from ekarus.analytical.turbulence_layers import TurbulenceLayers

from ekarus.e2e.alpao_deformable_mirror import ALPAODM
from ekarus.e2e.pyramid_wfs import PyramidWFS
from ekarus.e2e.detector import Detector
from ekarus.e2e.slope_computer import SlopeComputer

from ekarus.e2e.utils.image_utils import get_circular_mask, reshape_on_mask
from ekarus.analytical.kl_modes import make_modal_base_from_ifs_fft


class SingleStageAO():

    def __init__(self, tn, xp=np):

        self.basepath = os.getcwd()
        self.savepath = os.path.join(self.basepath,'ekarus','simulations','Results',str(tn))

        if not os.path.exists(self.savepath):
            os.mkdir(self.savepath)

        self._xp = xp
        self.dtype = xp.float32 if xp.__name__ == 'cupy' else xp.float64

        self.read_configuration(tn)
        self.initialize_devices()

        self.lambdaInM = None
        self.starMagnitude = None


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
            collecting_area = self._xp.pi/4*self.telescopeSizeInM**2
            collected_flux = self.throughput * total_flux * collecting_area
        return collected_flux
    

    def define_subaperture_masks(self):
        subap_path = os.path.join(self.savepath,'SubapertureMasks.fits')
        try:
            subaperture_masks = myfits.read_fits(subap_path, isBool=True)
            if self._xp.__name__ == 'cupy':
                subaperture_masks = self._xp.asarray(subaperture_masks)
            self.slope_computer._subaperture_masks = subaperture_masks
        except FileNotFoundError:
            _, subaperture_size = self._config.read_sensor_pars()
            piston = 1-self.cmask
            alpha = 10*self.pixelScale*self.oversampling
            modulated_intensity = self.wfs.modulate(piston, alpha, self.pixelScale)
            detector_image = self.ccd.image_on_detector(modulated_intensity)
            self.slope_computer.calibrate_sensor(subaperture_image = detector_image, Npix = subaperture_size)
            subaperture_masks = self.slope_computer._subaperture_masks
            if self._xp.__name__ == 'cupy':
                detector_image = detector_image.get()
            hdr_dict = {'APEX_ANG': self.wfs.apex_angle, 'PIX_SCAL': self.pixelScale, 'OVERSAMP': self.oversampling,  'SUBAPPIX': subaperture_size}
            myfits.save_fits(subap_path, (subaperture_masks).astype(self._xp.uint8), hdr_dict)
            
    
    def get_slopes(self, input_field, Nphotons, **kwargs):

        modulated_intensity = self.wfs.modulate(input_field, **kwargs, pixel_scale=self.pixelScale)
        detector_image = self.ccd.image_on_detector(modulated_intensity, photon_flux = Nphotons)
        slopes = self.slope_computer.compute_slopes(detector_image)

        return slopes
    
    
    def compute_reconstructor(self, MM, amps:float = 0.1, **kwargs):
        Nmodes = self._xp.shape(MM)[0]
        slopes = None
        electric_field_amp = 1-self.cmask

        Nphotons = None # None for 0 noise

        if isinstance(amps, float):
            amps *= self._xp.ones(Nmodes)

        for i in range(Nmodes):
            print(f'\rMode {i+1}/{Nmodes}', end='')
            amp = amps[i]
            mode_phase = reshape_on_mask(MM[i,:]*amp, self.cmask, xp=self._xp)
            input_field = self._xp.exp(1j*mode_phase) * electric_field_amp
            push_slope = self.get_slopes(input_field, Nphotons, **kwargs)/amp

            input_field = self._xp.conj(input_field)
            pull_slope = self.get_slopes(input_field, Nphotons, **kwargs)/amp

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
    
    
    def define_KL_modes(self, zern_modes:int = 5):
        KL_path = os.path.join(self.savepath,'KLmatrix.fits')
        m2c_path = os.path.join(self.savepath,'m2c.fits')
        try:
            KL = myfits.read_fits(KL_path)
            m2c = myfits.read_fits(m2c_path)
            if self._xp.__name__ == 'cupy':
                KL = self._xp.asarray(KL, dtype=self._xp.float32)
                m2c = self._xp.asarray(m2c, dtype=self._xp.float32)
        except FileNotFoundError:
            r0s, L0, _, _ = self._config.read_atmo_pars()
            r0 = self._xp.sqrt(self._xp.sum(r0s**2))
            print(r0)
            KL, m2c, _ = make_modal_base_from_ifs_fft(1-self.dm.mask, self.pupilSizeInPixels, \
                self.pupilSizeInM, self.dm.IFF.T, r0, L0, zern_modes=zern_modes,\
                oversampling=self.oversampling, verbose = True, xp=self._xp, dtype=self.dtype)
            hdr_dict = {'r0': r0, 'L0': L0, 'N_ZERN': zern_modes}
            myfits.save_fits(KL_path, KL, hdr_dict)
            myfits.save_fits(m2c_path, m2c, hdr_dict)
        return KL, m2c
    

    def calibrate_modes(self, MM, amps, **kwargs):
        IM_path = os.path.join(self.savepath,'IMmatrix.fits')
        Rec_path = os.path.join(self.savepath,'Rec.fits')
        try:
            IM = myfits.read_fits(IM_path)
            Rec = myfits.read_fits(Rec_path)
            if self._xp.__name__ == 'cupy':
                IM = self._xp.asarray(IM, dtype=self._xp.float32)
                Rec = self._xp.asarray(Rec, dtype=self._xp.float32)
        except FileNotFoundError:
            Rec, IM, _ = self.compute_reconstructor(MM, amps, **kwargs)
            myfits.save_fits(IM_path, IM)
            myfits.save_fits(Rec_path, Rec)
        return Rec, IM

    
    def read_configuration(self, tn):
        config_path = os.path.join(self.basepath,'ekarus','simulations','Config',str(tn)+'.ini')
        self._config = ConfigReader(config_path, self._xp)
        self.pupilSizeInM, self.pupilSizeInPixels, self.oversampling = self._config.read_pupil_pars()
        mask_shape = (self.oversampling * self.pupilSizeInPixels, self.oversampling * self.pupilSizeInPixels)
        self.cmask = get_circular_mask(mask_shape, mask_radius=self.pupilSizeInPixels//2, xp=self._xp)
        self.telescopeSizeInM, self.throughput = self._config.read_telescope_pars()
    
    
    def initialize_devices(self):
        apex_angle, _ = self._config.read_sensor_pars()
        detector_shape, RON = self._config.read_detector_pars()
        Nacts = self._config.read_dm_pars()

        self.wfs = PyramidWFS(apex_angle, xp=self._xp)
        self.ccd = Detector(detector_shape=detector_shape, xp=self._xp)
        self.dm = ALPAODM(Nacts, Npix=self.pupilSizeInPixels, xp=self._xp)
        self.slope_computer = SlopeComputer(wfs_type = 'PyrWFS', xp=self._xp)


    def initialize_turbulence(self, N:int=10):
        screenPixels = N*self.oversampling*self.pupilSizeInPixels
        screenMeters = N*self.oversampling*self.telescopeSizeInM#self.pupilSizeInM
        atmo_path = os.path.join(self.savepath, 'AtmoScreens.fits')
        r0s, L0, windSpeeds, windAngles = self._config.read_atmo_pars()
        self.layers = TurbulenceLayers(r0s, L0, windSpeeds, windAngles, atmo_path)
        print(f'Generating {self.layers.Nscreens:1.0f} phase-screens ...')
        self.layers.generate_phase_screens(screenPixels, screenMeters)
        self.layers.rescale_phasescreens(self.lambdaInM)
        self.layers.update_mask(self.cmask)


    def get_phase_screen(self, dt):
        masked_phases = self.layers.move_mask_on_phasescreens(dt)
        masked_phase = self._xp.sum(masked_phases,axis=0)
        return masked_phase


    def generate_phase_screens(self, N:int=10):
        screenPixels = N*self.oversampling*self.pupilSizeInPixels
        screenMeters = N*self.oversampling*self.telescopeSizeInM#self.pupilSizeInM
        self.pixelsPerMeter = screenPixels/screenMeters
        atmo_path = os.path.join(self.savepath, 'AtmoScreens.fits')
        try:
            phase_screens = myfits.read_fits(atmo_path)
            if self._xp.__name__ == 'cupy':
                phase_screens = self._xp.asarray(phase_screens, dtype=self._xp.float32)
        except FileNotFoundError:
            r0, L0, _, _ = self._config.read_atmo_pars()
            if isinstance(r0, float):
                Nscreens = 1
            else:
                Nscreens = len(r0)
            phs = PhaseScreenGenerator(screenPixels, screenMeters, outerScaleInMeters=L0, seed=42)
            phs.generate_normalized_phase_screens(Nscreens)
            phs.rescale_to(r0At500nm=r0)
            phs.get_in_radians_at(self.lambdaInM)
            phs.save_normalized_phase_screens(atmo_path)
            phase_screens = self._xp.asarray(phs._phaseScreens, dtype=self.dtype)
            hdr_dict = {'r0': r0, 'L0': L0, 'N_SCREEN': Nscreens}
            myfits.save_fits(atmo_path, phase_screens, hdr_dict)

        return phase_screens







        