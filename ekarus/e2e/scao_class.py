import numpy as np

from ekarus.e2e.utils.image_utils import get_circular_mask, reshape_on_mask
from ekarus.analytical.kl_modes import make_modal_base_from_ifs_fft


class SCAO():

    def __init__(self, wfs, ccd, slope_computer, dm, pupil_pixel_size, pupil_size, throughput = None, oversampling:int = 4, xp=np):

        mask_shape = (oversampling * pupil_pixel_size, oversampling * pupil_pixel_size)
        self.cmask = get_circular_mask(mask_shape, mask_radius=pupil_pixel_size//2, xp=xp)

        self.oversampling = oversampling

        self.pupilSizeInPixels = pupil_pixel_size
        self.pupilSizeInM = pupil_size

        self.wfs = wfs
        self.ccd = ccd
        self.dm = dm
        self.slope_computer = slope_computer

        self.throughput = throughput

        self.lambdaInM = None
        self.starMagnitude = None

        self._xp = xp
        self.dtype = xp.float32 if xp.__name__ == 'cupy' else xp.float64


    def get_wavelength(self):
        return self.lambdaInM
    
    def set_wavelength(self, lambdaInM):
        self.lambdaInM = lambdaInM

    def get_star_magnitude(self):
        return self.starMagnitude
    
    def set_star_magnitude(self, starMagnitude):
        self.starMagnitude = starMagnitude


    def get_pixel_size(self):
        return self.lambdaInM/self.pupilSizeInM/self.oversampling

    
    def get_photon_flux(self, B0 = 1e+10):
        Nphot = None
        if self.starMagnitude is not None:
            total_flux = B0 * 10**(-self.starMagnitude/2.5)
            Nphot = total_flux * self.throughput
        return Nphot
    
    
    def get_slopes(self, input_field, **kwargs):
        
        pix_scale = self.get_pixel_size()
        Nphot = self.get_photon_flux()

        modulated_intensity = self.wfs.modulate(input_field, **kwargs, pixel_scale=pix_scale)
        detector_image = self.ccd.image_on_detector(modulated_intensity, photon_flux = Nphot)
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

        IM = slopes.T
        U,S,Vt = self._xp.linalg.svd(IM, full_matrices=False)
        Rec = (Vt.T*1/S) @ U.T

        return IM, Rec


    def perform_loop_iteration(self, input_phase, Rec, m2c = None, **kwargs):

        if m2c is None:
            m2c = self._xp.eye((self.dm.Nacts,self._xp.shape(Rec)[0]))

        input_field = (1-self.cmask) * self._xp.exp(1j*input_phase)
        slopes = self.get_slopes(input_field, **kwargs)
        modes = Rec @ slopes
        cmd = m2c @ modes

        return cmd
    


    # def define_subaperture_masks(self, lambdaInM, subaperture_pixels, star_magnitude = None, modulation_in_lambda_over_d = 20):

    #     pix_scale = self._pixel_size(lambdaInM=lambdaInM)
    #     alpha = pix_scale * self.oversampling * modulation_in_lambda_over_d

    #     piston = 1-self.cmask
    #     modulated_intensity = self.wfs.modulate(piston, alpha, pix_scale)

    #     Nphot = self.photon_flux(starMagnitude=star_magnitude)

    #     self.ccd.define_subaperture_masks(modulated_intensity, Npix = subaperture_pixels, photon_flux = Nphot)

    #     # image = self.ccd.resize_on_detector(modulated_intensity, photon_flux = Nphot)
    #     #self.clope_computer = SlopeComputer(self.wfs, subaperture_image=image, Npix=subapertureSizeInpixels)





        