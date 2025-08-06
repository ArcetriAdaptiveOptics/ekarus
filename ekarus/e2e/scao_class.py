import numpy as np

# from arte.types.mask import CircularMask

from ekarus.e2e.utils.image_utils import get_circular_mask, reshape_on_mask
from ekarus.e2e.utils.kl_modes import make_modal_base_from_ifs_fft


class SCAO():

    def __init__(self, wfs, ccd, dm, pupil_pixel_size, pupil_size, throughput = None, oversampling:int = 4, xp=np):

        mask_shape = (oversampling * pupil_pixel_size, oversampling * pupil_pixel_size)
        self.cmask = get_circular_mask(mask_shape, mask_radius=pupil_pixel_size//2)

        self.oversampling = oversampling

        self.pupilSizeInPixels = pupil_pixel_size
        self.pupilSizeInM = pupil_size

        self.wfs = wfs
        self.ccd = ccd
        self.dm = dm

        self.throughput = throughput

        self._xp = xp
        self.dtype = xp.float32 if xp.__name__ == 'cupy' else xp.float64


    def _pixel_size(self, lambdaInM):
        return lambdaInM/self.pupilSizeInM/self.oversampling

    
    def photon_flux(self, starMagnitude, B0 = 1e+10):
        Nphot = None
        if starMagnitude is not None:
            total_flux = B0 * 10**(-starMagnitude/2.5)
            Nphot = total_flux * self.throughput
        return Nphot


    def define_subaperture_masks(self, lambdaInM, subaperture_pixels, star_magnitude = None, modulation_in_lambda_over_d = 20):

        pix_scale = self._pixel_size(lambdaInM=lambdaInM)
        alpha = pix_scale * self.oversampling * modulation_in_lambda_over_d

        piston = 1-self.cmask
        modulated_intensity = self.wfs.modulate(piston, alpha, pix_scale)

        Nphot = self.photon_flux(starMagnitude=star_magnitude)

        self.ccd.define_subaperture_masks(modulated_intensity, Npix = subaperture_pixels, photon_flux = Nphot)


    def define_KL_modal_base(self, r0, L0, zern2remove:int = 5):

        KL, m2c, _ = make_modal_base_from_ifs_fft(self.cmask, self.pupilSizeInPixels,
        self.pupilSizeInM, self.dm.IFF.T, r0, L0, zern_modes=zern2remove,
        oversampling=self.oversampling, verbose = True)

        self.KL = KL
        self.m2c = m2c


    def perform_loop_iteration(self, input_phase, lambdaInM, modulation_angle, starMagnitude = None):

        pix_scale = self._pixel_size(lambdaInM=lambdaInM)
        # alpha = pix_scale * self.oversampling * self.modN

        Nphot = self.photon_flux(starMagnitude=starMagnitude)

        input_field = self.cmask * self._xp.exp(1j*input_phase)
        meas_intensity = self.wfs.modulate(input_field, modulation_angle, pix_scale)
        slopes = self.ccd.compute_slopes(meas_intensity, photon_flux = Nphot)
        modes = self.Rec @ slopes
        cmd = self.m2c @ modes

        ccd_image = self.ccd.resize_on_detector(meas_intensity)

        return cmd, ccd_image


    def calibrate_modes(self, MM, lambdaInM, modulation_angle, amps:float = 0.1, starMagnitude = None):

        Nmodes = self._xp.shape(MM)[0]
        slope_len = int(self._xp.sum(1-self.ccd.subapertures[0])*2)
        IM = self._xp.zeros((slope_len,Nmodes))

        pix_scale = self._pixel_size(lambdaInM=lambdaInM)
        # alpha = pix_scale * self.oversampling * self.modN

        Nphot = self.photon_flux(starMagnitude=starMagnitude)

        if isinstance(amps, float):
            amps *= self._xp.ones(Nmodes)

        for i in range(Nmodes):
            amp = amps[i]
            # mode_phase = self._xp.zeros(self.cmask.shape)
            # mode_phase[~self.cmask] = MM[i,:]*amp
            # mode_phase = self._xp.reshape(mode_phase, self.cmask.shape)
            mode_phase = reshape_on_mask(MM[i,:]*amp, self.cmask)
            input_field = self._xp.exp(1j*mode_phase) * self.cmask
            modulated_intensity = self.wfs.modulate(input_field, modulation_angle, pix_scale)
            push_slope = self.ccd.compute_slopes(modulated_intensity, photon_flux = Nphot)/amp

            input_field = self._xp.conj(input_field)
            modulated_intensity = self.wfs.modulate(input_field, modulation_angle, pix_scale)
            pull_slope = self.ccd.compute_slopes(modulated_intensity, photon_flux = Nphot)/amp

            IM[:,i] = (push_slope-pull_slope)/2

        self.IM = IM

        U,S,Vt = self._xp.linalg.svd(IM, full_matrices=False)
        self.Rec = (Vt.T*1/S) @ U.T

        