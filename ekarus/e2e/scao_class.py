import numpy as np

from arte.types.mask import CircularMask
from ekarus.e2e.utils.kl_modes import make_modal_base_from_ifs_fft


class SCAO():

    def __init__(self, wfs, ccd, dm, Npix, pupilSizeInM, throughput = None, modulationAngleInLambdaoverD = 3.0, oversampling:int = 4, xp=np):

        self.Npix = Npix
        self.cmask = CircularMask((oversampling * Npix, oversampling * Npix), maskRadius=Npix // 2)

        self.oversampling = oversampling

        self.pupilSizeInM = pupilSizeInM
        self.modN = modulationAngleInLambdaoverD

        self.wfs = wfs
        self.ccd = ccd
        self.dm = dm

        self.throughput = throughput

        self._xp = xp


    def _pixel_size(self, lambdaInM):
        return lambdaInM/self.pupilSizeInM/self.oversampling

    
    def photon_flux(self, starMagnitude, B0 = 1e+10):
        Nphot = None
        if starMagnitude is not None:
            total_flux = B0 * 10**(-starMagnitude/2.5)
            Nphot = total_flux * self.throughput
        return Nphot


    def define_subaperture_masks(self, lambdaInM, subapertureSizeInPixels, starMagnitude = None, modulationAngleInLambdaoverD = 20):

        pix_scale = self._pixel_size(lambdaInM=lambdaInM)
        alpha = pix_scale * self.oversampling * modulationAngleInLambdaoverD

        piston = 1-self.cmask.mask()
        modulated_intensity = self.wfs.modulate(piston, alpha, pix_scale)

        Nphot = self.photon_flux(starMagnitude=starMagnitude)

        self.ccd.define_subaperture_masks(modulated_intensity, Npix = subapertureSizeInPixels, photon_flux = Nphot)

        # image = self.ccd.resize_on_detector(modulated_intensity, photon_flux = Nphot)
        #self.clope_computer = SlopeComputer(self.wfs, subaperture_image=image, Npix=subapertureSizeInpixels)


    def define_KL_modal_base(self, r0, L0, zern2remove:int = 5):

        KL, m2c, singular_values = make_modal_base_from_ifs_fft(1-self.dm.mask, \
        self.pupilSizeInM, self.dm.IFF.T, r0, L0, zern_modes=zern2remove, zern_mask = self.cmask, \
        oversampling=self.oversampling, verbose = True)

        return KL, m2c


    def perform_loop_iteration(self, input_phase, Rec, lambdaInM, m2c = None, starMagnitude = None):

        pix_scale = self._pixel_size(lambdaInM=lambdaInM)
        alpha = pix_scale * self.oversampling * self.modN

        Nphot = self.photon_flux(starMagnitude=starMagnitude)

        if m2c is None:
            m2c = self._xp.eye((self.dm.Nacts,self._xp.shape(Rec)[0]))

        input_field = self.cmask.asTransmissionValue() * self._xp.exp(1j*input_phase)
        meas_intensity = self.wfs.modulate(input_field, alpha, pix_scale)
        slopes = self.ccd.compute_slopes(meas_intensity, photon_flux = Nphot)
        modes = Rec @ slopes
        cmd = m2c @ modes

        ccd_image = self.ccd.resize_on_detector(meas_intensity)

        return cmd, ccd_image


    def calibrate_modes(self, MM, lambdaInM, amps:float = 0.1, starMagnitude = None):

        Nmodes = self._xp.shape(MM)[0]
        slope_len = int(self._xp.sum(1-self.ccd.subapertures[0])*2)
        IM = self._xp.zeros((slope_len,Nmodes))

        pix_scale = self._pixel_size(lambdaInM=lambdaInM)
        alpha = pix_scale * self.oversampling * self.modN

        Nphot = self.photon_flux(starMagnitude=starMagnitude)

        if isinstance(amps, float):
            amps *= self._xp.ones(Nmodes)

        for i in range(Nmodes):
            amp = amps[i]
            mode_phase = self._xp.zeros(self.cmask.mask().shape)
            mode_phase[~self.cmask.mask()] = MM[i,:]*amp
            mode_phase = self._xp.reshape(mode_phase, self.cmask.mask().shape)
            input_field = self._xp.exp(1j*mode_phase) * self.cmask.asTransmissionValue()
            modulated_intensity = self.wfs.modulate(input_field, alpha, pix_scale)
            push_slope = self.ccd.compute_slopes(modulated_intensity, photon_flux = Nphot)/amp

            input_field = self._xp.conj(input_field)
            modulated_intensity = self.wfs.modulate(input_field, alpha, pix_scale)
            pull_slope = self.ccd.compute_slopes(modulated_intensity, photon_flux = Nphot)/amp

            IM[:,i] = (push_slope-pull_slope)/2

        U,S,Vt = self._xp.linalg.svd(IM, full_matrices=False)
        Rec = (Vt.T*1/S) @ U.T

        return IM, Rec

        