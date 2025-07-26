from arte.types.mask import CircularMask
from arte.atmo.phase_screen_generator import PhaseScreenGenerator

from ekarus.e2e.utils.zernike_coefficients import create_field_from_zernike_coefficients
from ekarus.e2e.utils.kl_modes import make_modal_base_from_ifs_fft

import numpy as np

# TODO make this into a class, leaving out the atmosphere turbulence_generator


def turbulence_generator(lambdaInM, r0, L0, Nscreens, screenSizeInPixels, screenSizeInMeters, savepath:str= None):

    phs = PhaseScreenGenerator(screenSizeInPixels, screenSizeInMeters, \
                               outerScaleInMeters=L0, seed=42)
    
    phs.generate_normalized_phase_screens(Nscreens)
    phs.rescale_to(r0At500nm=r0)
    phs.get_in_radians_at(wavelengthInMeters=lambdaInM)

    if savepath is not None :
        phs.save_normalized_phase_screens(savepath)

    return phs._phaseScreens


def perform_loop_iteration(wfs, ccd, dm, Rec, m2c, input_phase, Npix, oversampling, lambdaInM, pupilSizeInM):

    pix2rad = lambdaInM/pupilSizeInM/oversampling
    alpha = lambdaInM/pupilSizeInM #/oversampling

    mask = CircularMask((oversampling * Npix, oversampling * Npix), maskRadius=Npix // 2)

    input_field = mask.asTransmissionValue() * np.exp(1j*input_phase)
    meas_intensity = wfs.modulate(input_field, alpha, pix2rad)
    slopes = ccd.compute_slopes(meas_intensity)
    modes = Rec @ slopes
    cmd = m2c @ modes

    ccd_image = ccd.resize_on_detector(meas_intensity)

    return cmd, ccd_image, (mask.mask()).astype(bool)



def calibrate_modes(wfs, ccd, MM, Npix, oversampling, lambdaInM, pupilSizeInM, amps:float = 0.1, xp=np):

    Nmodes = xp.shape(MM)[0]
    slope_len = int(xp.sum(1-ccd.subapertures[0])*2)
    IM = xp.zeros((slope_len,Nmodes))

    pix2rad = lambdaInM/pupilSizeInM/oversampling
    alpha = lambdaInM/pupilSizeInM #/oversampling

    mask = CircularMask((oversampling * Npix, oversampling * Npix), maskRadius=Npix // 2)

    if isinstance(amps, float):
        amps *= xp.ones(Nmodes)

    for i in range(Nmodes):
        amp = amps[i]
        mode_phase = xp.zeros(mask.mask().shape)
        mode_phase[~mask.mask()] = MM[i,:]*amp
        mode_phase = xp.reshape(mode_phase,mask.mask().shape)
        input_field = xp.exp(1j*mode_phase) * mask.asTransmissionValue()
        modulated_intensity = wfs.modulate(input_field, alpha, pix2rad)
        push_slope = ccd.compute_slopes(modulated_intensity)/amp

        input_field = np.conj(input_field)
        modulated_intensity = wfs.modulate(input_field, alpha, pix2rad)
        pull_slope = ccd.compute_slopes(modulated_intensity)/amp

        IM[:,i] = (push_slope-pull_slope)/2

    return IM



def define_subaperture_masks(wfs, ccd, Npix, oversampling, lambdaInM, pupilSizeInM, subapertureSizeInPixels):

    pix2rad = lambdaInM/pupilSizeInM/oversampling
    alpha = lambdaInM/pupilSizeInM #/oversampling

    mask = CircularMask((oversampling * Npix, oversampling * Npix), maskRadius=Npix // 2)
    piston =  create_field_from_zernike_coefficients(mask, 1, 1)

    modulated_intensity = wfs.modulate(piston, 10*alpha, pix2rad)
    ccd.define_subaperture_masks(modulated_intensity, Npix = subapertureSizeInPixels)

    return ccd.subapertures, modulated_intensity


def make_KL_modal_base_from_dm(dm, Npix, oversampling, pupilSizeInM, r0, L0, zern2remove:int = 5):
    
    mask = CircularMask((oversampling * Npix, oversampling * Npix), maskRadius=Npix // 2)

    KL, m2c, singular_values = make_modal_base_from_ifs_fft(1-dm.mask, pupilSizeInM, \
    dm.IFF.T, r0, L0, zern_modes=zern2remove, zern_mask = mask, oversampling=oversampling, verbose = True)

    return KL, m2c





