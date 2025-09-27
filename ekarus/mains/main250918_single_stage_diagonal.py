import xupy as xp
import numpy as np
masked_array = xp.masked_array

import matplotlib.pyplot as plt

from ekarus.e2e.utils.image_utils import showZoomCenter, reshape_on_mask, myimshow
from ekarus.e2e.single_stage_ao_class import SingleStageAO
import os.path as op

import ekarus.e2e.utils.my_fits_package as myfits


def main(tn:str='example_single_stage_diag', show:bool=False):

    ssao = SingleStageAO(tn)

    ssao.initialize_turbulence()

    KL, m2c = ssao.define_KL_modes(ssao.dm, zern_modes=5)
    ssao.pyr.set_modulation_angle(ssao.sc.modulationAngleInLambdaOverD)

    Rec, _ = ssao.compute_reconstructor(ssao.sc, KL, ssao.pyr.lambdaInM, amps=0.2)
    Rec_diag, _ = ssao.compute_reconstructor(ssao.sc, KL, ssao.pyr.lambdaInM, use_diagonal=True, amps=0.2)

    print('Running the loop ...')
    try:
        if ssao.recompute is True:
            raise FileNotFoundError('Recompute is True')        
        m2rad = 2 * xp.pi / ssao.pyr.lambdaInM
        masked_input_phases, _, masked_residual_phases, _, _, _ = ssao.load_telemetry_data()        
        _, _, diag_masked_residual_phases, _, _, _ = ssao.load_telemetry_data('diag_')
        
        residual_phases = xp.zeros([ssao.Nits,int(xp.sum(1-ssao.cmask))])
        input_phases = xp.zeros([ssao.Nits,int(xp.sum(1-ssao.cmask))])
        diag_residual_phases = xp.zeros([ssao.Nits,int(xp.sum(1-ssao.cmask))])
        # diag_input_phases = xp.zeros([ssao.Nits,int(xp.sum(1-ssao.cmask))])
        for i in range(ssao.Nits):
            diag_ma_res_phase = diag_masked_residual_phases[i,:,:]
            diag_residual_phases[i,:] = diag_ma_res_phase[~ssao.cmask]
            ma_res_phase = masked_residual_phases[i,:,:]
            residual_phases[i,:] = ma_res_phase[~ssao.cmask]
            ma_in_phase = masked_input_phases[i,:,:]
            input_phases[i,:] = ma_in_phase[~ssao.cmask]
        diag_sig2 = xp.std(diag_residual_phases * m2rad, axis=-1) ** 2
        sig2 = xp.std(residual_phases * m2rad, axis=-1) ** 2
        input_sig2 = xp.std(input_phases * m2rad, axis=-1) ** 2

    except:
        ssao.sc.load_reconstructor(Rec,m2c)
        sig2, input_sig2 = ssao.run_loop(ssao.pyr.lambdaInM, ssao.starMagnitude, save_prefix='')
        ssao.sc.load_reconstructor(Rec_diag,m2c)
        diag_sig2, _ = ssao.run_loop(ssao.pyr.lambdaInM, ssao.starMagnitude, use_diagonal=True, save_prefix='diag_')

    # Post-processing and plotting
    print('Plotting results ...')
    if show:
        subap_masks = xp.sum(ssao.sc._subaperture_masks,axis=0)
        plt.figure()
        myimshow(subap_masks,title='Subaperture masks')

        KL = myfits.read_fits(op.join(ssao.savecalibpath,'KLmodes.fits'))
        N=9
        plt.figure(figsize=(2*N,7))
        for i in range(N):
            plt.subplot(4,N,i+1)
            ssao.dm.plot_surface(KL[i,:],title=f'KL Mode {i}')
            plt.subplot(4,N,i+1+N)
            ssao.dm.plot_surface(KL[i+N,:],title=f'KL Mode {i+N}')
            plt.subplot(4,N,i+1+N*2)
            ssao.dm.plot_surface(KL[-i-1-N,:],title=f'KL Mode {xp.shape(KL)[0]-i-1-N}')
            plt.subplot(4,N,i+1+N*3)
            ssao.dm.plot_surface(KL[-i-1,:],title=f'KL Mode {xp.shape(KL)[0]-i-1}')

        IM = myfits.read_fits(op.join(ssao.savecalibpath,'IM.fits'))
        IM_std = xp.std(IM,axis=0)
        if xp.on_gpu:
            IM_std = IM_std.get()
        plt.figure()
        plt.plot(IM_std,'-o')
        plt.grid()
        plt.title('Interaction matrix standard deviation')

        screen = ssao.get_phasescreen_at_time(0)
        if xp.on_gpu:
            screen = screen.get()
        plt.figure()
        myimshow(masked_array(screen,ssao.cmask), title='Atmo screen [m]', cmap='RdBu')

    masked_input_phases, _, masked_residual_phases, detector_frames, rec_modes, dm_commands = ssao.load_telemetry_data()
    _, _, diag_ma_residual_phases, diag_det_frames, diag_rec_modes, diag_dm_commands = ssao.load_telemetry_data('diag_')

    oversampling = 8
    padding_len = int(ssao.cmask.shape[0]*(oversampling-1)/2)
    psf_mask = xp.pad(ssao.cmask, padding_len, mode='constant', constant_values=1)
    electric_field_amp = 1-psf_mask
    lambdaRef = ssao.pyr.lambdaInM

    last_res_phase = xp.array(masked_residual_phases[-1,:,:])
    residual_phase = last_res_phase[~ssao.cmask]
    input_phase = reshape_on_mask(residual_phase, psf_mask)
    electric_field = electric_field_amp * xp.exp(-1j*xp.asarray(input_phase*(2*xp.pi)/ssao.pyr.lambdaInM))
    field_on_focal_plane = xp.fft.fftshift(xp.fft.fft2(electric_field))
    psf = abs(field_on_focal_plane)**2

    diag_last_res_phase = xp.array(diag_ma_residual_phases[-1,:,:])
    diag_residual_phase = diag_last_res_phase[~ssao.cmask]
    input_phase = reshape_on_mask(diag_residual_phase, psf_mask)
    electric_field = electric_field_amp * xp.exp(-1j*xp.asarray(input_phase*(2*xp.pi)/ssao.pyr.lambdaInM))
    field_on_focal_plane = xp.fft.fftshift(xp.fft.fft2(electric_field))
    diag_psf = abs(field_on_focal_plane)**2

    cmask = ssao.cmask.get() if xp.on_gpu else ssao.cmask.copy()
    if xp.on_gpu: # Convert to numpy for plotting
        input_sig2 = input_sig2.get()
        sig2 = sig2.get()
        diag_sig2 = diag_sig2.get()
        rec_modes = rec_modes.get()

    plt.figure()
    pixelsPerMAS = ssao.pyr.lambdaInM/ssao.pupilSizeInM/ssao.pyr.oversampling*180/xp.pi*3600*1000
    plt.subplot(2,3,1)
    showZoomCenter(psf, pixelsPerMAS, shrink=0.8, \
        title = f'Corrected PSF\nSR = {xp.exp(-sig2[-1]):1.3f} @ {lambdaRef*1e+9:1.0f} [nm]',cmap='inferno') 
    plt.subplot(2,3,2)
    myimshow(detector_frames[-1], title = 'Detector frame', shrink=0.8)
    plt.subplot(2,3,3)
    ssao.dm.plot_position(dm_commands[-1])
    plt.title('Mirror command [m]')
    plt.axis('off')

    plt.subplot(2,3,4)
    showZoomCenter(diag_psf, pixelsPerMAS, shrink=0.8, \
        title = f'Corrected PSF\nSR = {xp.exp(-sig2[-1]):1.3f} @ {lambdaRef*1e+9:1.0f} [nm]',cmap='inferno') 
    plt.subplot(2,3,5)
    myimshow(diag_det_frames[-1], title = 'Detector frame', shrink=0.8)
    plt.subplot(2,3,6)
    ssao.dm.plot_position(diag_dm_commands[-1])
    plt.title('Mirror command [m]')
    plt.axis('off')

    tvec = xp.arange(ssao.Nits)*ssao.dt*1e+3
    tvec = tvec.get() if xp.on_gpu else tvec.copy()
    plt.figure()#figsize=(1.7*Nits/10,3))
    plt.plot(tvec,input_sig2,'-o',label='open loop')
    plt.plot(tvec,sig2,'-o',label='closed loop')
    plt.plot(tvec,diag_sig2,'-o',label='closed loop (diagonals)')
    plt.legend()
    plt.grid()
    plt.xlim([0.0,tvec[-1]])
    plt.xlabel('Time [ms]')
    plt.ylabel(r'$\sigma^2 [rad^2]$')
    plt.gca().set_yscale('log')

    plt.figure()
    plt.plot(tvec,rec_modes[:,:10],'-o',label='reference')
    plt.plot(tvec,diag_rec_modes[:,:10],'-x',label='diagonal')
    plt.legend()
    plt.grid()
    plt.xlim([0.0,tvec[-1]])
    plt.xlabel('Time [ms]')
    plt.ylabel('amplitude [m]')
    plt.title('Reconstructor modes\n(first 10)')
    
    plt.show()

    return ssao

if __name__ == '__main__':
    ssao = main(show=True)