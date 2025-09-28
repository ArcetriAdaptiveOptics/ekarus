import xupy as xp
import numpy as np
masked_array = xp.masked_array

import matplotlib.pyplot as plt

from ekarus.e2e.utils.image_utils import showZoomCenter, reshape_on_mask, myimshow
from ekarus.e2e.test_pupil_shift_ao_class import PupilShift

# import os.path as op
# import ekarus.e2e.utils.my_fits_package as myfits


def main(tn:str='example_pupil_shift', pupilPixelShift:float=0.2): 

    ssao = PupilShift(tn)

    ssao.initialize_turbulence()

    KL, m2c = ssao.define_KL_modes(ssao.dm, zern_modes=5)
    ssao.pyr.set_modulation_angle(ssao.sc.modulationAngleInLambdaOverD)
    Rec, _ = ssao.compute_reconstructor(ssao.sc, KL, ssao.pyr.lambdaInM, amps=0.2)
    ssao.sc.load_reconstructor(Rec,m2c)

    print('Running the loop ...')
    try:
        if ssao.recompute is True:
            raise FileNotFoundError('Recompute is True')
        masked_input_phases, _, masked_residual_phases, _, _, _ = ssao.load_telemetry_data()
        m2rad = 2 * xp.pi / ssao.pyr.lambdaInM
        residual_phases = xp.zeros([ssao.Nits,int(xp.sum(1-ssao.cmask))])
        input_phases = xp.zeros([ssao.Nits,int(xp.sum(1-ssao.cmask))])
        for i in range(ssao.Nits):
            ma_res_phase = masked_residual_phases[i,:,:]
            residual_phases[i,:] = ma_res_phase[~ssao.cmask]
            ma_in_phase = masked_input_phases[i,:,:]
            input_phases[i,:] = ma_in_phase[~ssao.cmask]
        sig2 = xp.std(residual_phases * m2rad, axis=-1) ** 2
        input_sig2 = xp.std(input_phases * m2rad, axis=-1) ** 2
    except:
        sig2, input_sig2 = ssao.run_loop(ssao.pyr.lambdaInM, ssao.starMagnitude, save_prefix='')

    pyr2det = ssao.pupilSizeInPixels*ssao.pyr.oversampling/max(ssao.ccd.detector_shape)
    wedgeAmp = pupilPixelShift * pyr2det
    subap_masks = xp.sum(ssao.sc._subaperture_masks,axis=0)

    print('Testing pupil shifts before DM')
    phi = xp.linspace(0, xp.pi/2, 3)
    sig_beforeDM = xp.zeros([len(phi)+2,ssao.Nits])
    sig_beforeDM[1,:] = sig2
    sig_beforeDM[0,:] = input_sig2
    for k in range(len(phi)):
        wedgeX = wedgeAmp * xp.cos(phi[k])
        wedgeY = wedgeAmp * xp.sin(phi[k])
        print(f'Now applying a shift of ({wedgeX:1.2f},{wedgeY:1.2f})')
        wedgeShift = (wedgeX, wedgeY)
        mmWedgeAmp = pupilPixelShift/ssao.dm.pixel_scale*pyr2det*1e+3
        save_str = f'wedge{mmWedgeAmp:1.0f}_ang{phi[k]*180/xp.pi:1.0f}_beforeDM_'
        sig_beforeDM[k+2,:], _ = ssao.run_loop(ssao.pyr.lambdaInM, ssao.starMagnitude, tilt_before_DM=wedgeShift, save_prefix=save_str)
        ccd_frame = ssao.ccd.last_frame
        plt.figure()
        myimshow(ccd_frame/xp.max(ccd_frame)-2*subap_masks,title=f'Angle: {phi[k]*180/xp.pi:1.0f}\nAfter DM')

    print('Testing pupil shifts after DM')
    sig_afterDM = sig_beforeDM.copy()
    for k in range(len(phi)):
        wedgeX = wedgeAmp * xp.cos(phi[k])
        wedgeY = wedgeAmp * xp.sin(phi[k])
        print(f'Now applying a shift of ({wedgeX:1.2f},{wedgeY:1.2f})')
        wedgeShift = (wedgeX, wedgeY)
        mmWedgeAmp = pupilPixelShift/ssao.dm.pixel_scale*pyr2det*1e+3
        save_str = f'wedge{mmWedgeAmp:1.0f}_ang{phi[k]*180/xp.pi:1.0f}_afterDM_'
        sig_afterDM[k+2,:], _ = ssao.run_loop(ssao.pyr.lambdaInM, ssao.starMagnitude, tilt_after_DM=wedgeShift, save_prefix=save_str)
        ccd_frame = ssao.ccd.last_frame
        plt.figure()
        myimshow(ccd_frame/xp.max(ccd_frame)-2*subap_masks,title=f'Angle: {phi[k]*180/xp.pi:1.0f}\nBefore DM')

    # Post-processing and plotting
    print('Plotting results ...')

    masked_input_phases, _, masked_residual_phases, detector_frames, rec_modes, dm_commands = ssao.load_telemetry_data()

    last_res_phase = xp.array(masked_residual_phases[-1,:,:])
    residual_phase = last_res_phase[~ssao.cmask]

    oversampling = 8
    padding_len = int(ssao.cmask.shape[0]*(oversampling-1)/2)
    psf_mask = xp.pad(ssao.cmask, padding_len, mode='constant', constant_values=1)
    electric_field_amp = 1-psf_mask
    input_phase = reshape_on_mask(residual_phase, psf_mask)
    electric_field = electric_field_amp * xp.exp(-1j*xp.asarray(input_phase*(2*xp.pi)/ssao.pyr.lambdaInM))
    field_on_focal_plane = xp.fft.fftshift(xp.fft.fft2(electric_field))
    psf = abs(field_on_focal_plane)**2
    # psf = abs(ssao.pyr.field_on_focal_plane)**2

    cmask = ssao.cmask.get() if xp.on_gpu else ssao.cmask.copy()
    if xp.on_gpu: # Convert to numpy for plotting
        input_sig2 = input_sig2.get()
        sig2 = sig2.get()
        sig_beforeDM = sig_beforeDM.get()
        sig_afterDM = sig_afterDM.get()
        rec_modes = rec_modes.get()

    lambdaRef = ssao.pyr.lambdaInM

    plt.figure(figsize=(9,9))
    plt.subplot(2,2,1)
    myimshow(masked_array(masked_input_phases[-1],cmask), \
        title=f'Atmosphere phase [m]\nSR = {xp.exp(-input_sig2[-1]):1.3f} @ {lambdaRef*1e+9:1.0f} [nm]',\
        cmap='RdBu',shrink=0.8)
    plt.axis('off')

    pixelsPerMAS = ssao.pyr.lambdaInM/ssao.pupilSizeInM/ssao.pyr.oversampling*180/xp.pi*3600*1000
    plt.subplot(2,2,2)
    showZoomCenter(psf, pixelsPerMAS, shrink=0.8, \
        title = f'Corrected PSF\nSR = {xp.exp(-sig2[-1]):1.3f} @ {lambdaRef*1e+9:1.0f} [nm]',cmap='inferno') 

    plt.subplot(2,2,3)
    myimshow(detector_frames[-1], title = 'Detector frame', shrink=0.8)

    plt.subplot(2,2,4)
    ssao.dm.plot_position(dm_commands[-1])
    plt.title('Mirror command [m]')
    plt.axis('off')

    tvec = xp.arange(ssao.Nits)*ssao.dt*1e+3
    tvec = tvec.get() if xp.on_gpu else tvec.copy()
    # plt.figure()#figsize=(1.7*Nits/10,3))
    # plt.plot(tvec,input_sig2,'-o',label='open loop')
    # plt.plot(tvec,sig2,'-o',label='closed loop')
    # plt.legend()
    # plt.grid()
    # plt.xlim([0.0,tvec[-1]])
    # plt.xlabel('Time [ms]')
    # plt.ylabel(r'$\sigma^2 [rad^2]$')
    # plt.gca().set_yscale('log')

    plt.figure()
    plt.plot(tvec,sig_beforeDM[0],label=f'atmo')
    sr_ss = np.mean(np.exp(-sig_beforeDM[1,100:]))
    plt.plot(tvec,sig_beforeDM[1],label=f'reference, SR={sr_ss:1.2f}')
    for k,ang in enumerate(phi):
        sr_ss = np.mean(np.exp(-sig_beforeDM[k+2,100:]))
        plt.plot(tvec,sig_beforeDM[k+2],label=f'ang={ang*180/xp.pi:1.1f}, SR={sr_ss:1.2f}')
    plt.legend()
    plt.grid()
    plt.xlim([0.0,tvec[-1]])
    plt.xlabel('Time [ms]')
    plt.ylabel(r'$\sigma^2 [rad^2]$')
    plt.gca().set_yscale('log')
    plt.title(f'{pupilPixelShift:1.2f} subaperture tilt before DM\nSR @ {lambdaRef*1e+9:1.0f} [nm]')

    plt.figure()
    plt.plot(tvec,sig_afterDM[0],label=f'atmo')
    sr_ss = np.mean(np.exp(-sig_afterDM[1,100:]))
    plt.plot(tvec,sig_afterDM[1],label=f'reference, SR={sr_ss:1.2f}')
    for k,ang in enumerate(phi):
        sr_ss = np.mean(np.exp(-sig_afterDM[k+2,100:]))
        plt.plot(tvec,sig_afterDM[k+2],label=f'ang={ang*180/xp.pi:1.1f}, SR={sr_ss:1.2f}')
    plt.legend()
    plt.grid()
    plt.xlim([0.0,tvec[-1]])
    plt.xlabel('Time [ms]')
    plt.ylabel(r'$\sigma^2 [rad^2]$')
    plt.gca().set_yscale('log')
    plt.title(f'{pupilPixelShift:1.2f} subaperture tilt after DM\nSR @ {lambdaRef*1e+9:1.0f} [nm]')

    # plt.figure()
    # plt.plot(tvec,rec_modes[:,:10],'-o')
    # plt.grid()
    # plt.xlim([0.0,tvec[-1]])
    # plt.xlabel('Time [ms]')
    # plt.ylabel('amplitude [m]')
    # plt.title('Reconstructor modes\n(first 10)')
    
    plt.show()

    return ssao

if __name__ == '__main__':
    ssao = main()