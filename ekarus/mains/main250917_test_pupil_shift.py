import xupy as xp
import numpy as np
# from numpy.ma import masked_array

import matplotlib.pyplot as plt

from ekarus.e2e.utils.image_utils import myimshow
from ekarus.e2e.pupil_shift_ao_class import PupilShift

# import os.path as op
# import ekarus.e2e.utils.my_fits_package as myfits


def main(tn:str='example_pupil_shift', pupilPixelShift:float=0.2, it_ss:int=100):

    ssao = PupilShift(tn)

    ssao.initialize_turbulence()

    KL, m2c = ssao.define_KL_modes(ssao.dm, zern_modes=5)
    ssao.pyr.set_modulation_angle(ssao.sc.modulationAngleInLambdaOverD)
    Rec, _ = ssao.compute_reconstructor(ssao.sc, KL, ssao.pyr.lambdaInM, amps=0.2)
    ssao.sc.load_reconstructor(Rec,m2c)

    print('Running the loop ...')
    sig2, input_sig2 = ssao.run_loop(ssao.pyr.lambdaInM, ssao.starMagnitude, save_prefix='')


    pyr2det = ssao.pupilSizeInPixels/max(ssao.ccd.detector_shape) #*ssao.pyr.oversampling
    wedgeAmp = pupilPixelShift * pyr2det
    subap_masks = xp.sum(ssao.sc._subaperture_masks,axis=0)

    print('Testing pupil shifts before DM')
    phi = xp.linspace(0.0, xp.pi/2, 5)
    sig_beforeDM = xp.zeros([len(phi),ssao.Nits])
    for k in range(len(phi)):
        pup_ssao = PupilShift(tn)
        pup_ssao.initialize_turbulence()
        pup_ssao.pyr.set_modulation_angle(ssao.sc.modulationAngleInLambdaOverD)
        pup_ssao.sc.load_reconstructor(Rec,m2c)
        wedgeX = wedgeAmp * xp.cos(phi[k])
        wedgeY = wedgeAmp * xp.sin(phi[k])
        print(f'Now applying a shift of ({wedgeX:1.2f},{wedgeY:1.2f}) pixels')
        wedgeShift = (wedgeX, wedgeY)
        mmWedgeAmp = pupilPixelShift/ssao.dm.pixel_scale*pyr2det*1e+3
        save_str = f'{mmWedgeAmp*1e+3:1.0f}um_ang{phi[k]*180/xp.pi:1.0f}_beforeDM_'
        sig_beforeDM[k,:], _ = pup_ssao.run_loop(ssao.pyr.lambdaInM, ssao.starMagnitude, tilt_before_DM=wedgeShift, save_prefix=save_str)
        # pup_ssao.plot_iteration(lambdaRef, frame_id=-1, save_prefix=save_str)
        if wedgeAmp>1.0:
            ccd_frame = pup_ssao.ccd.last_frame
            plt.figure()
            myimshow(ccd_frame/xp.max(ccd_frame)-subap_masks,cmap='twilight',title=f'Angle: {phi[k]*180/xp.pi:1.0f}\nBefore DM')
            # raise ValueError('Stop')

    print('Testing pupil shifts after DM')
    sig_afterDM = xp.zeros([len(phi),ssao.Nits])
    for k in range(len(phi)):
        pup_ssao = PupilShift(tn)
        pup_ssao.initialize_turbulence()
        pup_ssao.pyr.set_modulation_angle(ssao.sc.modulationAngleInLambdaOverD)
        pup_ssao.sc.load_reconstructor(Rec,m2c)
        wedgeX = wedgeAmp * xp.cos(phi[k])
        wedgeY = wedgeAmp * xp.sin(phi[k])
        print(f'Now applying a shift of ({wedgeX:1.2f},{wedgeY:1.2f}) pixels')
        wedgeShift = (wedgeX, wedgeY)
        mmWedgeAmp = pupilPixelShift/ssao.dm.pixel_scale*pyr2det*1e+3
        save_str = f'{mmWedgeAmp*1e+3:1.0f}um_ang{phi[k]*180/xp.pi:1.0f}_afterDM_'
        sig_afterDM[k,:], _ = pup_ssao.run_loop(ssao.pyr.lambdaInM, ssao.starMagnitude, tilt_after_DM=wedgeShift, save_prefix=save_str)
        # pup_ssao.plot_iteration(lambdaRef, frame_id=-1, save_prefix=save_str)
        if wedgeAmp>1.0:
            ccd_frame = pup_ssao.ccd.last_frame
            plt.figure()
            myimshow(ccd_frame/xp.max(ccd_frame)-subap_masks,cmap='twilight',title=f'Angle: {phi[k]*180/xp.pi:1.0f}\nAfter DM')

    # Post-processing and plotting
    print('Plotting results ...')
    lambdaRef = ssao.pyr.lambdaInM
    ssao.plot_iteration(lambdaRef, frame_id=-1, save_prefix='')
    ssao.plot_iteration(lambdaRef, frame_id=-1, save_prefix=save_str)

    # cmask = ssao.cmask.get() if xp.on_gpu else ssao.cmask.copy()
    if xp.on_gpu: # Convert to numpy for plotting
        input_sig2 = input_sig2.get()
        sig2 = sig2.get()
        sig_beforeDM = sig_beforeDM.get()
        sig_afterDM = sig_afterDM.get()
        # rec_modes = rec_modes.get()

    tvec = xp.arange(ssao.Nits)*ssao.dt*1e+3
    tvec = xp.asnumpy(tvec)

    plt.figure()
    plt.plot(tvec,input_sig2,label=f'atmo')
    sr_ss = np.mean(np.exp(-sig2[it_ss:]))
    plt.plot(tvec,sig2,label=f'reference, SR={sr_ss:1.2f}')
    for k,ang in enumerate(phi):
        sr_ss = np.mean(np.exp(-sig_beforeDM[k,it_ss:]))
        plt.plot(tvec,sig_beforeDM[k],'-.',label=f'ang={ang*180/xp.pi:1.0f}, SR={sr_ss:1.2f}')
    plt.legend()
    plt.grid()
    plt.xlim([0.0,tvec[-1]])
    plt.xlabel('Time [ms]')
    plt.ylabel(r'$\sigma^2 [rad^2]$')
    plt.gca().set_yscale('log')
    plt.title(f'{wedgeAmp:1.2f} subaperture tilt before DM\nSR @ {lambdaRef*1e+9:1.0f} [nm]')

    plt.figure()
    plt.plot(tvec,input_sig2,label=f'atmo')
    sr_ss = np.mean(np.exp(-sig2[it_ss:]))
    plt.plot(tvec,sig2,label=f'reference, SR={sr_ss:1.2f}')
    for k,ang in enumerate(phi):
        sr_ss = np.mean(np.exp(-sig_afterDM[k,it_ss:]))
        plt.plot(tvec,sig_afterDM[k],'-.',label=f'ang={ang*180/xp.pi:1.0f}, SR={sr_ss:1.2f}')
    plt.legend()
    plt.grid()
    plt.xlim([0.0,tvec[-1]])
    plt.xlabel('Time [ms]')
    plt.ylabel(r'$\sigma^2 [rad^2]$')
    plt.gca().set_yscale('log')
    plt.title(f'{wedgeAmp:1.2f} subaperture tilt after DM\nSR @ {lambdaRef*1e+9:1.0f} [nm]')

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