import xupy as xp
import numpy as np
# from numpy.ma import masked_array

import matplotlib.pyplot as plt

from ekarus.e2e.utils.image_utils import myimshow
from ekarus.e2e.pupil_shift_ao_class import PupilShift

# import os.path as op
# import ekarus.e2e.utils.my_fits_package as myfits


def main(tn:str='example_pupil_shift', 
        pupilShiftsInPixel:list=[0.1,0.2,0.3], 
        shiftAnglesInDegrees:list=[0,45,90],
        atmo_tn='ekarus_5cm',
        it_ss:int = 1000):

    pupilShiftsInPixel = xp.array(pupilShiftsInPixel)
    shiftAngles = xp.array(shiftAnglesInDegrees)*xp.pi/180

    pup_ssao = PupilShift(tn)
    pup_ssao.initialize_turbulence(atmo_tn)
    KL, m2c = pup_ssao.define_KL_modes(pup_ssao.dm, zern_modes=2)
    Rec, IM = pup_ssao.compute_reconstructor(pup_ssao.sc, KL, pup_ssao.pyr.lambdaInM, ampsInM=50e-9)
    pup_ssao.sc.load_reconstructor(IM,m2c)

    print('Running the loop ...')
    sig2, input_sig2 = pup_ssao.run_loop(pup_ssao.pyr.lambdaInM, pup_ssao.starMagnitude, save_prefix='')

    # pyr2det = pup_ssao.pupilSizeInPixels/max(pup_ssao.ccd.detector_shape) #*pup_ssao.pyr.oversampling
    # wedgeAmp = pupilPixelShift * pyr2det
    subap_masks = xp.sum(pup_ssao.sc._roi_masks,axis=0)

    ss_it = max(200,int(pup_ssao.Nits//2))

    print('Testing pupil shifts before DM')
    sig2_beforeDM = xp.zeros([len(shiftAngles),len(pupilShiftsInPixel),pup_ssao.Nits])
    for i,angle in enumerate(shiftAngles):
        for j,pupilPixelShift in enumerate(pupilShiftsInPixel):
            wedgeX = pupilPixelShift * xp.cos(angle)
            wedgeY = pupilPixelShift * xp.sin(angle)
            print(f'Now applying a shift of ({wedgeX:1.2f},{wedgeY:1.2f}) pixels')
            wedgeShift = (wedgeX, wedgeY)
            save_str = f'{pupilPixelShift*1e+3:1.0f}mpix_ang{angle*180/xp.pi:1.0f}_beforeDM_'
            sig2_beforeDM[i,j,:], _ = pup_ssao.run_loop(pup_ssao.pyr.lambdaInM, pup_ssao.starMagnitude, tilt_before_DM=wedgeShift, save_prefix=save_str)
            # pup_ssao.plot_iteration(lambdaRef, frame_id=-1, save_prefix=save_str)
            if pupilPixelShift>1.0:
                ccd_frame = pup_ssao.ccd.last_frame
                plt.figure()
                myimshow(ccd_frame/xp.max(ccd_frame)-subap_masks,cmap='twilight',title=f'Angle: {angle*180/xp.pi:1.0f}\nBefore DM')
                # raise ValueError('Stop')

    print('Testing pupil shifts after DM')
    sig2_afterDM = xp.zeros([len(shiftAngles),len(pupilShiftsInPixel),pup_ssao.Nits])
    for i,angle in enumerate(shiftAngles):
        for j,pupilPixelShift in enumerate(pupilShiftsInPixel):
            wedgeX = pupilPixelShift * xp.cos(angle)
            wedgeY = pupilPixelShift * xp.sin(angle)
            print(f'Now applying a shift of ({wedgeX:1.2f},{wedgeY:1.2f}) pixels')
            wedgeShift = (wedgeX, wedgeY)
            save_str = f'{pupilPixelShift*1e+3:1.0f}mpix_ang{angle*180/xp.pi:1.0f}_afterDM_'
            sig2_afterDM[i,j,:], _ = pup_ssao.run_loop(pup_ssao.pyr.lambdaInM, pup_ssao.starMagnitude, tilt_after_DM=wedgeShift, save_prefix=save_str)
            # pup_ssao.plot_iteration(lambdaRef, frame_id=-1, save_prefix=save_str)
            if pupilPixelShift>1.0:
                ccd_frame = pup_ssao.ccd.last_frame
                plt.figure()
                myimshow(ccd_frame/xp.max(ccd_frame)-subap_masks,cmap='twilight',title=f'Angle: {angle*180/xp.pi:1.0f}\nAfter DM')
                # raise ValueError('Stop')

    # Post-processing and plotting
    print('Plotting results ...')
    lambdaRef = pup_ssao.pyr.lambdaInM
    pup_ssao.plot_iteration(lambdaRef, frame_id=-1, save_prefix='')
    pup_ssao.plot_iteration(lambdaRef, frame_id=-1, save_prefix=save_str)

    if it_ss is not None:
        ss_it = it_ss

    SR_ref = xp.mean(xp.exp(-sig2[-ss_it:]))
    SR_beforeDM = xp.mean(xp.exp(-sig2_beforeDM[:,:,-ss_it:]),axis=-1)
    SR_afterDM = xp.mean(xp.exp(-sig2_afterDM[:,:,-ss_it:]),axis=-1)

    pup_ssao.SRr_beforeDM = SR_beforeDM/SR_ref
    pup_ssao.SRr_afterDM = SR_afterDM/SR_ref

    # Convert to numpy for plotting
    input_sig2 = xp.asnumpy(input_sig2)
    sig2 = xp.asnumpy(sig2)
    sig2_beforeDM = xp.asnumpy(sig2_beforeDM)
    sig2_afterDM = xp.asnumpy(sig2_afterDM)
    # rec_modes = rec_modes.get()

    tvec = xp.arange(pup_ssao.Nits)*pup_ssao.dt*1e+3
    tvec = xp.asnumpy(tvec)

    for j,shiftAmp in enumerate(pupilShiftsInPixel):
        plt.figure()
        plt.plot(tvec,input_sig2,label=f'atmo')
        plt.plot(tvec,sig2,label=f'reference, SR={SR_ref.get():1.2f}')
        for i,ang in enumerate(shiftAngles):
            sr_ss = SR_beforeDM[i,j].get()
            plt.plot(tvec,sig2_beforeDM[i,j,:],'-.',label=f'ang={ang*180/xp.pi:1.0f}, SR={sr_ss:1.2f}')
        plt.legend()
        plt.grid()
        plt.xlim([0.0,tvec[-1]])
        plt.xlabel('Time [ms]')
        plt.ylabel(r'$\sigma^2 [rad^2]$')
        plt.gca().set_yscale('log')
        plt.title(f'{shiftAmp:1.2f} subaperture tilt before DM\nSR @ {lambdaRef*1e+9:1.0f} [nm]')

    for j,shiftAmp in enumerate(pupilShiftsInPixel):
        plt.figure()
        plt.plot(tvec,input_sig2,label=f'atmo')
        plt.plot(tvec,sig2,label=f'reference, SR={SR_ref.get():1.2f}')
        for i,ang in enumerate(shiftAngles):
            sr_ss = SR_afterDM[i,j].get()
            plt.plot(tvec,sig2_afterDM[i,j,:],'-.',label=f'ang={ang*180/xp.pi:1.0f}, SR={sr_ss:1.2f}')
        plt.legend()
        plt.grid()
        plt.xlim([0.0,tvec[-1]])
        plt.xlabel('Time [ms]')
        plt.ylabel(r'$\sigma^2 [rad^2]$')
        plt.gca().set_yscale('log')
        plt.title(f'{shiftAmp:1.2f} subaperture tilt after DM\nSR @ {lambdaRef*1e+9:1.0f} [nm]')

    # plt.figure()
    # plt.plot(tvec,rec_modes[:,:10],'-o')
    # plt.grid()
    # plt.xlim([0.0,tvec[-1]])
    # plt.xlabel('Time [ms]')
    # plt.ylabel('amplitude [m]')
    # plt.title('Reconstructor modes\n(first 10)')

    plt.show()

    return pup_ssao

if __name__ == '__main__':
    pup_ssao = main()