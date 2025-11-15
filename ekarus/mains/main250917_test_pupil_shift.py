import xupy as xp
import numpy as np
# from numpy.ma import masked_array

import matplotlib.pyplot as plt

from ekarus.e2e.utils.image_utils import myimshow
from ekarus.e2e.pupil_shift_ao_class import PupilShift

# import os.path as op
# import ekarus.e2e.utils.my_fits_package as myfits


def main(tn:str='example_pupil_shift', pupilShiftsInPixel:list=[0.2], shiftAnglesInDegrees:list=[0.0], ss_it:int=200):

    pupilShiftsInPixel = xp.array(pupilShiftsInPixel)
    shiftAngles = xp.array(shiftAnglesInDegrees)*xp.pi/180

    ssao = PupilShift(tn)
    ssao.initialize_turbulence()
    KL, m2c = ssao.define_KL_modes(ssao.dm, zern_modes=5)
    ssao.pyr.set_modulation_angle(ssao.sc.modulationAngleInLambdaOverD)
    Rec, IM = ssao.compute_reconstructor(ssao.sc, KL, ssao.pyr.lambdaInM, ampsInM=50e-9)
    ssao.sc.load_reconstructor(IM,m2c)

    print('Running the loop ...')
    sig2, input_sig2 = ssao.run_loop(ssao.pyr.lambdaInM, ssao.starMagnitude, save_prefix='')

    # pyr2det = ssao.pupilSizeInPixels/max(ssao.ccd.detector_shape) #*ssao.pyr.oversampling
    # wedgeAmp = pupilPixelShift * pyr2det
    subap_masks = xp.sum(ssao.sc._subaperture_masks,axis=0)

    print('Testing pupil shifts before DM')
    sig2_beforeDM = xp.zeros([len(shiftAngles),len(pupilShiftsInPixel),ssao.Nits])
    for i,angle in enumerate(shiftAngles):
        for j,pupilPixelShift in enumerate(pupilShiftsInPixel):
            pup_ssao = PupilShift(tn)
            pup_ssao.initialize_turbulence()
            pup_ssao.pyr.set_modulation_angle(ssao.sc.modulationAngleInLambdaOverD)
            pup_ssao.sc.load_reconstructor(IM,m2c)
            wedgeX = pupilPixelShift * xp.cos(angle)
            wedgeY = pupilPixelShift * xp.sin(angle)
            print(f'Now applying a shift of ({wedgeX:1.2f},{wedgeY:1.2f}) pixels')
            wedgeShift = (wedgeX, wedgeY)
            save_str = f'{pupilPixelShift*1e+3:1.0f}mpix_ang{angle*180/xp.pi:1.0f}_beforeDM_'
            sig2_beforeDM[i,j,:], _ = pup_ssao.run_loop(ssao.pyr.lambdaInM, ssao.starMagnitude, tilt_before_DM=wedgeShift, save_prefix=save_str)
            # pup_ssao.plot_iteration(lambdaRef, frame_id=-1, save_prefix=save_str)
            if pupilPixelShift>1.0:
                ccd_frame = pup_ssao.ccd.last_frame
                plt.figure()
                myimshow(ccd_frame/xp.max(ccd_frame)-subap_masks,cmap='twilight',title=f'Angle: {angle*180/xp.pi:1.0f}\nBefore DM')
                # raise ValueError('Stop')

    print('Testing pupil shifts after DM')
    sig2_afterDM = xp.zeros([len(shiftAngles),len(pupilShiftsInPixel),ssao.Nits])
    for i,angle in enumerate(shiftAngles):
        for j,pupilPixelShift in enumerate(pupilShiftsInPixel):
            pup_ssao = PupilShift(tn)
            pup_ssao.initialize_turbulence()
            pup_ssao.pyr.set_modulation_angle(ssao.sc.modulationAngleInLambdaOverD)
            pup_ssao.sc.load_reconstructor(Rec,m2c)
            wedgeX = pupilPixelShift * xp.cos(angle)
            wedgeY = pupilPixelShift * xp.sin(angle)
            print(f'Now applying a shift of ({wedgeX:1.2f},{wedgeY:1.2f}) pixels')
            wedgeShift = (wedgeX, wedgeY)
            save_str = f'{pupilPixelShift*1e+3:1.0f}mpix_ang{angle*180/xp.pi:1.0f}_afterDM_'
            sig2_afterDM[i,j,:], _ = pup_ssao.run_loop(ssao.pyr.lambdaInM, ssao.starMagnitude, tilt_after_DM=wedgeShift, save_prefix=save_str)
            # pup_ssao.plot_iteration(lambdaRef, frame_id=-1, save_prefix=save_str)
            if pupilPixelShift>1.0:
                ccd_frame = pup_ssao.ccd.last_frame
                plt.figure()
                myimshow(ccd_frame/xp.max(ccd_frame)-subap_masks,cmap='twilight',title=f'Angle: {angle*180/xp.pi:1.0f}\nAfter DM')
                # raise ValueError('Stop')

    # Post-processing and plotting
    print('Plotting results ...')
    lambdaRef = ssao.pyr.lambdaInM
    ssao.plot_iteration(lambdaRef, frame_id=-1, save_prefix='')
    ssao.plot_iteration(lambdaRef, frame_id=-1, save_prefix=save_str)


    SR_ref = xp.mean(xp.exp(-sig2[-ss_it:]))
    SR_beforeDM = xp.mean(xp.exp(-sig2_beforeDM[:,:,-ss_it:]),axis=-1)
    SR_afterDM = xp.mean(xp.exp(-sig2_afterDM[:,:,-ss_it:]),axis=-1)

    # Convert to numpy for plotting
    input_sig2 = xp.asnumpy(input_sig2)
    sig2 = xp.asnumpy(sig2)
    sig2_beforeDM = xp.asnumpy(sig2_beforeDM)
    sig2_afterDM = xp.asnumpy(sig2_afterDM)
    # rec_modes = rec_modes.get()

    tvec = xp.arange(ssao.Nits)*ssao.dt*1e+3
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


    ssao.SR_ref = SR_ref
    ssao.SR_beforeDM = SR_beforeDM
    ssao.SR_afterDM = SR_afterDM
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