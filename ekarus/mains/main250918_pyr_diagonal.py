import xupy as xp
import numpy as np
masked_array = xp.masked_array

import matplotlib.pyplot as plt

# from ekarus.e2e.utils.image_utils import showZoomCenter, reshape_on_mask, myimshow
from ekarus.e2e.single_stage_ao_class import SingleStageAO
# import os.path as op
# import ekarus.e2e.utils.my_fits_package as myfits


def main(tn:str='example_pyr_diag'):

    ssao = SingleStageAO(tn)
    ssao.initialize_turbulence()
    KL, m2c = ssao.define_KL_modes(ssao.dm, zern_modes=5)
    ssao.pyr.set_modulation_angle(ssao.sc.modulationAngleInLambdaOverD)
    Rec, _ = ssao.compute_reconstructor(ssao.sc, KL, ssao.pyr.lambdaInM, amps=0.2)
    ssao.sc.load_reconstructor(Rec,m2c)

    diag_ssao = SingleStageAO(tn)
    diag_ssao.initialize_turbulence()
    diag_ssao.pyr.set_modulation_angle(ssao.sc.modulationAngleInLambdaOverD)
    Rec_diag, _ = diag_ssao.compute_reconstructor(ssao.sc, KL, ssao.pyr.lambdaInM,
                                             method='diagonal_slopes', amps=0.2, save_prefix='diag_')
    diag_ssao.sc.load_reconstructor(Rec_diag,m2c)

    raw_ssao = SingleStageAO(tn)
    raw_ssao.initialize_turbulence()
    raw_ssao.pyr.set_modulation_angle(ssao.sc.modulationAngleInLambdaOverD)
    Rec_raw, _ = raw_ssao.compute_reconstructor(ssao.sc, KL, ssao.pyr.lambdaInM,
                                             method='raw_intensity', amps=0.2, save_prefix='raw_')
    raw_ssao.sc.load_reconstructor(Rec_raw,m2c)

    print('Running the loop ...')
    
    sig2, input_sig2 = ssao.run_loop(ssao.pyr.lambdaInM, ssao.starMagnitude, save_prefix='')
    diag_sig2, _ = diag_ssao.run_loop(ssao.pyr.lambdaInM, ssao.starMagnitude, method='diagonal_slopes', save_prefix='diag_')
    raw_sig2, _ = raw_ssao.run_loop(ssao.pyr.lambdaInM, ssao.starMagnitude, method='raw_intensity', save_prefix='raw_')

    # Post-processing and plotting
    print('Plotting results ...')

    # IM = myfits.read_fits(op.join(ssao.savecalibpath,'IM.fits'))
    # IM_std = xp.std(IM,axis=0)
    # diag_IM = myfits.read_fits(op.join(ssao.savecalibpath,'diag_IM.fits'))
    # diag_IM_std = xp.std(diag_IM,axis=0)
    # if xp.on_gpu:
    #     IM_std = IM_std.get()
    #     diag_IM_std = diag_IM_std.get()
    # plt.figure()
    # plt.plot(IM_std,'-o',label='standard')
    # plt.plot(diag_IM_std,'-o',label='using diagonal')
    # plt.legend()
    # plt.grid()
    # plt.title('Interaction matrix standard deviation')

    lambdaRef = ssao.pyr.lambdaInM
    ssao.KL = KL
    ssao.plot_iteration(lambdaRef, frame_id=-1, save_prefix='')
    ssao.plot_iteration(lambdaRef, frame_id=-1, save_prefix='diag_')
    ssao.plot_iteration(lambdaRef, frame_id=-1, save_prefix='raw_')

    # cmask = ssao.cmask.get() if xp.on_gpu else ssao.cmask.copy()
    if xp.on_gpu: # Convert to numpy for plotting
        input_sig2 = input_sig2.get()
        sig2 = sig2.get()
        diag_sig2 = diag_sig2.get()
        raw_sig2 = raw_sig2.get()
        # rec_modes = rec_modes.get()

    tvec = xp.arange(ssao.Nits)*ssao.dt*1e+3
    tvec = xp.asnumpy(tvec)
    plt.figure()#figsize=(1.7*Nits/10,3))
    plt.plot(tvec,input_sig2,'-.',label='open loop')
    plt.plot(tvec,sig2,'-.',label='slopes')
    plt.plot(tvec,diag_sig2,'-.',label='slopes (including Sxy)')
    plt.plot(tvec,raw_sig2,'-.',label='raw intensity')
    plt.legend()
    plt.grid()
    plt.xlim([0.0,tvec[-1]])
    plt.xlabel('Time [ms]')
    plt.ylabel(r'$\sigma^2 [rad^2]$')
    plt.gca().set_yscale('log')

    _,S,_ = xp.linalg.svd(Rec,full_matrices=False)
    _,diag_S,_ = xp.linalg.svd(Rec_diag,full_matrices=False)
    _,raw_S,_ = xp.linalg.svd(Rec_raw,full_matrices=False)
    plt.figure()
    plt.plot(xp.asnumpy(S),'-o',label=f'slopes')
    plt.plot(xp.asnumpy(diag_S),'-x',label=f'slopes, including Sxy')
    plt.plot(xp.asnumpy(raw_S),'-x',label=f'raw intensity')
    plt.legend()
    plt.xlabel('Mode number')
    plt.grid()
    plt.title('Reconstructor singular values')

    # plt.figure()
    # plt.plot(tvec,rec_modes[:,:10],'-o',label='reference')
    # plt.plot(tvec,diag_rec_modes[:,:10],'-x',label='diagonal')
    # plt.legend()
    # plt.grid()
    # plt.xlim([0.0,tvec[-1]])
    # plt.xlabel('Time [ms]')
    # plt.ylabel('amplitude [m]')
    # plt.title('Reconstructor modes\n(first 10)')

    plt.show()

    return ssao

if __name__ == '__main__':
    ssao = main()