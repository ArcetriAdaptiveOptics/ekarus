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
    diag_ssao.pyr.set_modulation_angle(diag_ssao.sc.modulationAngleInLambdaOverD)
    Rec_diag, _ = diag_ssao.compute_reconstructor(ssao.sc, KL, ssao.pyr.lambdaInM,
                                             use_diagonal=True, amps=0.2, save_prefix='diag_')
    diag_ssao.sc.load_reconstructor(Rec_diag,m2c)

    print('Running the loop ...')
    
    sig2, input_sig2 = ssao.run_loop(ssao.pyr.lambdaInM, ssao.starMagnitude, save_prefix='')
    diag_sig2, _ = diag_ssao.run_loop(ssao.pyr.lambdaInM, ssao.starMagnitude, use_diagonal=True, save_prefix='diag_')

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
    ssao.plot_iteration(lambdaRef, frame_id=-1, save_prefix='')
    diag_ssao.plot_iteration(lambdaRef, frame_id=-1, save_prefix='diag_')

    # cmask = ssao.cmask.get() if xp.on_gpu else ssao.cmask.copy()
    if xp.on_gpu: # Convert to numpy for plotting
        input_sig2 = input_sig2.get()
        sig2 = sig2.get()
        diag_sig2 = diag_sig2.get()
        # rec_modes = rec_modes.get()

    tvec = xp.arange(ssao.Nits)*ssao.dt*1e+3
    tvec = xp.asnumpy(tvec)
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

    _,S,_ = xp.linalg.svd(Rec,full_matrices=False)
    _,diag_S,_ = xp.linalg.svd(Rec_diag,full_matrices=False)
    if xp.on_gpu:
        S = S.get()
        diag_S = diag_S.get()
    plt.figure()
    plt.plot(S,'-o',label=f'Sx, Sy')
    plt.plot(diag_S,'-x',label=f'Sx, Sy, Sxy')
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