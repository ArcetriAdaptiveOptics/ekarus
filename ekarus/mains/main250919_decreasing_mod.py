import xupy as xp
# import numpy as np
masked_array = xp.masked_array

import matplotlib.pyplot as plt

# from ekarus.e2e.utils.image_utils import showZoomCenter, myimshow #, reshape_on_mask
from ekarus.e2e.single_stage_ao_class import SingleStageAO
from ekarus.e2e.changing_gain_ao_class import ChangingGainSSAO

# import os.path as op
# import ekarus.e2e.utils.my_fits_package as myfits


def main(tn:str='example_decreasing_mod'):

    ssao = SingleStageAO(tn)
    ssao.initialize_turbulence()
    KL, m2c = ssao.define_KL_modes(ssao.dm, zern_modes=5)
    ssao.pyr.set_modulation_angle(ssao.sc.modulationAngleInLambdaOverD)
    Rec, IM = ssao.compute_reconstructor(ssao.sc, KL, ssao.pyr.lambdaInM, amps=0.2)
    ssao.sc.load_reconstructor(IM,m2c)
    sig2, input_sig2 = ssao.run_loop(ssao.pyr.lambdaInM, ssao.starMagnitude, save_prefix='')

    unmod_ssao = SingleStageAO(tn)
    unmod_ssao.initialize_turbulence()
    unmod_ssao.pyr.set_modulation_angle(0.0)
    mod0_Rec, mod0_IM = unmod_ssao.compute_reconstructor(unmod_ssao.sc, KL, unmod_ssao.pyr.lambdaInM, amps=0.002, save_prefix='mod0_')
    unmod_ssao.sc.load_reconstructor(mod0_IM,m2c)
    mod0_sig2, _ = unmod_ssao.run_loop(ssao.pyr.lambdaInM, ssao.starMagnitude, save_prefix='mod0_')

    dmod_ssao = ChangingGainSSAO(tn)
    dmod_ssao.initialize_turbulence()
    dmod_ssao.sc.load_reconstructor(mod0_IM,m2c)
    dmod_sig2, input_sig2 = dmod_ssao.run_loop(dmod_ssao.pyr.lambdaInM, dmod_ssao.starMagnitude,
                                      new_gains=[1.0], changeGain_it_numbers=[100], save_prefix='dmod_')
                                    #   new_Rec=mod0_Rec, changeMod_it_number=200, save_prefix='dmod_')

    lambdaRef = dmod_ssao.pyr.lambdaInM
    # ssao.plot_iteration(lambdaRef, frame_id=-1, save_prefix='')
    # unmod_ssao.plot_iteration(lambdaRef, frame_id=-1, save_prefix='mod0_')
    dmod_ssao.plot_iteration(lambdaRef, frame_id=-1, save_prefix='dmod_')


    # ssao.plot_rec_modes(save_prefix='')
    # unmod_ssao.plot_rec_modes(save_prefix='mod0_')
    dmod_ssao.plot_rec_modes(save_prefix='dmod_')

    ssao.sig2 = sig2
    ssao.dmod_sig2 = dmod_sig2

    if xp.on_gpu: # Convert to numpy for plotting
        input_sig2 = input_sig2.get()
        sig2 = sig2.get()
        mod0_sig2 = mod0_sig2.get()
        dmod_sig2 = dmod_sig2.get()

    tvec = xp.asnumpy(xp.arange(dmod_ssao.Nits)*dmod_ssao.dt*1e+3)
    # plt.figure()#figsize=(1.7*Nits/10,3))
    # plt.plot(tvec,input_sig2,'-o',label='open loop')
    # plt.plot(tvec,sig2,'-o',label=f'closed loop, modulation: {ssao.pyr.modulationAngleInLambdaOverD} $lambda/D$')
    # plt.plot(tvec,mod0_sig2,'-o',label=f'closed loop, modulation: {unmod_ssao.pyr.modulationAngleInLambdaOverD} $lambda/D$')
    # plt.legend()
    # plt.grid()
    # plt.xlim([0.0,tvec[-1]])
    # plt.xlabel('Time [ms]')
    # plt.ylabel(r'$\sigma^2 [rad^2]$')
    # plt.gca().set_yscale('log')

    plt.figure()
    plt.plot(tvec,input_sig2,'-o',label='open loop')
    plt.plot(tvec,sig2,'-o',label=r'closed loop, modulation = 3 $\lambda/D$')
    plt.plot(tvec,mod0_sig2,'-o',label=r'closed loop, modulation = 0 $\lambda/D$')
    plt.plot(tvec,dmod_sig2,'-o',label=r'closed loop, modulation = 0 $\lambda/D$, changing gain')
    plt.legend()
    plt.grid()
    plt.xlim([0.0,tvec[-1]])
    plt.xlabel('Time [ms]')
    plt.ylabel(r'$\sigma^2 [rad^2]$')
    plt.gca().set_yscale('log')

    _,S,_ = xp.linalg.svd(Rec,full_matrices=False)
    _,mod0_S,_ = xp.linalg.svd(mod0_Rec,full_matrices=False)
    S = xp.asnumpy(S)
    mod0_S = xp.asnumpy(mod0_S)
    plt.figure()
    plt.plot(S,'-o',label=f'modulation: {ssao.pyr.modulationAngleInLambdaOverD} $lambda/D$')
    plt.plot(mod0_S,'-x',label=f'modulation: {unmod_ssao.pyr.modulationAngleInLambdaOverD} $lambda/D$')
    plt.legend()
    plt.xlabel('KL mode number')
    plt.yscale('log')
    plt.grid()
    plt.title('Reconstructor singular values')

    _,S,_ = xp.linalg.svd(IM,full_matrices=False)
    _,mod0_S,_ = xp.linalg.svd(mod0_IM,full_matrices=False)
    S = xp.asnumpy(S)
    mod0_S = xp.asnumpy(mod0_S)
    plt.figure()
    plt.plot(S,'-o',label=f'modulation: {ssao.pyr.modulationAngleInLambdaOverD} $lambda/D$')
    plt.plot(mod0_S,'-x',label=f'modulation: {unmod_ssao.pyr.modulationAngleInLambdaOverD} $lambda/D$')
    plt.legend()
    plt.xlabel('KL mode number')
    plt.yscale('log')
    plt.grid()
    plt.title('Interaction matrix singular values')
    
    plt.show()

    return ssao, dmod_ssao

if __name__ == '__main__':
    ssao, dmod_ssao = main()