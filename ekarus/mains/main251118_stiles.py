import matplotlib.pyplot as plt

from ekarus.e2e.cascading_stiles_class import CascadingBench

import xupy as xp
import numpy as np


def main(tn:str='cascading_stiles', 
         atmo_tn='paranal',
         lambdaRef=800e-9, 
         show_contrast:bool=True,
         optimize_gain:bool=False, 
         gain1_list:list=None, 
         gain2_list:list=None,
         save_prefix:str=None):

    print('Initializing devices ...')
    cascao = CascadingBench(tn)
    cascao.initialize_turbulence(atmo_tn)

    amp1 = 50e-9
    amp2 = 50e-9
    if cascao.sc1.modulationAngleInLambdaOverD < 1.0:
        amp1 = 20e-9
    if cascao.sc2.modulationAngleInLambdaOverD < 1.0:
        amp2 = 20e-9

    KL1, m2c1 = cascao.define_KL_modes(cascao.dm1, zern_modes=2, save_prefix='DM1_')
    Rec1, IM1 = cascao.compute_reconstructor(cascao.sc1, KL1, cascao.pyr1.lambdaInM, ampsInM=amp1, save_prefix='SC1_')
    cascao.sc1.load_reconstructor(IM1,m2c1)

    KL2, m2c2 = cascao.define_KL_modes(cascao.dm2, zern_modes=2, save_prefix='DM2_')
    Rec2, IM2 = cascao.compute_reconstructor(cascao.sc2, KL2, cascao.pyr2.lambdaInM, ampsInM=amp2, save_prefix='SC2_')
    cascao.sc2.load_reconstructor(IM2,m2c2)

    cascao.get_photons_per_subap(starMagnitude=cascao.starMagnitude)

    if save_prefix is None:
        save_prefix = f'mag{cascao.starMagnitude:1.0f}_'

    if gain1_list is not None or gain2_list is not None:
        optimize_gain = True

    if optimize_gain:

        if gain1_list is not None:
            gain1_vec = xp.array(gain1_list)
        else:
            gain1_vec = xp.array([0.4,0.5,0.6,0.7,0.8,0.9])
            if cascao.sc1.modulationAngleInLambdaOverD == 0.0:
                gain1_vec = xp.array([0.8,0.9,1.0,1.1,1.2,1.3])

        if gain2_list is not None:
            gain2_vec = xp.array(gain2_list)
        else:
            gain2_vec = xp.array([0.4,0.5,0.6,0.7,0.8,0.9,1.0])
            if cascao.sc2.modulationAngleInLambdaOverD == 0.0:
                gain2_vec = xp.array([0.8,0.9,1.0,1.1,1.2,1.3])

        Ni = len(gain1_vec)
        Nj = len(gain2_vec)
        SR_mat = xp.zeros([Ni,Nj])

        best_gain1 = gain1_vec[0]
        best_gain2 = gain2_vec[0]
        best_SR = 0.0
        ss_it = max(200,int(cascao.Nits//2))

        print('Finding the best gain')
        for i in range(Ni):
            for j in range(Nj):
                cascao.sc1.set_new_gain(gain1_vec[i])
                cascao.sc2.set_new_gain(gain2_vec[j])
                sig2, _, _ = cascao.run_loop(lambdaRef, cascao.starMagnitude)
                SR = xp.mean(xp.exp(-sig2[-ss_it:]))
                SR_mat[i,j] = SR.copy()
                print(f'First loop gain = {cascao.sc1.intGain:1.2f}, second loop gain = {cascao.sc2.intGain:1.2f}, final SR = {SR*100:1.2f}%')
                if SR_mat[i,j] > best_SR:
                    best_SR = SR_mat[i,j]
                    best_gain1 = gain1_vec[i]
                    best_gain2 = gain2_vec[j]

        plt.figure()
        plt.imshow(xp.asnumpy(SR_mat),cmap='jet')
        plt.colorbar()
        plt.yticks(np.arange(Ni),labels=[str(g1) for g1 in gain1_vec])
        plt.xticks(np.arange(Nj),labels=[str(g2) for g2 in gain2_vec])
        plt.ylabel('First loop gain')
        plt.xlabel('Second loop gain')
        # plt.zlabel('SR %')
        for i in range(Ni):
            for j in range(Nj):
                plt.text(j,i, f'{SR_mat[i,j]*100:1.2f}', ha='center', va='center', color='w')

        cascao.tested_gains1 = gain1_vec        
        cascao.tested_gains2 = gain2_vec
        cascao.SR_mat = SR_mat

        cascao.sc1.set_new_gain(best_gain1)
        cascao.sc2.set_new_gain(best_gain2)
        print(f'Selecting first loop gain = {cascao.sc1.intGain}, second loop gain = {cascao.sc2.intGain}, yielding Strehl {best_SR*1e+2:1.2f}')


    cascao.KL = KL1 if KL1.shape[0] > KL2.shape[0] else KL2
    print('Running the loop ...')
    dm2_sig2, dm1_sig2, input_sig2 = cascao.run_loop(lambdaRef, cascao.starMagnitude, save_prefix=save_prefix)

    cascao.dm1_sig2 = dm1_sig2
    cascao.dm2_sig2 = dm2_sig2

    cascao.plot_iteration(lambdaRef, save_prefix=save_prefix)

    if show_contrast:
        cascao.psd1, cascao.psd2, cascao.pix_scale = cascao.plot_contrast(lambdaRef=lambdaRef, oversampling=8,
                                                    frame_ids=xp.arange(cascao.Nits-200,cascao.Nits).tolist(), 
                                                    save_prefix=save_prefix)

    tvec = xp.asnumpy(xp.arange(cascao.Nits)*cascao.dt*1e+3)
    plt.figure()#figsize=(1.7*Nits/10,3))
    plt.plot(tvec,xp.asnumpy(input_sig2),'-.',label='open loop')
    plt.plot(tvec,xp.asnumpy(dm1_sig2),'-.',label='after DM1')
    plt.plot(tvec,xp.asnumpy(dm2_sig2),'-.',label='after DM2')
    plt.legend()
    plt.grid()
    plt.xlim([0.0,tvec[-1]])
    plt.xlabel('Time [ms]')
    plt.ylabel(r'$\sigma^2 [rad^2]$')
    plt.gca().set_yscale('log')

    plt.show()

    return cascao

if __name__ == '__main__':
    cascao = main(show=True)