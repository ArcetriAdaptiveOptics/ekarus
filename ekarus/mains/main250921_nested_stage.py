import matplotlib.pyplot as plt
import os.path as op

from ekarus.e2e.utils.image_utils import showZoomCenter, myimshow, reshape_on_mask
from ekarus.e2e.nested_stage_ao_class import NestedStageAO

import ekarus.e2e.utils.my_fits_package as myfits   

import xupy as xp
import numpy as np
from numpy.ma import masked_array


def main(tn:str='example_nested_stage', show:bool=False, lambdaRef:float=750e-9,
         optimize_gain:bool=False, gain1_list:list=None, gain2_list:list=None):

    print('Initializing devices ...')
    cascao = NestedStageAO(tn)

    amp1 = 0.2
    amp2 = 0.2
    if cascao.sc1.modulationAngleInLambdaOverD < 1.0:
        amp1 = 0.02
    if cascao.sc2.modulationAngleInLambdaOverD < 1.0:
        amp2 = 0.02

    cascao.initialize_turbulence()
    KL1, m2c1 = cascao.define_KL_modes(cascao.dm1, zern_modes=5, save_prefix='DM1_')
    cascao.pyr1.set_modulation_angle(cascao.sc1.modulationAngleInLambdaOverD)
    Rec1, _ = cascao.compute_reconstructor(cascao.sc1, KL1, cascao.pyr1.lambdaInM, amps=amp1, save_prefix='SC1_')
    cascao.sc1.load_reconstructor(Rec1,m2c1)

    KL2, m2c2 = cascao.define_KL_modes(cascao.dm2, zern_modes=5, save_prefix='DM2_')
    cascao.pyr2.set_modulation_angle(cascao.sc2.modulationAngleInLambdaOverD)
    Rec2, _ = cascao.compute_reconstructor(cascao.sc2, KL2, cascao.pyr2.lambdaInM, amps=amp2, save_prefix='SC2_')
    cascao.sc2.load_reconstructor(Rec2,m2c2)

    cascao.get_photons_per_subap(cascao.starMagnitude)

    if gain1_list is not None or gain2_list is not None:
        optimize_gain = True

    if optimize_gain:

        if gain1_list is not None:
            gain1_vec = xp.array(gain1_list)
        else:
            gain1_vec = xp.array([0.4,0.5,0.6,0.7,0.8,0.9])
            # if cascao.sc1.modulationAngleInLambdaOverD == 0.0:
            #     gain1_vec = xp.array([0.8,0.9,1.0,1.1,1.2,1.3])

        if gain2_list is not None:
            gain2_vec = xp.array(gain2_list)
        else:
            gain2_vec = xp.array([0.4,0.5,0.6,0.7,0.8,0.9])
            # if cascao.sc2.modulationAngleInLambdaOverD == 0.0:
            #     gain2_vec = xp.array([0.8,0.9,1.0,1.1,1.2,1.3])

        Ni = len(gain1_vec)
        Nj = len(gain2_vec)
        SR_mat = xp.zeros([Ni,Nj])

        best_gain1 = gain1_vec[0]
        best_gain2 = gain2_vec[0]
        best_SR = 0.0

        ss_it = 100

        print('Finding the best gain')
        for i in range(Ni):
            for j in range(Nj):
                cascao = NestedStageAO(tn)
                cascao.initialize_turbulence()
                cascao.pyr1.set_modulation_angle(cascao.sc1.modulationAngleInLambdaOverD)
                cascao.sc1.load_reconstructor(Rec1,m2c1)
                cascao.pyr2.set_modulation_angle(cascao.sc2.modulationAngleInLambdaOverD)
                cascao.sc2.load_reconstructor(Rec2,m2c2)
                cascao.sc1.intGain = gain1_vec[i]
                cascao.sc2.intGain = gain2_vec[j]
                sig2, _, _ = cascao.run_loop(lambdaRef, cascao.starMagnitude)
                SR = xp.mean(xp.exp(-sig2[-ss_it:]))
                SR_mat[i,j] = SR.copy()
                print(f'Inner loop gain = {cascao.sc1.intGain:1.2f}, outer loop gain = {cascao.sc2.intGain:1.2f}, final SR = {SR*100:1.2f}%')
                if SR_mat[i,j] > best_SR:
                    best_SR = SR.copy()
                    best_gain1 = gain1_vec[i].copy()
                    best_gain2 = gain2_vec[j].copy()


        cascao.KL1 = KL1
        cascao.KL2 = KL2

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

        cascao.sc1.intGain = best_gain1
        cascao.sc2.intGain = best_gain2
        print(f'Selecting first loop gain = {cascao.sc1.intGain}, second loop gain = {cascao.sc2.intGain}, yielding Strehl {best_SR*1e+2:1.2f}')


    print('Running the loop ...')
    dm_outer_sig2, dm_inner_sig2, input_sig2 = cascao.run_loop(lambdaRef, cascao.starMagnitude, save_prefix='')
    cascao.sig2 = dm_outer_sig2

    # Post-processing and plotting
    print('Plotting results ...')
    if show:
        subap1_masks = xp.sum(cascao.sc1._subaperture_masks,axis=0)
        plt.figure()
        myimshow(subap1_masks,title='Subaperture masks CCD1')
        subap2_masks = xp.sum(cascao.sc2._subaperture_masks,axis=0)
        plt.figure()
        myimshow(subap2_masks,title='Subaperture masks CCD2')

        KL = myfits.read_fits(op.join(cascao.savecalibpath,'DM1_KLmodes.fits'))
        N=9
        plt.figure(figsize=(2*N,7))
        for i in range(N):
            plt.subplot(4,N,i+1)
            cascao.dm1.plot_surface(KL[i,:],title=f'KL Mode {i}')
            plt.subplot(4,N,i+1+N)
            cascao.dm1.plot_surface(KL[i+N,:],title=f'KL Mode {i+N}')
            plt.subplot(4,N,i+1+N*2)
            cascao.dm1.plot_surface(KL[-i-1-N,:],title=f'KL Mode {xp.shape(KL)[0]-i-1-N}')
            plt.subplot(4,N,i+1+N*3)
            cascao.dm1.plot_surface(KL[-i-1,:],title=f'KL Mode {xp.shape(KL)[0]-i-1}')

        KL = myfits.read_fits(op.join(cascao.savecalibpath,'DM2_KLmodes.fits'))
        N=9
        plt.figure(figsize=(2*N,7))
        for i in range(N):
            plt.subplot(4,N,i+1)
            cascao.dm2.plot_surface(KL[i,:],title=f'KL Mode {i}')
            plt.subplot(4,N,i+1+N)
            cascao.dm2.plot_surface(KL[i+N,:],title=f'KL Mode {i+N}')
            plt.subplot(4,N,i+1+N*2)
            cascao.dm2.plot_surface(KL[-i-1-N,:],title=f'KL Mode {xp.shape(KL)[0]-i-1-N}')
            plt.subplot(4,N,i+1+N*3)
            cascao.dm2.plot_surface(KL[-i-1,:],title=f'KL Mode {xp.shape(KL)[0]-i-1}')

        IM1 = myfits.read_fits(op.join(cascao.savecalibpath,'SC1_IM.fits'))
        _,IM1_eig,_ = xp.asnumpy(xp.linalg.svd(IM1,full_matrices=False))
        IM2 = myfits.read_fits(op.join(cascao.savecalibpath,'SC2_IM.fits'))
        _,IM2_eig,_ = xp.asnumpy(xp.linalg.svd(IM2,full_matrices=False))
        plt.figure()
        plt.plot(IM1_eig,'-o')
        plt.plot(IM2_eig,'-o')
        plt.grid()
        plt.title('Interaction matrix singular values')

        screen = cascao.get_phasescreen_at_time(0.5)
        if xp.on_gpu:
            screen = screen.get()
        plt.figure()
        myimshow(masked_array(screen,cascao.cmask), title='Atmo screen [m]', cmap='RdBu')

    cascao.plot_iteration(lambdaRef, frame_id=-1, save_prefix='')
    cascao.psd1, cascao.psd2,cascao.pix_scale = cascao.plot_contrast(lambdaRef=lambdaRef, frame_id=-1, save_prefix='')

    # cmask = cascao.cmask.get() if xp.__name__ == 'cupy' else cascao.cmask.copy()
    if xp.on_gpu: # Convert to numpy for plotting
        dm_inner_sig2 = dm_inner_sig2.get()
        dm_outer_sig2 = dm_outer_sig2.get()
        input_sig2 = input_sig2.get()
        # rec_in_modes = rec_in_modes.get()
        # rec_out_modes = rec_out_modes.get()

    tvec = xp.asnumpy(xp.arange(cascao.Nits)*cascao.dt*1e+3)
    plt.figure()#figsize=(1.7*Nits/10,3))
    plt.plot(tvec,input_sig2,'-.',label='open loop')
    plt.plot(tvec,dm_inner_sig2,'-.',label='after DM1 (inner loop)')
    plt.plot(tvec,dm_outer_sig2,'-.',label='after DM2 (outer loop)')
    plt.legend()
    plt.grid()
    plt.xlim([0.0,tvec[-1]])
    plt.xlabel('Time [ms]')
    plt.ylabel(r'$\sigma^2 [rad^2]$')
    plt.gca().set_yscale('log')

    # plt.figure()
    # plt.plot(tvec,rec_in_modes[:,:10],'-o')
    # plt.grid()
    # plt.xlim([0.0,tvec[-1]])
    # plt.xlabel('Time [ms]')
    # plt.ylabel('amplitude [m]')
    # plt.title('Reconstructor (inner loop) modes\n(first 10)')

    # plt.figure()
    # plt.plot(tvec,rec_in_modes[:,:10],'-o')
    # plt.grid()
    # plt.xlim([0.0,tvec[-1]])
    # plt.xlabel('Time [ms]')
    # plt.ylabel('amplitude [m]')
    # plt.title('Reconstructor (outer loop) modes\n(first 10)')
    # plt.show()

    plt.show()

    return cascao

if __name__ == '__main__':
    cascao = main(show=True)