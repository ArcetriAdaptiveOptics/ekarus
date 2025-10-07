import matplotlib.pyplot as plt
import os.path as op

from ekarus.e2e.utils.image_utils import myimshow#, showZoomCenter,  reshape_on_mask
from ekarus.e2e.cascading_stage_ao_class import CascadingAO

import ekarus.e2e.utils.my_fits_package as myfits   

import xupy as xp
import numpy as np
from numpy.ma import masked_array


def main(tn:str='example_cascading_stage', lambdaRef=800e-9, show:bool=False, optimize_gain:bool=False):

    print('Initializing devices ...')
    cascao = CascadingAO(tn)
    cascao.initialize_turbulence()

    gain_vec = xp.array([0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1])
    N = len(gain_vec)
    SR_mat = xp.zeros([N,N])

    KL, m2c = cascao.define_KL_modes(cascao.dm1, zern_modes=5, save_prefix='DM1_')
    cascao.pyr1.set_modulation_angle(cascao.sc1.modulationAngleInLambdaOverD)
    Rec, _ = cascao.compute_reconstructor(cascao.sc1, KL, cascao.pyr1.lambdaInM, amps=0.2, save_prefix='SC1_')
    cascao.sc1.load_reconstructor(Rec,m2c)

    KL, m2c = cascao.define_KL_modes(cascao.dm2, zern_modes=5, save_prefix='DM2_')
    cascao.pyr2.set_modulation_angle(cascao.sc2.modulationAngleInLambdaOverD)
    Rec, _ = cascao.compute_reconstructor(cascao.sc2, KL, cascao.pyr2.lambdaInM, amps=0.2, save_prefix='SC2_')
    cascao.sc2.load_reconstructor(Rec,m2c)

    if optimize_gain:
        best_gain1 = gain_vec[0]
        best_gain2 = gain_vec[0]
        best_SR = 0.0

        ss_it = cascao.Nits//2

        print('Finding the best gain')
        for i in range(N):
            for j in range(N):
                cascao.sc1.intGain = gain_vec[i]
                cascao.sc2.intGain = gain_vec[j]
                print(f'First loop gain: {cascao.sc1.intGain:1.1f}, second loop gain: {cascao.sc2.intGain:1.1f}')
                sig2, _, _ = cascao.run_loop(lambdaRef, cascao.starMagnitude)
                SR_mat[i,j] = xp.mean(xp.exp(-sig2[ss_it:]))
                if SR_mat[i,j] > best_SR:
                    best_SR = SR_mat[i,j]
                    best_gain1 = gain_vec[i]
                    best_gain2 = gain_vec[j]

        plt.figure()
        plt.matshow(xp.asnumpy(SR_mat))
        plt.colorbar()
        # plt.gca().set_xticks(xp.asnumpy(gain_vec))
        # plt.gca().set_yticks(xp.asnumpy(gain_vec))
        plt.ylabel('First loop gain id')
        plt.xlabel('Second loop gain id')

        cascao.tested_gains = gain_vec
        cascao.SR_mat = SR_mat

        cascao.sc1.intGain = best_gain1
        cascao.sc2.intGain = best_gain2
        print(f'Selecting first loop gain = {cascao.sc1.intGain}, second loop gain = {cascao.sc2.intGain}, yielding Strehl {best_SR*1e+2:1.2f}')

    print('Running the loop ...')
    dm2_sig2, dm1_sig2, input_sig2 = cascao.run_loop(lambdaRef, cascao.starMagnitude, save_prefix='')

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

    cascao.dm1_sig2 = dm1_sig2
    cascao.dm2_sig2 = dm2_sig2

    # cmask = cascao.cmask.get() if xp.__name__ == 'cupy' else cascao.cmask.copy()
    if xp.on_gpu: # Convert to numpy for plotting
        dm1_sig2 = dm1_sig2.get()
        dm2_sig2 = dm2_sig2.get()
        input_sig2 = input_sig2.get()
        # rec1_modes = rec1_modes.get()
        # rec2_modes = rec2_modes.get()

    cascao.plot_iteration(lambdaRef, frame_id=-1, save_prefix='')

    tvec = xp.asnumpy(xp.arange(cascao.Nits)*cascao.dt*1e+3)
    plt.figure()#figsize=(1.7*Nits/10,3))
    plt.plot(tvec,input_sig2,'-o',label='open loop')
    plt.plot(tvec,dm1_sig2,'-o',label='after DM1')
    plt.plot(tvec,dm2_sig2,'-o',label='after DM2')
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