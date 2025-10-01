import matplotlib.pyplot as plt
import os.path as op

from ekarus.e2e.utils.image_utils import showZoomCenter, myimshow, reshape_on_mask
from ekarus.e2e.nested_stage_ao_class import NestedStageAO

import ekarus.e2e.utils.my_fits_package as myfits   

import xupy as xp
import numpy as np
from numpy.ma import masked_array


def main(tn:str='example_nested_stage', show:bool=False):

    print('Initializing devices ...')
    cascao = NestedStageAO(tn)

    cascao.initialize_turbulence()

    KL, m2c = cascao.define_KL_modes(cascao.dm1, zern_modes=5, save_prefix='DM1_')
    cascao.pyr1.set_modulation_angle(cascao.sc1.modulationAngleInLambdaOverD)
    Rec, _ = cascao.compute_reconstructor(cascao.sc1, KL, cascao.pyr1.lambdaInM, amps=0.2, save_prefix='SC1_')
    cascao.sc1.load_reconstructor(Rec,m2c)

    KL, m2c = cascao.define_KL_modes(cascao.dm2, zern_modes=5, save_prefix='DM2_')
    cascao.pyr2.set_modulation_angle(cascao.sc2.modulationAngleInLambdaOverD)
    Rec, _ = cascao.compute_reconstructor(cascao.sc2, KL, cascao.pyr2.lambdaInM, amps=0.2, save_prefix='SC2_')
    cascao.sc2.load_reconstructor(Rec,m2c)

    print('Running the loop ...')
    lambdaRef = 800e-9
    dm_outer_sig2, dm_inner_sig2, input_sig2 = cascao.run_loop(lambdaRef, cascao.starMagnitude, save_prefix='')

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
        IM1_std = xp.std(IM1,axis=0)
        IM2 = myfits.read_fits(op.join(cascao.savecalibpath,'SC2_IM.fits'))
        IM2_std = xp.std(IM2,axis=0)
        if xp.on_gpu:
            IM1_std = IM1_std.get()
            IM2_std = IM2_std.get()
        plt.figure()
        plt.plot(IM1_std,'-o')
        plt.plot(IM2_std,'-o')
        plt.grid()
        plt.title('Interaction matrix standard deviation')

        screen = cascao.get_phasescreen_at_time(0.5)
        if xp.on_gpu:
            screen = screen.get()
        plt.figure()
        myimshow(masked_array(screen,cascao.cmask), title='Atmo screen [m]', cmap='RdBu')

    lambdaRef = 1000e-9
    cascao.plot_iteration(lambdaRef, frame_id=-1, save_prefix='')

    # cmask = cascao.cmask.get() if xp.__name__ == 'cupy' else cascao.cmask.copy()
    if xp.on_gpu: # Convert to numpy for plotting
        dm_inner_sig2 = dm_inner_sig2.get()
        dm_outer_sig2 = dm_outer_sig2.get()
        input_sig2 = input_sig2.get()
        # rec_in_modes = rec_in_modes.get()
        # rec_out_modes = rec_out_modes.get()

    tvec = xp.arange(cascao.Nits)*cascao.dt*1e+3
    tvec = tvec.get() if xp.on_gpu else tvec.copy()
    plt.figure()#figsize=(1.7*Nits/10,3))
    plt.plot(tvec,input_sig2,'-o',label='open loop')
    plt.plot(tvec,dm_inner_sig2,'-o',label='after DM1 (inner loop)')
    plt.plot(tvec,dm_outer_sig2,'-o',label='after DM2 (outer loop)')
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