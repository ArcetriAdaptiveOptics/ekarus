import xupy as xp
import numpy as np
masked_array = xp.masked_array

import matplotlib.pyplot as plt

from ekarus.e2e.utils.image_utils import myimshow #showZoomCenter, reshape_on_mask, 
from ekarus.e2e.single_stage_ao_class import SingleStageAO
import os.path as op

import ekarus.e2e.utils.my_fits_package as myfits


def main(tn:str='example_single_stage', show:bool=False, gain_list=None, 
         optimize_gain:bool=False, starMagnitudes=None):
    
    if gain_list is not None:
        optimize_gain = True

    if optimize_gain is True:
        if gain_list is not None:
            gain_vec = xp.array(gain_list)
        else:
            gain_vec = xp.arange(1,11)/10

    ssao = SingleStageAO(tn)
    ssao.initialize_turbulence()
    ssao.pyr.set_modulation_angle(ssao.sc.modulationAngleInLambdaOverD)
    KL, m2c = ssao.define_KL_modes(ssao.dm, zern_modes=5)
    Rec, _ = ssao.compute_reconstructor(ssao.sc, KL, ssao.pyr.lambdaInM, amps=0.2)
    ssao.sc.load_reconstructor(Rec,m2c)

    lambdaRef = ssao.pyr.lambdaInM
    it_ss = 200

    if optimize_gain is True:
        print('Finding best gains:')
        N = len(gain_vec)
        SR_vec = xp.zeros(N)
        for jj in range(N):
            g = gain_vec[jj]
            # ssao = SingleStageAO(tn)
            # ssao.initialize_turbulence()
            # ssao.pyr.set_modulation_angle(ssao.sc.modulationAngleInLambdaOverD)
            # ssao.sc.load_reconstructor(Rec,m2c)
            ssao.sc.intGain = g
            err2, _ = ssao.run_loop(lambdaRef, ssao.starMagnitude)
            SR = xp.mean(xp.exp(-err2[-it_ss:]))
            SR_vec[jj] = SR.copy()
            print(f'Tested gain = {g:1.2f}, final SR = {SR*100:1.2f}%' )

        best_gain = gain_vec[xp.argmax(SR_vec)]
        print(f'Selecting best integrator gain: {best_gain:1.1f}, yielding SR={xp.max(SR_vec):1.2f} @{lambdaRef*1e+9:1.0f}[nm]')   
        # ssao = SingleStageAO(tn)
        # ssao.initialize_turbulence()
        # ssao.pyr.set_modulation_angle(ssao.sc.modulationAngleInLambdaOverD)
        # ssao.sc.load_reconstructor(Rec,m2c)
        ssao.sc.intGain = best_gain
        if xp.on_gpu:
            gain_vec = gain_vec.get()
            SR_vec = SR_vec.get()
        plt.figure()
        plt.plot(gain_vec,SR_vec*100,'-o')
        plt.grid()
        plt.xlabel('Integrator gain')
        plt.ylabel('SR %')
        plt.title('Strehl ratio vs integrator gain')

    sig2, input_sig2 = ssao.run_loop(lambdaRef, ssao.starMagnitude, save_prefix='')
    # ssao.SR_in = xp.exp(-input_sig2)
    # ssao.SR_out = xp.exp(-sig2)
    
    if starMagnitudes is not None:
        # new_ssao = SingleStageAO(tn)
        # new_ssao.initialize_turbulence()
        # new_ssao.pyr.set_modulation_angle(ssao.sc.modulationAngleInLambdaOverD)
        # new_ssao.sc.load_reconstructor(Rec,m2c)
        sig = xp.zeros([len(starMagnitudes),ssao.Nits])
        for k in range(len(starMagnitudes)):
            starMag = starMagnitudes[k]
            print(f'Now simulating for magnitude: {starMag:1.1f}')
            sig[k,:],_ = ssao.run_loop(lambdaRef, starMag, save_prefix=f'magV{starMag:1.0f}_')
        ssao.SR = xp.mean(xp.exp(-sig[:,-it_ss:]),axis=1)

    # Post-processing and plotting
    print('Plotting results ...')
    if show:
        subap_masks = xp.sum(ssao.sc._subaperture_masks,axis=0)
        plt.figure()
        myimshow(subap_masks,title='Subaperture masks')

        KL = myfits.read_fits(op.join(ssao.savecalibpath,'KLmodes.fits'))
        N=9
        plt.figure(figsize=(2*N,7))
        for i in range(N):
            plt.subplot(4,N,i+1)
            ssao.dm.plot_surface(KL[i,:],title=f'KL Mode {i}')
            plt.subplot(4,N,i+1+N)
            ssao.dm.plot_surface(KL[i+N,:],title=f'KL Mode {i+N}')
            plt.subplot(4,N,i+1+N*2)
            ssao.dm.plot_surface(KL[-i-1-N,:],title=f'KL Mode {xp.shape(KL)[0]-2*N+i}')
            plt.subplot(4,N,i+1+N*3)
            ssao.dm.plot_surface(KL[-i-1,:],title=f'KL Mode {xp.shape(KL)[0]-N+i}')

        IM = myfits.read_fits(op.join(ssao.savecalibpath,'IM.fits'))
        # IM_std = xp.std(IM,axis=0)
        _,Sim,_ = xp.linalg.svd(IM, full_matrices=False)
        plt.figure()
        plt.plot(xp.asnumpy(Sim),'-o')
        plt.grid()
        plt.title('Interaction matrix eigenvalues')

        screen = ssao.get_phasescreen_at_time(0)
        if xp.on_gpu:
            screen = screen.get()
        plt.figure()
        myimshow(masked_array(screen,ssao.cmask), title='Atmo screen [m]', cmap='RdBu')

    ssao.plot_iteration(lambdaRef, frame_id=-1, save_prefix='')
    if show:
        ssao.plot_rec_modes(save_prefix='')
    ssao.sig2 = sig2

    if xp.on_gpu: # Convert to numpy for plotting
        input_sig2 = input_sig2.get()
        sig2 = sig2.get()
        if starMagnitudes is not None:
            sig = sig.get()

    tvec = xp.arange(ssao.Nits)*ssao.dt*1e+3
    tvec = tvec.get() if xp.on_gpu else tvec.copy()
    plt.figure()#figsize=(1.7*Nits/10,3))
    plt.plot(tvec,input_sig2,'-.',label='open loop')
    plt.plot(tvec,sig2,'-.',label='closed loop')
    plt.legend()
    plt.grid()
    plt.xlim([0.0,tvec[-1]])
    plt.xlabel('Time [ms]')
    plt.ylabel(r'$\sigma^2 [rad^2]$')
    plt.gca().set_yscale('log')

    if starMagnitudes is not None:
        plt.figure()
        SR_stars = np.zeros(len(starMagnitudes))
        for k,mag in enumerate(starMagnitudes):
            SR_stars[k] = np.mean(np.exp(-sig[k,-it_ss:]))
            plt.plot(tvec,sig[k],'-.',label=f'magV={mag:1.1f}, SR={SR_stars[k]:1.2f}')
        plt.legend()
        plt.grid()
        plt.xlim([0.0,tvec[-1]])
        plt.xlabel('Time [ms]')
        plt.ylabel(r'$\sigma^2 [rad^2]$')
        plt.gca().set_yscale('log')
        plt.title(f'Strehl @ {lambdaRef*1e+9:1.0f} [nm] vs star magnitude')

        plt.figure()
        plt.plot(np.array(starMagnitudes),SR_stars*100,'-o')
        # plt.errorbar(np.array(starMagnitudes),SR_stars*100,yerr=0.12,fmt='-o',capsize=4.0)
        plt.grid()
        plt.xlabel('magV')
        plt.ylabel('SR %')
        plt.title('Strehl ratio vs star magnitude')
        plt.xticks(np.array(starMagnitudes))
    
    plt.show()

    return ssao

if __name__ == '__main__':
    ssao = main(show=True)