import xupy as xp
import numpy as np
masked_array = xp.masked_array

import matplotlib.pyplot as plt

from ekarus.e2e.utils.image_utils import myimshow #showZoomCenter, reshape_on_mask, 
from ekarus.e2e.single_stage_ao_class import SingleStageAO
import os.path as op

import ekarus.e2e.utils.my_fits_package as myfits


def main(tn:str='example_single_stage',
         show:bool=False, 
         gain_list=None, 
         show_contrast:bool =False,
         optimize_gain:bool=False, 
         starMagnitudes=None, 
         atmo_tn='example_single_stage',
         lambdaRef:float=750e-9,
         save_prefix:str=None):
    
    if gain_list is not None:
        optimize_gain = True

    if optimize_gain is True:
        if gain_list is not None:
            gain_vec = xp.array(gain_list)
        else:
            gain_vec = xp.arange(1,11)/10

    ssao = SingleStageAO(tn)
    ssao.initialize_turbulence(tn=atmo_tn)
    KL, m2c = ssao.define_KL_modes(ssao.dm, zern_modes=2)
    Rec, IM = ssao.compute_reconstructor(ssao.sc, KL, ssao.pyr.lambdaInM, ampsInM=50e-9)
    ssao.sc.load_reconstructor(IM,m2c)
    ssao.KL = KL

    print(ssao.sc._slope_method)

    it_ss = max(200,int(ssao.Nits//2))


    if optimize_gain is True:
        print('Finding best gains:')
        N = len(gain_vec)
        SR_vec = xp.zeros(N)
        for jj in range(N):
            g = gain_vec[jj]
            ssao.sc.set_new_gain(g)
            err2, _ = ssao.run_loop(lambdaRef, ssao.starMagnitude)
            SR = xp.mean(xp.exp(-err2[-it_ss:]))
            SR_vec[jj] = SR.copy()
            print(f'Tested gain = {g:1.2f}, final SR = {SR*100:1.2f}%' )

        best_gain = gain_vec[xp.argmax(SR_vec)]
        print(f'Selecting best integrator gain: {best_gain:1.1f}, yielding SR={xp.max(SR_vec):1.2f} @{lambdaRef*1e+9:1.0f}[nm]')   
        ssao.sc.set_new_gain(best_gain)
        if xp.on_gpu:
            gain_vec = gain_vec.get()
            SR_vec = SR_vec.get()
        plt.figure()
        plt.plot(gain_vec,SR_vec*100,'-o')
        plt.grid()
        plt.xlabel('Integrator gain')
        plt.ylabel('SR %')
        plt.title('Strehl ratio vs integrator gain')

    
    if starMagnitudes is not None:
        sig = xp.zeros([len(starMagnitudes),ssao.Nits])
        for k in range(len(starMagnitudes)):
            starMag = starMagnitudes[k]
            save_prefix = f'magV{starMag:1.0f}_'+ssao.atmo_pars_str
            print(f'Now simulating for magnitude: {starMag:1.1f}')
            sig[k,:],_ = ssao.run_loop(lambdaRef, starMag, save_prefix=save_prefix)
        ssao.SR = xp.mean(xp.exp(-sig[:,-it_ss:]),axis=1)
        ssao.plot_iteration(lambdaRef, frame_id=-1, save_prefix=save_prefix)
    else:
        if save_prefix is None:
            save_prefix = f'mag{ssao.starMagnitude:1.0f}_'+ssao.atmo_pars_str
        sig2, input_sig2 = ssao.run_loop(lambdaRef, ssao.starMagnitude, save_prefix=save_prefix)
        ssao.plot_iteration(lambdaRef, frame_id=-1, save_prefix=save_prefix)
        # ssao.SR_in = xp.exp(-input_sig2)
        # ssao.SR_out = xp.exp(-sig2)


    # Post-processing and plotting
    print('Plotting results ...')
    if show:
        subap_masks = xp.sum(ssao.sc._roi_masks,axis=0)
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

        IM = xp.asarray(myfits.read_fits(op.join(ssao.savecalibpath,'IM.fits')))
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

        if ssao.sc.slope_null is not None:
            modes_null = Rec @ ssao.sc.slope_null
            modes_null *= lambdaRef/(2*xp.pi)
            plt.figure()
            plt.plot(xp.asnumpy(xp.abs(modes_null))*1e+9,'-o')
            plt.grid()
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('Mode #')
            plt.ylabel('[nm]')
            plt.title('Slope null modes')

    if show_contrast:
        ssao.plot_contrast(lambdaRef, frame_ids=xp.arange(ssao.Nits-100,ssao.Nits).tolist(), save_prefix=save_prefix)

    if starMagnitudes is not None:
        ssao.sig2 = sig
    else:
        ssao.sig2 = sig2

    tvec = xp.asnumpy(xp.arange(ssao.Nits)*ssao.dt*1e+3)

    if starMagnitudes is not None:
        plt.figure()
        SR_stars = np.zeros(len(starMagnitudes))
        for k,mag in enumerate(starMagnitudes):
            SR_stars[k] = np.mean(np.exp(-sig[k,-it_ss:]))
            plt.plot(tvec,xp.asnumpy(sig[k]),'-.',label=f'magV={mag:1.1f}, SR={xp.asnumpy(SR_stars[k]):1.2f}')
        plt.legend()
        plt.grid()
        plt.xlim([0.0,tvec[-1]])
        plt.xlabel('Time [ms]')
        plt.ylabel(r'$\sigma^2 [rad^2]$')
        plt.gca().set_yscale('log')
        plt.title(f'Strehl @ {lambdaRef*1e+9:1.0f} [nm] vs star magnitude')

        # plt.figure()
        # plt.plot(np.array(starMagnitudes),xp.asnumpy(SR_stars)*100,'-o')
        # # plt.errorbar(np.array(starMagnitudes),SR_stars*100,yerr=0.12,fmt='-o',capsize=4.0)
        # plt.grid()
        # plt.xlabel('magV')
        # plt.ylabel('SR %')
        # plt.title('Strehl ratio vs star magnitude')
        # plt.xticks(np.array(starMagnitudes))
    else:
        plt.figure()#figsize=(1.7*Nits/10,3))
        plt.plot(tvec,xp.asnumpy(input_sig2),'-.',label='open loop')
        plt.plot(tvec,xp.asnumpy(sig2),'-.',label='closed loop')
        plt.legend()
        plt.grid()
        plt.xlim([0.0,tvec[-1]])
        plt.xlabel('Time [ms]')
        plt.ylabel(r'$\sigma^2 [rad^2]$')
        plt.gca().set_yscale('log')

    
    plt.show()

    return ssao

if __name__ == '__main__':
    ssao = main(show=True)