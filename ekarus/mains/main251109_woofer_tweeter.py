import matplotlib.pyplot as plt
import os.path as op

from ekarus.e2e.woofer_tweeter_ao_class import WooferTweeterAO

import ekarus.e2e.utils.my_fits_package as myfits   
import xupy as xp


def main(tn:str, 
         atmo_tn:str='paranal',
         show:bool=False, 
         show_contrast:bool=False,
         lambdaRef:float=750e-9,
         optimize_gain:bool=False, 
         gain1_list:list=None, 
         gain2_list:list=None):

    print('Initializing devices ...')
    wooftweet = WooferTweeterAO(tn)

    amp = 50e-9

    wooftweet.initialize_turbulence(tn=atmo_tn)
    KL1, m2c1 = wooftweet.define_KL_modes(wooftweet.dm1, zern_modes=2, save_prefix='woof_')

    # KL2, m2c2 = wooftweet.define_KL_modes(wooftweet.dm2, filt_modes=KL1[:wooftweet.sc1.nModes,:], save_prefix='tweet_')
    KL2 = KL1[wooftweet.sc1.nModes:,:]
    m2c2 = m2c1[:,wooftweet.sc1.nModes:]

    Rec1, IM1 = wooftweet.compute_reconstructor(wooftweet.sc1, KL1[:wooftweet.sc1.nModes,:], wooftweet.pyr1.lambdaInM, ampsInM=amp, save_prefix='woof_')
    wooftweet.sc1.load_reconstructor(IM1,m2c1)

    _, IM2 = wooftweet.compute_reconstructor(wooftweet.sc2, KL2, wooftweet.pyr2.lambdaInM, ampsInM=amp, save_prefix='tweet_')
    wooftweet.sc2.load_reconstructor(IM2,m2c2)

    # # Slope null
    # modes_null = Rec1 @ wooftweet.sc1.slope_null
    # modes_null *= wooftweet.pyr1.lambdaInM/(2*xp.pi)
    # plt.figure()
    # plt.plot(xp.asnumpy(xp.abs(modes_null)),'-o')
    # plt.grid()
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.title('Slope null modes')
    # # slope_null = None

    wooftweet.get_photons_per_subap(wooftweet.starMagnitude)

    if gain1_list is not None or gain2_list is not None:
        optimize_gain = True


    if optimize_gain:

        if gain1_list is not None:
            gain1_vec = xp.array(gain1_list)
        else:
            gain1_vec = xp.arange(1,11)/10 + 0.1

        if gain2_list is not None:
            gain2_vec = xp.array(gain2_list)
        else:
            gain2_vec = xp.arange(1,11)/10

        best_gain1 = gain1_vec[0]
        best_gain2 = gain2_vec[0]
        best_SR = 0.0

        ss_it = 200

        print('Finding the best gain for woofer (DM1)')
        for gain in gain1_vec:
            wooftweet.sc1.set_new_gain(gain)
            sig2, _, _ = wooftweet.run_loop(lambdaRef, wooftweet.starMagnitude, enable_tweeter=False)
            SR = xp.mean(xp.exp(-sig2[-ss_it:]))
            print(f'Woofer gain = {wooftweet.sc1.intGain:1.2f}, tweeter gain = {wooftweet.sc2.intGain:1.2f}, final SR = {SR*100:1.2f}%')
            if SR > best_SR:
                best_SR = SR.copy()
                best_gain1 = gain.copy()

        print('Finding the best gain for tweeter (DM2)')
        for gain in gain2_vec:
            wooftweet.sc1.set_new_gain(best_gain1)
            wooftweet.sc2.set_new_gain(gain)
            sig2, _, _ = wooftweet.run_loop(lambdaRef, wooftweet.starMagnitude)
            SR = xp.mean(xp.exp(-sig2[-ss_it:]))
            print(f'Woofer gain = {wooftweet.sc1.intGain:1.2f}, tweeter gain = {wooftweet.sc2.intGain:1.2f}, final SR = {SR*100:1.2f}%')
            if SR > best_SR:
                best_SR = SR.copy()
                best_gain2 = gain.copy()

        wooftweet.tested_gains1 = gain1_vec        
        wooftweet.tested_gains2 = gain2_vec

        wooftweet.sc1.set_new_gain(best_gain1)
        wooftweet.sc2.set_new_gain(best_gain2)
        print(f'Selecting tweeter (DM2) gain = {wooftweet.sc2.intGain}, woofer (DM1) gain = {wooftweet.sc1.intGain}, yielding Strehl {best_SR*1e+2:1.2f}')

    wooftweet.KL = KL1.copy()
    
    print('Running the loop ...')
    saveprefix = f'mag{wooftweet.starMagnitude:1.0f}_'
    tweeter_sig2, woofer_sig2, input_sig2 = wooftweet.run_loop(lambdaRef, wooftweet.starMagnitude, save_prefix=saveprefix)
    wooftweet.sig2 = tweeter_sig2

    # Post-processing and plotting
    print('Plotting results ...')
    if show:

        KL = KL1[:wooftweet.sc1.nModes,:]
        N=9
        plt.figure(figsize=(2*N,7))
        for i in range(N):
            plt.subplot(4,N,i+1)
            wooftweet.dm1.plot_surface(KL[i,:],title=f'KL Mode {i}')
            plt.subplot(4,N,i+1+N)
            wooftweet.dm1.plot_surface(KL[i+N,:],title=f'KL Mode {i+N}')
            plt.subplot(4,N,i+1+N*2)
            wooftweet.dm1.plot_surface(KL[-i-1-N,:],title=f'KL Mode {xp.shape(KL)[0]-i-1-N}')
            plt.subplot(4,N,i+1+N*3)
            wooftweet.dm1.plot_surface(KL[-i-1,:],title=f'KL Mode {xp.shape(KL)[0]-i-1}')

        KL = KL2[:wooftweet.sc2.nModes,:]
        N=9
        plt.figure(figsize=(2*N,7))
        for i in range(N):
            plt.subplot(4,N,i+1)
            wooftweet.dm2.plot_surface(KL[i,:],title=f'KL Mode {i}')
            plt.subplot(4,N,i+1+N)
            wooftweet.dm2.plot_surface(KL[i+N,:],title=f'KL Mode {i+N}')
            plt.subplot(4,N,i+1+N*2)
            wooftweet.dm2.plot_surface(KL[-i-1-N,:],title=f'KL Mode {xp.shape(KL)[0]-i-1-N}')
            plt.subplot(4,N,i+1+N*3)
            wooftweet.dm2.plot_surface(KL[-i-1,:],title=f'KL Mode {xp.shape(KL)[0]-i-1}')

        IM1 = xp.asarray(myfits.read_fits(op.join(wooftweet.savecalibpath,'woof_IM.fits')))
        _,IM1_eig,_ = xp.linalg.svd(IM1,full_matrices=False)
        IM2 = xp.asarray(myfits.read_fits(op.join(wooftweet.savecalibpath,'tweet_IM.fits')))
        _,IM2_eig,_ = xp.linalg.svd(IM2,full_matrices=False)
        plt.figure()
        plt.plot(xp.asnumpy(IM1_eig),'-o',label='woofer')
        plt.plot(xp.asnumpy(IM2_eig),'-o',label='tweeter')
        plt.legend()
        plt.xscale('log')
        plt.yscale('log')
        plt.grid()
        plt.title('Interaction matrix singular values')

    wooftweet.plot_iteration(lambdaRef, frame_id=-1, save_prefix=saveprefix)
    if show_contrast:
        wooftweet.psd1, wooftweet.psd2,wooftweet.pix_scale = wooftweet.plot_contrast(lambdaRef=lambdaRef, frame_ids=xp.arange(wooftweet.Nits-200,wooftweet.Nits).tolist(),save_prefix=saveprefix)

    tvec = xp.asnumpy(xp.arange(wooftweet.Nits)*wooftweet.dt*1e+3)
    plt.figure()#figsize=(1.7*Nits/10,3))
    plt.plot(tvec,xp.asnumpy(input_sig2),'-.',label='open loop')
    plt.plot(tvec,xp.asnumpy(woofer_sig2),'-.',label='after woofer (DM1)')
    plt.plot(tvec,xp.asnumpy(tweeter_sig2),'-.',label='after tweeter (DM2)')
    plt.legend()
    plt.grid()
    plt.xlim([0.0,tvec[-1]])
    plt.xlabel('Time [ms]')
    plt.ylabel(r'$\sigma^2 [rad^2]$')
    plt.gca().set_yscale('log')

    plt.show()

    return wooftweet

if __name__ == '__main__':
    wooftweet = main(show=True)