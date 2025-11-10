import matplotlib.pyplot as plt
import os.path as op

from ekarus.e2e.woofer_tweeter_ao_class import WooferTweeterAO

import ekarus.e2e.utils.my_fits_package as myfits   
import xupy as xp


def main(tn:str='example_woofer_tweeter', 
         show:bool=False, 
         show_contrast:bool=False,
         lambdaRef:float=750e-9,
         optimize_gain:bool=False, 
         gain1_list:list=None, 
         gain2_list:list=None):

    print('Initializing devices ...')
    cascao = WooferTweeterAO(tn)

    amp = 0.02

    cascao.initialize_turbulence()

    KL1, m2c1 = cascao.define_KL_modes(cascao.dm1, zern_modes=5, save_prefix='woof_')

    # KL2, m2c2 = cascao.define_KL_modes(cascao.dm2, filt_modes=KL1[:cascao.sc1.nModes,:], save_prefix='tweet_')
    KL2 = KL1[cascao.sc1.nModes:,:]
    m2c2 = m2c1[:,cascao.sc1.nModes:]

    cascao.pyr1.set_modulation_angle(cascao.sc1.modulationAngleInLambdaOverD)
    _, IM1 = cascao.compute_reconstructor(cascao.sc1, KL1[:cascao.sc1.nModes,:], cascao.pyr1.lambdaInM, amps=amp, save_prefix='woof_')
    cascao.sc1.load_reconstructor(IM1,m2c1)

    cascao.pyr2.set_modulation_angle(cascao.sc2.modulationAngleInLambdaOverD)
    _, IM2 = cascao.compute_reconstructor(cascao.sc2, KL2, cascao.pyr2.lambdaInM, amps=amp, save_prefix='tweet_')
    cascao.sc2.load_reconstructor(IM2,m2c2)

    cascao.get_photons_per_subap(cascao.starMagnitude)

    if gain1_list is not None or gain2_list is not None:
        optimize_gain = True


    if optimize_gain:

        if gain1_list is not None:
            gain1_vec = xp.array(gain1_list)
        else:
            gain1_vec = xp.arange(1,11)/10

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
            cascao = WooferTweeterAO(tn)
            cascao.initialize_turbulence()
            cascao.pyr1.set_modulation_angle(cascao.sc1.modulationAngleInLambdaOverD)
            cascao.sc1.load_reconstructor(IM1,m2c1)
            cascao.pyr2.set_modulation_angle(cascao.sc2.modulationAngleInLambdaOverD)
            cascao.sc2.load_reconstructor(IM2,m2c2)
            cascao.sc1.intGain = gain
            cascao.sc2.intGain = 0.0
            sig2, _, _ = cascao.run_loop(lambdaRef, cascao.starMagnitude, enable_tweeter=False)
            SR = xp.mean(xp.exp(-sig2[-ss_it:]))
            print(f'Woofer gain = {cascao.sc1.intGain:1.2f}, tweeter gain = {cascao.sc2.intGain:1.2f}, final SR = {SR*100:1.2f}%')
            if SR > best_SR:
                best_SR = SR.copy()
                best_gain1 = gain.copy()

        print('Finding the best gain for tweeter (DM2)')
        for gain in gain2_vec:
            cascao = WooferTweeterAO(tn)
            cascao.initialize_turbulence()
            cascao.pyr1.set_modulation_angle(cascao.sc1.modulationAngleInLambdaOverD)
            cascao.sc1.load_reconstructor(IM1,m2c1)
            cascao.pyr2.set_modulation_angle(cascao.sc2.modulationAngleInLambdaOverD)
            cascao.sc2.load_reconstructor(IM2,m2c2)
            cascao.sc1.intGain = best_gain1
            cascao.sc2.intGain = gain
            sig2, _, _ = cascao.run_loop(lambdaRef, cascao.starMagnitude)
            SR = xp.mean(xp.exp(-sig2[-ss_it:]))
            print(f'Woofer gain = {cascao.sc1.intGain:1.2f}, tweeter gain = {cascao.sc2.intGain:1.2f}, final SR = {SR*100:1.2f}%')
            if SR > best_SR:
                best_SR = SR.copy()
                best_gain2 = gain.copy()

        cascao = WooferTweeterAO(tn)
        cascao.initialize_turbulence()
        cascao.pyr1.set_modulation_angle(cascao.sc1.modulationAngleInLambdaOverD)
        cascao.sc1.load_reconstructor(IM1,m2c1)
        cascao.pyr2.set_modulation_angle(cascao.sc2.modulationAngleInLambdaOverD)
        cascao.sc2.load_reconstructor(IM2,m2c2)

        cascao.tested_gains1 = gain1_vec        
        cascao.tested_gains2 = gain2_vec

        cascao.sc1.intGain = best_gain1
        cascao.sc2.intGain = best_gain2
        print(f'Selecting tweeter (DM2) gain = {cascao.sc2.intGain}, woofer (DM1) gain = {cascao.sc1.intGain}, yielding Strehl {best_SR*1e+2:1.2f}')

    cascao.KL = KL1 if KL1.shape[0] > KL2.shape[0] else KL2
    
    print('Running the loop ...')
    tweeter_sig2, woofer_sig2, input_sig2 = cascao.run_loop(lambdaRef, cascao.starMagnitude, save_prefix='')
    cascao.sig2 = tweeter_sig2

    # Post-processing and plotting
    print('Plotting results ...')
    if show:

        KL = KL1.copy() #myfits.read_fits(op.join(cascao.savecalibpath,'tweet_KLmodes.fits'))
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

        KL = KL2.copy()
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

        IM1 = xp.asarray(myfits.read_fits(op.join(cascao.savecalibpath,'woof_IM.fits')))
        _,IM1_eig,_ = xp.linalg.svd(IM1,full_matrices=False)
        IM2 = xp.asarray(myfits.read_fits(op.join(cascao.savecalibpath,'tweet_IM.fits')))
        _,IM2_eig,_ = xp.linalg.svd(IM2,full_matrices=False)
        plt.figure()
        plt.plot(xp.asnumpy(IM1_eig),'-o',label='woofer')
        plt.plot(xp.asnumpy(IM2_eig),'-o',label='tweeter')
        plt.legend()
        plt.xscale('log')
        plt.yscale('log')
        plt.grid()
        plt.title('Interaction matrix singular values')

    cascao.plot_iteration(lambdaRef, frame_id=-1, save_prefix='')
    if show_contrast:
        cascao.psd1, cascao.psd2,cascao.pix_scale = cascao.plot_contrast(lambdaRef=lambdaRef, frame_ids=xp.arange(cascao.Nits-200,cascao.Nits).tolist(),save_prefix='')

    tvec = xp.asnumpy(xp.arange(cascao.Nits)*cascao.dt*1e+3)
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

    return cascao

if __name__ == '__main__':
    cascao = main(show=True)