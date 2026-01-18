import matplotlib.pyplot as plt
import os.path as op

from ekarus.e2e.woofer_tweeter_ao_class import WooferTweeterAO

import ekarus.e2e.utils.my_fits_package as myfits   
import xupy as xp


def main(tn:str, 
         atmo_tn:str='paranal',
         show:bool=False,
         show_contrast:bool=True,
         lambdaRef:float=730e-9,
         optimize_gain:bool=False, 
         gain1_list:list=None, 
         gain2_list:list=None,
         saveprefix:str=None):

    print('Initializing devices ...')
    wooftweet = WooferTweeterAO(tn)

    amp = 25e-9

    wooftweet.initialize_turbulence(tn=atmo_tn)
    KL, m2c = wooftweet.define_KL_modes(wooftweet.dm, zern_modes=2)

    _, IM1 = wooftweet.compute_reconstructor(wooftweet.sc1, KL[:wooftweet.sc1.nModes,:], wooftweet.wfs1.lambdaInM, ampsInM=amp, save_prefix='wfs1_')
    wooftweet.sc1.load_reconstructor(IM1,m2c[:,:wooftweet.sc1.nModes])

    _, IM2 = wooftweet.compute_reconstructor(wooftweet.sc2, KL[wooftweet.sc1.nModes:,:], wooftweet.wfs2.lambdaInM, ampsInM=amp, save_prefix='wfs2_')
    wooftweet.sc2.load_reconstructor(IM2,m2c[:,wooftweet.sc1.nModes:])

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
            sig2, _ = wooftweet.run_loop(lambdaRef, wooftweet.starMagnitude, enable_tweeter=False)
            SR = xp.mean(xp.exp(-sig2[-ss_it:]))
            print(f'Woofer gain = {wooftweet.sc1.intGain:1.2f}, tweeter gain = {wooftweet.sc2.intGain:1.2f}, final SR = {SR*100:1.2f}%')
            if SR > best_SR:
                best_SR = SR.copy()
                best_gain1 = gain.copy()

        print('Finding the best gain for tweeter (DM2)')
        for gain in gain2_vec:
            wooftweet.sc1.set_new_gain(best_gain1)
            wooftweet.sc2.set_new_gain(gain)
            sig2, _ = wooftweet.run_loop(lambdaRef, wooftweet.starMagnitude)
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

    wooftweet.KL = KL.copy()
    
    print('Running the loop ...')
    if saveprefix is None:
        saveprefix = f'mag{wooftweet.starMagnitude:1.0f}_'
    sig2, input_sig2 = wooftweet.run_loop(lambdaRef, wooftweet.starMagnitude, save_prefix=saveprefix)
    wooftweet.sig2 = sig2

    if show_contrast:
        wooftweet.psd, wooftweet.pix_scale = wooftweet.plot_contrast(lambdaRef=lambdaRef, 
                                                                                     frame_ids=xp.arange(wooftweet.Nits-160,wooftweet.Nits).tolist(),
                                                                                     save_prefix=saveprefix, oversampling=10)
        wooftweet.smf = wooftweet.plot_ristretto_contrast(lambdaRef=lambdaRef,frame_ids=xp.arange(wooftweet.Nits).tolist(), save_prefix=saveprefix, oversampling=10)
        
    if show:
        wooftweet.plot_iteration(lambdaRef, frame_id=-1, save_prefix=saveprefix)
        
        tvec = xp.asnumpy(xp.arange(wooftweet.Nits)*wooftweet.dt*1e+3)
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

    return wooftweet

if __name__ == '__main__':
    wooftweet = main(show=True)