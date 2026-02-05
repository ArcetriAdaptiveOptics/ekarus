import matplotlib.pyplot as plt
# import os.path as op

from ekarus.e2e.woofer_tweeter_ao_class import WooferTweeterAO

# import ekarus.e2e.utils.my_fits_package as myfits   
import xupy as xp

# from ekarus.e2e.utils.image_utils import reshape_on_mask
# from numpy.ma import masked_array


def main(tn:str, 
         atmo_tn:str='paranal',
         show:bool=False,
         show_contrast:bool=True,
         lambdaRef:float=730e-9,
         optimize_gain:bool=False, 
         gain1_list:list=None, 
         gain2_list:list=None,
         saveprefix:str=None,
         bootStrapIts:int=0):

    print('Initializing devices ...')
    wooftweet = WooferTweeterAO(tn)

    amp1 = 25e-9
    amp2 = 15e-9

    wooftweet.initialize_turbulence(tn=atmo_tn)
    KL, m2c = wooftweet.define_KL_modes(wooftweet.dm, zern_modes=2)
    # display_modes(1-wooftweet.cmask, xp.asnumpy(KL.T), N=8)

    _, IM1 = wooftweet.compute_reconstructor(wooftweet.sc1, KL, lambdaInM=wooftweet.wfs1.lambdaInM, ampsInM=amp1)
    wooftweet.sc1.load_reconstructor(IM1,m2c)

    _, IM2 = wooftweet.compute_reconstructor(wooftweet.sc2, KL[:wooftweet.sc2.nModes,:], lambdaInM=wooftweet.wfs2.lambdaInM, ampsInM=amp2, save_prefix='wfs2_')
    wooftweet.sc2.load_reconstructor(IM2,m2c[:,:wooftweet.sc2.nModes])

    wooftweet.get_photons_per_subap(wooftweet.starMagnitude)

    if gain1_list is not None or gain2_list is not None:
        optimize_gain = True

    wooftweet.bootStrapIts = bootStrapIts

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

        print('Finding the best gain for low order modes (LO)')
        for gain in gain1_vec:
            wooftweet.sc1.set_new_gain(gain)
            sig2, _ = wooftweet.run_loop(lambdaRef, wooftweet.starMagnitude)
            SR = xp.mean(xp.exp(-sig2[-ss_it:]))
            print(f'LO gain = {wooftweet.sc1.intGain:1.2f}, HO gain = {wooftweet.sc2.intGain:1.2f}, final SR = {SR*100:1.2f}%')
            if SR > best_SR:
                best_SR = SR.copy()
                best_gain1 = gain.copy()

        print('Finding the best gain for high order modes (HO)')
        for gain in gain2_vec:
            wooftweet.sc1.set_new_gain(best_gain1)
            wooftweet.sc2.set_new_gain(gain)
            sig2, _ = wooftweet.run_loop(lambdaRef, wooftweet.starMagnitude)
            SR = xp.mean(xp.exp(-sig2[-ss_it:]))
            print(f'LO gain = {wooftweet.sc1.intGain:1.2f}, HO gain = {wooftweet.sc2.intGain:1.2f}, final SR = {SR*100:1.2f}%')
            if SR > best_SR:
                best_SR = SR.copy()
                best_gain2 = gain.copy()

        wooftweet.tested_gains1 = gain1_vec        
        wooftweet.tested_gains2 = gain2_vec

        wooftweet.sc1.set_new_gain(best_gain1)
        wooftweet.sc2.set_new_gain(best_gain2)
        print(f'Selecting HO gain = {wooftweet.sc2.intGain}, LO gain = {wooftweet.sc1.intGain}, yielding Strehl {best_SR*1e+2:1.2f}')

    wooftweet.KL = KL.copy()
    
    print('Running the loop ...')
    if saveprefix is None:
        saveprefix = f'wt_mag{wooftweet.starMagnitude:1.0f}_'
    sig2, input_sig2 = wooftweet.run_loop(lambdaRef, wooftweet.starMagnitude, save_prefix=saveprefix)
    wooftweet.sig2 = sig2

    if show_contrast:
        wooftweet.psd, wooftweet.pix_scale = wooftweet.plot_contrast(lambdaRef=lambdaRef, 
                                                                    frame_ids=xp.arange(100+wooftweet.bootStrapIts,wooftweet.Nits).tolist(),
                                                                    save_prefix=saveprefix, oversampling=10)
        # wooftweet.smf = wooftweet.plot_ristretto_contrast(lambdaRef=lambdaRef,frame_ids=xp.arange(wooftweet.Nits).tolist(), save_prefix=saveprefix, oversampling=10)
        
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

# def plot_mode_j(pupil,Mat,j:int,title:str=''):
#     mode = reshape_on_mask(Mat[:,j],(1-pupil).astype(bool))
#     plt.imshow(masked_array(xp.asnumpy(mode),xp.asnumpy(1-pupil)),origin='lower',cmap='RdBu')
#     plt.colorbar()
#     plt.axis('off')
#     plt.title(title)

# def display_modes(pupil, Mat, N:int=8):
#     nModes = xp.shape(Mat)[1]
#     plt.figure(figsize=(2.25*N,12))
#     for i in range(N):
#         plt.subplot(6,N,i+1)
#         plot_mode_j(pupil,Mat,i,title=f'Mode {i}')
#         plt.subplot(6,N,i+1+N)
#         plot_mode_j(pupil,Mat,i+N,title=f'Mode {i+N}')

#         plt.subplot(6,N,i+1+N*2)
#         plot_mode_j(pupil,Mat,nModes//2-N+i,title=f'Mode {nModes//2-N+i}')
#         plt.subplot(6,N,i+1+N*3)
#         plot_mode_j(pupil,Mat,nModes//2+N+i,title=f'Mode {nModes//2+N+i}')

#         plt.subplot(6,N,i+1+N*4)
#         plot_mode_j(pupil,Mat,nModes-2*N+i,title=f'Mode {nModes-2*N+i}')
#         plt.subplot(6,N,i+1+N*5)
#         plot_mode_j(pupil,Mat,nModes-N+i,title=f'Mode {nModes-N+i}')

if __name__ == '__main__':
    wooftweet = main(show=True)