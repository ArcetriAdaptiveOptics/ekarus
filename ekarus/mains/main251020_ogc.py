import xupy as xp
masked_array = xp.masked_array
import matplotlib.pyplot as plt

from ekarus.e2e.single_stage_ao_class import SingleStageAO

import os
from ekarus.e2e.utils import my_fits_package as myfits


def main(tn:str='optical_gains', 
         optimize_gain:bool=False,
         gain_list = None,
         atmo_tn='five_layers_25mOS',
         N:int=40,
         new_r0=None):
    
    if gain_list is not None:
        optimize_gain = True
    
    if optimize_gain is True and gain_list is None:
        gain_list = xp.arange(1,10)/10
        gain_list = gain_list.tolist()

    ssao = SingleStageAO(tn)
    ssao.initialize_turbulence(atmo_tn)
    if new_r0 is not None:
        ssao.layers.update_r0(new_r0)
    saveprefix = f'mod{ssao.sc.modulationAngleInLambdaOverD:1.0f}_r0{ssao.layers._r0*1e+2:1.1f}cm_'

    amp = 50e-9
    KL, m2c = ssao.define_KL_modes(ssao.dm, zern_modes=2)
    _, IM = ssao.compute_reconstructor(ssao.sc, KL, ssao.pyr.lambdaInM, ampsInM=amp, save_prefix='')
    ssao.sc.load_reconstructor(IM,m2c)
    ssao.KL = KL

    lambdaRef = ssao.pyr.lambdaInM

    # try:
    #     ma_res_screens = myfits.read_fits(os.path.join(ssao.savepath,'residual_phases.fits'))
    # except FileNotFoundError:
    sig2, input_sig2 = ssao.run_loop(lambdaRef, ssao.starMagnitude, save_prefix='')
    ma_res_screens = myfits.read_fits(os.path.join(ssao.savepath,'residual_phases.fits'))

    offset = 100

    # M = xp.shape(xp.asarray(ma_res_screens.data))[0]-offset
    # all_phases = xp.zeros([M,int(xp.sum(1-ssao.cmask))])
    # for j in range(M):
    #     all_phases[j,:] = xp.asarray(ma_res_screens[j+offset].data[~ma_res_screens[j+offset].mask])
    # sn = ssao.calibrate_slope_nulls_from_precorrected_screens(all_phases,slope_computer=ssao.sc,save_prefix=saveprefix)
    # plt.figure()
    # plt.plot(xp.asnumpy(sn),'--.')
    # plt.xscale('log')
    # plt.grid()
    # plt.title('Slope Nulls')

    loop_residual_phases = xp.zeros([N,int(xp.sum(1-ssao.cmask))])
    step = (xp.shape(xp.asarray(ma_res_screens.data))[0]-offset)//N
    for j in range(N):
        loop_residual_phases[j,:] = xp.asarray(ma_res_screens[j*step+offset].data[~ma_res_screens[j*step+offset].mask])
    print('Calibrating optical gains ...')
    cl_opt_gains, pl_opt_gains = ssao.calibrate_optical_gains_from_precorrected_screens(
                                 loop_residual_phases,
                                 slope_computer=ssao.sc, MM=KL, 
                                 save_prefix = saveprefix, ampsInM=amp,
                                 return_perfect_ogs=True)
    cl_opt_gains = xp.mean(cl_opt_gains,axis=0)
    pl_opt_gains = xp.mean(pl_opt_gains,axis=0)

    plt.figure()
    plt.plot(xp.asnumpy(cl_opt_gains),'-.',label='closed loop')
    plt.plot(xp.asnumpy(pl_opt_gains),'-.',label='perfect loop')
    plt.legend()
    plt.title('Optical gains vs mode')
    plt.grid()

    ogcl_ssao = SingleStageAO(tn)
    ogcl_ssao.initialize_turbulence(atmo_tn)
    if new_r0 is not None:
        ogcl_ssao.layers.update_r0(new_r0)
    ogcl_ssao.sc.load_reconstructor(IM,m2c)
    ogcl_ssao.sc.modalGains /= cl_opt_gains
    # ogcl_ssao.sc.slope_null /= cl_opt_gains
    # ogcl_ssao.sc.slope_null = sn.copy()

    ogpl_ssao = SingleStageAO(tn)
    ogpl_ssao.initialize_turbulence(atmo_tn)
    if new_r0 is not None:
        ogpl_ssao.layers.update_r0(new_r0)
    ogpl_ssao.sc.load_reconstructor(IM,m2c)
    ogpl_ssao.sc.modalGains /= pl_opt_gains
    # ogpl_ssao.sc.slope_null /= pl_opt_gains
    # ogpl_ssao.sc.slope_null = sn.copy()

    it_ss = max(200,int(ssao.Nits//2))
    if optimize_gain is True:
        gain_list = xp.array(gain_list)
        print('Finding best gains:')
        N = len(gain_list)
        SR_vec = xp.zeros(N)
        for jj in range(N):
            g = gain_list[jj]
            ogcl_ssao.sc.set_new_gain(g)
            err2, _ = ogcl_ssao.run_loop(lambdaRef, ssao.starMagnitude)
            SR = xp.mean(xp.exp(-err2[-it_ss:]))
            SR_vec[jj] = SR.copy()
            print(f'Tested gain = {g:1.2f}, final SR = {SR*100:1.2f}%' )

        best_gain = gain_list[xp.argmax(SR_vec)]
        print(f'Selecting best integrator gain: {best_gain:1.1f}, yielding SR={xp.max(SR_vec):1.2f} @{lambdaRef*1e+9:1.0f}[nm]')   
        ogcl_ssao.sc.set_new_gain(best_gain)
        if xp.on_gpu:
            gain_list = gain_list.get()
            SR_vec = SR_vec.get()
        plt.figure()
        plt.plot(gain_list,SR_vec*100,'-o')
        plt.grid()
        plt.xlabel('Integrator gain')
        plt.ylabel('SR %')
        plt.title('Strehl ratio vs integrator gain')

    ogcl_ssao.KL = KL
    ogpl_ssao.KL = KL

    ogcl_sig2, _ = ogcl_ssao.run_loop(lambdaRef, ssao.starMagnitude, save_prefix='ogcl_')
    ogpl_sig2, _ = ogpl_ssao.run_loop(lambdaRef, ssao.starMagnitude, save_prefix='ogpl_')

    ssao.plot_iteration(lambdaRef, frame_id=-1, save_prefix='')
    ogcl_ssao.plot_iteration(lambdaRef, frame_id=-1, save_prefix='ogcl_')
    ogpl_ssao.plot_iteration(lambdaRef, frame_id=-1, save_prefix='ogpl_')

    # plt.figure()
    # plt.plot(xp.asnumpy(pix_dist),xp.asnumpy(psd),'--',label='no OG compensation')
    # plt.plot(xp.asnumpy(pix_dist),xp.asnumpy(ogc_psd),'--',label='with OG compensation')
    # plt.grid()
    # plt.legend()
    # plt.yscale('log')
    # plt.xlabel(r'$\lambda/D$')
    # plt.xlim([0,30])

    tvec = xp.arange(ssao.Nits)*ssao.dt*1e+3
    plt.figure()#figsize=(1.7*Nits/10,3))
    plt.plot(xp.asnumpy(tvec),xp.asnumpy(input_sig2),'-.',label='open loop')
    plt.plot(xp.asnumpy(tvec),xp.asnumpy(sig2),'-.',label='no compensation')
    plt.plot(xp.asnumpy(tvec),xp.asnumpy(ogcl_sig2),'-.',label='OG (close loop) compensation')
    # plt.plot(xp.asnumpy(tvec),xp.asnumpy(ogol_sig2),'-.',label='OG (open loop) compensation')
    plt.plot(xp.asnumpy(tvec),xp.asnumpy(ogpl_sig2),'-.',label='OG (perfect loop) compensation')
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