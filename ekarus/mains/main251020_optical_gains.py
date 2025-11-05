import xupy as xp
masked_array = xp.masked_array
import matplotlib.pyplot as plt

from ekarus.e2e.single_stage_ao_class import SingleStageAO


def main(tn:str='optical_gains', gain:float=0.3):


    ssao = SingleStageAO(tn)
    ssao.initialize_turbulence()
    ssao.pyr.set_modulation_angle(ssao.sc.modulationAngleInLambdaOverD)
    KL, m2c = ssao.define_KL_modes(ssao.dm, zern_modes=5)
    Rec, IM = ssao.compute_reconstructor(ssao.sc, KL, ssao.pyr.lambdaInM, amps=0.2)
    ssao.sc.load_reconstructor(IM,m2c)
    ssao.KL = KL

    lambdaRef = ssao.pyr.lambdaInM

    print('Calibrating optical gains ...')
    N = 4
    ol_opt_gains, cl_opt_gains, pl_opt_gains = ssao.calibrate_optical_gains(N, slope_computer=ssao.sc, MM=KL, amps=0.2)
    plt.figure()
    plt.plot(xp.asnumpy(ol_opt_gains),'-.',label='open loop')
    plt.plot(xp.asnumpy(cl_opt_gains),'-.',label='closed loop')
    plt.plot(xp.asnumpy(pl_opt_gains),'-.',label='perfect loop')
    plt.legend()
    plt.title('Optical gains vs mode')
    plt.grid()

    ogcl_ssao = SingleStageAO(tn)
    ogcl_ssao.initialize_turbulence()
    ogcl_ssao.pyr.set_modulation_angle(ssao.sc.modulationAngleInLambdaOverD)
    ogcl_ssao.sc.load_reconstructor(IM,m2c,cl_opt_gains)
    ogcl_ssao.sc.intGain = gain

    ogol_ssao = SingleStageAO(tn)
    ogol_ssao.initialize_turbulence()
    ogol_ssao.pyr.set_modulation_angle(ssao.sc.modulationAngleInLambdaOverD)
    ogol_ssao.sc.load_reconstructor(IM,m2c,ol_opt_gains)
    ogol_ssao.sc.intGain = gain

    ogpl_ssao = SingleStageAO(tn)
    ogpl_ssao.initialize_turbulence()
    ogpl_ssao.pyr.set_modulation_angle(ssao.sc.modulationAngleInLambdaOverD)
    ogpl_ssao.sc.load_reconstructor(IM,m2c,pl_opt_gains)
    ogpl_ssao.sc.intGain = gain

    # it_ss = 200
    # if optimize_gain is True:
    #     print('Finding best gains:')
    #     N = len(gain_vec)
    #     SR_vec = xp.zeros(N)
    #     for jj in range(N):
    #         g = gain_vec[jj]
    #         ogc_ssao.sc.intGain = g
    #         err2, _ = ogc_ssao.run_loop(lambdaRef, ssao.starMagnitude)
    #         SR = xp.mean(xp.exp(-err2[-it_ss:]))
    #         SR_vec[jj] = SR.copy()
    #         print(f'Tested gain = {g:1.2f}, final SR = {SR*100:1.2f}%' )

    #     best_gain = gain_vec[xp.argmax(SR_vec)]
    #     print(f'Selecting best integrator gain: {best_gain:1.1f}, yielding SR={xp.max(SR_vec):1.2f} @{lambdaRef*1e+9:1.0f}[nm]')   
    #     ogc_ssao = SingleStageAO(tn)
    #     ogc_ssao.initialize_turbulence()
    #     ogc_ssao.pyr.set_modulation_angle(ssao.sc.modulationAngleInLambdaOverD)
    #     ogc_ssao.sc.load_reconstructor(Rec,m2c)
    #     ogc_ssao.sc.load_optical_gains(cl_opt_gains)
    #     ogc_ssao.sc.intGain = best_gain
    #     if xp.on_gpu:
    #         gain_vec = gain_vec.get()
    #         SR_vec = SR_vec.get()
    #     plt.figure()
    #     plt.plot(gain_vec,SR_vec*100,'-o')
    #     plt.grid()
    #     plt.xlabel('Integrator gain')
    #     plt.ylabel('SR %')
    #     plt.title('Strehl ratio vs integrator gain')

    ssao.KL = KL
    ogcl_ssao.KL = KL
    ogol_ssao.KL = KL
    ogpl_ssao.KL = KL

    sig2, input_sig2 = ssao.run_loop(lambdaRef, ssao.starMagnitude, save_prefix='')
    ogcl_sig2, _ = ogcl_ssao.run_loop(lambdaRef, ssao.starMagnitude, save_prefix='ogcl_')
    ogol_sig2, _ = ogol_ssao.run_loop(lambdaRef, ssao.starMagnitude, save_prefix='ogol_')
    ogpl_sig2, _ = ogpl_ssao.run_loop(lambdaRef, ssao.starMagnitude, save_prefix='ogpl_')

    ssao.plot_iteration(lambdaRef, frame_id=-1, save_prefix='')
    # psd,pix_dist=ssao.plot_contrast(lambdaRef, frame_ids=xp.arange(ssao.Nits-100,ssao.Nits).tolist(), save_prefix='')
    ogcl_ssao.plot_iteration(lambdaRef, frame_id=-1, save_prefix='ogcl_')
    ogol_ssao.plot_iteration(lambdaRef, frame_id=-1, save_prefix='ogol_')
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
    plt.plot(xp.asnumpy(tvec),xp.asnumpy(ogol_sig2),'-.',label='OG (open loop) compensation')
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