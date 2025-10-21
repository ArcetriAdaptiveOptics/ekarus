import xupy as xp
masked_array = xp.masked_array
import matplotlib.pyplot as plt

from ekarus.e2e.single_stage_ao_class import SingleStageAO


def main(tn:str='optical_gains',
         gain_list=None, optimize_gain:bool=False):

    if gain_list is not None:
        optimize_gain = True

    if optimize_gain is True:
        if gain_list is not None:
            gain_vec = xp.array(gain_list)
        else:
            gain_vec = xp.arange(1,7)/10

    ssao = SingleStageAO(tn)
    ssao.initialize_turbulence()
    ssao.pyr.set_modulation_angle(ssao.sc.modulationAngleInLambdaOverD)
    KL, m2c = ssao.define_KL_modes(ssao.dm, zern_modes=5)
    Rec, _ = ssao.compute_reconstructor(ssao.sc, KL, ssao.pyr.lambdaInM, amps=0.02)
    ssao.sc.load_reconstructor(Rec,m2c)
    ssao.KL = KL

    lambdaRef = ssao.pyr.lambdaInM

    print('Calibrating optical gains ...')
    t_vec = xp.arange(4)/10
    opt_gains = xp.zeros([ssao.sc.nModes,len(t_vec)])
    for k,t in enumerate(t_vec):
        opt_gains[:,k] = ssao.calibrate_optical_gains(timeInSeconds=t, slope_computer=ssao.sc, MM=KL, amps=0.02)
    opt_gains = xp.mean(opt_gains,axis=1)    
    plt.figure()
    plt.plot(xp.asnumpy(opt_gains),'-.')
    plt.title('Optical gains vs mode')
    plt.grid()

    ogc_ssao = SingleStageAO(tn)
    ogc_ssao.initialize_turbulence()
    ogc_ssao.pyr.set_modulation_angle(ogc_ssao.sc.modulationAngleInLambdaOverD)
    ogc_ssao.sc.load_reconstructor(Rec,m2c)
    ogc_ssao.sc.load_optical_gains(opt_gains)
    ogc_ssao.sc.intGain = 0.2

    it_ss = 200

    if optimize_gain is True:
        print('Finding best gains:')
        N = len(gain_vec)
        SR_vec = xp.zeros(N)
        for jj in range(N):
            g = gain_vec[jj]
            ogc_ssao.sc.intGain = g
            err2, _ = ogc_ssao.run_loop(lambdaRef, ssao.starMagnitude)
            SR = xp.mean(xp.exp(-err2[-it_ss:]))
            SR_vec[jj] = SR.copy()
            print(f'Tested gain = {g:1.2f}, final SR = {SR*100:1.2f}%' )

        best_gain = gain_vec[xp.argmax(SR_vec)]
        print(f'Selecting best integrator gain: {best_gain:1.1f}, yielding SR={xp.max(SR_vec):1.2f} @{lambdaRef*1e+9:1.0f}[nm]')   
        ogc_ssao = SingleStageAO(tn)
        ogc_ssao.initialize_turbulence()
        ogc_ssao.pyr.set_modulation_angle(ssao.sc.modulationAngleInLambdaOverD)
        ogc_ssao.sc.load_reconstructor(Rec,m2c)
        ogc_ssao.sc.load_optical_gains(opt_gains)
        ogc_ssao.sc.intGain = best_gain
        if xp.on_gpu:
            gain_vec = gain_vec.get()
            SR_vec = SR_vec.get()
        plt.figure()
        plt.plot(gain_vec,SR_vec*100,'-o')
        plt.grid()
        plt.xlabel('Integrator gain')
        plt.ylabel('SR %')
        plt.title('Strehl ratio vs integrator gain')

    ssao.KL = KL
    ogc_ssao.KL = KL

    sig2, input_sig2 = ssao.run_loop(lambdaRef, ssao.starMagnitude, save_prefix='')
    ogc_sig2, _ = ogc_ssao.run_loop(lambdaRef, ssao.starMagnitude, save_prefix='ogc_')

    ssao.plot_iteration(lambdaRef, frame_id=-1, save_prefix='')
    psd,pix_dist=ssao.plot_contrast(lambdaRef, frame_id=-1, save_prefix='')
    ogc_ssao.plot_iteration(lambdaRef, frame_id=-1, save_prefix='ogc_')
    ogc_psd,_=ogc_ssao.plot_contrast(lambdaRef, frame_id=-1, save_prefix='ogc_')

    plt.figure()
    plt.plot(xp.asnumpy(pix_dist),xp.asnumpy(psd),'--',label='no OG compensation')
    plt.plot(xp.asnumpy(pix_dist),xp.asnumpy(ogc_psd),'--',label='with OG compensation')
    plt.grid()
    plt.legend()
    plt.yscale('log')
    plt.xlabel(r'$\lambda/D$')
    plt.xlim([0,30])

    tvec = xp.arange(ssao.Nits)*ssao.dt*1e+3
    plt.figure()#figsize=(1.7*Nits/10,3))
    plt.plot(xp.asnumpy(tvec),xp.asnumpy(input_sig2),'-.',label='open loop')
    plt.plot(xp.asnumpy(tvec),xp.asnumpy(sig2),'-.',label='no OG compensation')
    plt.plot(xp.asnumpy(tvec),xp.asnumpy(ogc_sig2),'-.',label='with OG compensation')
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