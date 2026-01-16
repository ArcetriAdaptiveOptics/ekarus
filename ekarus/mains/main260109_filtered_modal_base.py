import xupy as xp
from numpy.ma import masked_array

import matplotlib.pyplot as plt

from ekarus.e2e.utils.image_utils import reshape_on_mask
from ekarus.e2e.single_stage_ao_class import SingleStageAO


def main(tn:str='filtered_modalbase',
         gain_list=None, 
         show_contrast:bool =False,
         optimize_gain:bool=False, 
         atmo_tn='nine_layers_25mOS',
         lambdaRef:float=700e-9):
    
    if gain_list is not None:
        optimize_gain = True

    if optimize_gain is True:
        if gain_list is not None:
            gain_vec = xp.array(gain_list)
        else:
            gain_vec = xp.arange(1,11)/10

    ssao = SingleStageAO(tn)
    ssao.initialize_turbulence(tn=atmo_tn)

    amp = 50e-9
    if ssao.sc.modulationAngleInLambdaOverD < 1.0:
        amp = 25e-9

    ff = lambda f: 0.0 + 1.0*f*(f<=8)
    KL, m2c = ssao.define_KL_modes(ssao.dm, zern_modes=2, freq_filter=ff, save_prefix='filt_')
    _, IM = ssao.compute_reconstructor(ssao.sc, KL, ssao.pyr.lambdaInM, ampsInM=amp, save_prefix='filt_')
    ssao.sc.load_reconstructor(IM,m2c)
    ssao.KL = KL

    pupil = 1-ssao.cmask.copy()
    display_modes(pupil, xp.asnumpy(KL.T), N=8)

    it_ss = max(200,int(ssao.Nits//2))
    err2, in_err2 = ssao.run_loop(lambdaRef, ssao.starMagnitude, save_prefix='filt_')

    if optimize_gain is True:
        print('Finding best gains:')
        N = len(gain_vec)
        SR_vec = xp.zeros(N)
        for jj in range(N):
            g = gain_vec[jj]
            ssao.sc.set_new_gain(g)
            err2, _ = ssao.run_loop(lambdaRef, ssao.starMagnitude, save_prefix='filt_')
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

    if show_contrast:
        ssao.plot_contrast(lambdaRef, frame_ids=xp.arange(ssao.Nits-100,ssao.Nits).tolist(), save_prefix='filt_')

    tvec = xp.asnumpy(xp.arange(ssao.Nits)*ssao.dt*1e+3)

    plt.figure()#figsize=(1.7*Nits/10,3))
    plt.plot(tvec,xp.asnumpy(in_err2),'-.',label='open loop')
    plt.plot(tvec,xp.asnumpy(err2),'-.',label='closed loop')
    plt.legend()
    plt.grid()
    plt.xlim([0.0,tvec[-1]])
    plt.xlabel('Time [ms]')
    plt.ylabel(r'$\sigma^2 [rad^2]$')
    plt.gca().set_yscale('log')

    plt.show()

    return ssao

if __name__ == '__main__':
    ssao = main()

def plot_mode_j(pupil,Mat,j:int,title:str=''):
    mode = reshape_on_mask(Mat[:,j],(1-pupil).astype(bool))
    plt.imshow(masked_array(xp.asnumpy(mode),xp.asnumpy(1-pupil)),origin='lower',cmap='RdBu')
    plt.colorbar()
    plt.axis('off')
    plt.title(title)

def display_modes(pupil, Mat, N:int=8):
    nModes = xp.shape(Mat)[1]
    plt.figure(figsize=(2.25*N,12))
    for i in range(N):
        plt.subplot(6,N,i+1)
        plot_mode_j(pupil,Mat,i,title=f'Mode {i}')
        plt.subplot(6,N,i+1+N)
        plot_mode_j(pupil,Mat,i+N,title=f'Mode {i+N}')

        plt.subplot(6,N,i+1+N*2)
        plot_mode_j(pupil,Mat,nModes//2-N+i,title=f'Mode {nModes//2-N+i}')
        plt.subplot(6,N,i+1+N*3)
        plot_mode_j(pupil,Mat,nModes//2+N+i,title=f'Mode {nModes//2+N+i}')

        plt.subplot(6,N,i+1+N*4)
        plot_mode_j(pupil,Mat,nModes-2*N+i,title=f'Mode {nModes-2*N+i}')
        plt.subplot(6,N,i+1+N*5)
        plot_mode_j(pupil,Mat,nModes-N+i,title=f'Mode {nModes-N+i}')