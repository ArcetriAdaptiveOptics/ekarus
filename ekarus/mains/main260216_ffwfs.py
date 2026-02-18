import xupy as xp
import matplotlib.pyplot as plt

from ekarus.e2e.woofer_tweeter_ao_class import WooferTweeterAO
from ekarus.e2e.utils.image_utils import reshape_on_mask
from numpy.ma import masked_array

from ekarus.analytical.fourier_modes import define_fourier_basis
import ekarus.e2e.utils.my_fits_package as myfits
import os

from ekarus.e2e.devices.slope_computer import SlopeComputer
from ekarus.e2e.devices.pyr3_wfs import Pyr3WFS


# Initialization
wt = WooferTweeterAO('test_wfs')
KL,_ = wt.define_KL_modes(wt.dm,zern_modes=2)

fourier_modes_path = os.path.join(wt.savecalibpath,'fourier_modes.fits')
try:
    FM = xp.asarray(myfits.read_fits(fourier_modes_path))
except FileNotFoundError:
    FM = define_fourier_basis(wt.cmask,max_freq=40)
    myfits.save_fits(fourier_modes_path,FM)

lambdaInM = wt.wfs1.lambdaInM
lambdaOverD = wt.wfs1.lambdaInM/wt.pupilSizeInM
ef_amp = 1-wt.cmask
m2rad = 2*xp.pi/lambdaInM

ccd = wt.ccd1

def get_signal(in_ef,wfs,roi_mask,norm_amp=None):
    push_int = wfs.get_intensity(in_ef,lambdaOverD)
    push_img = ccd.image_on_detector(push_int,photon_flux=None)
    pull_int = wfs.get_intensity(xp.conj(in_ef),lambdaOverD)
    pull_img = ccd.image_on_detector(pull_int,photon_flux=None)
    if norm_amp is None:
        norm_amp = 1.0
    signal = (push_img[roi_mask]-pull_img[roi_mask])/(2*norm_amp)
    return signal

def get_throughput(wfs,roi_mask):
    intensity = wfs.get_intensity(ef_amp.astype(xp.cfloat),lambdaOverD)
    frame = ccd.image_on_detector(intensity,photon_flux=None)
    ref_signal = frame[roi_mask]
    thrp = xp.sum(ref_signal)/xp.sum(frame)
    return thrp, ref_signal

def plot_mode_j(pupil,Mat,j:int,title:str=''):
    mode = reshape_on_mask(Mat[:,j],(1-pupil).astype(bool))
    plt.imshow(masked_array(xp.asnumpy(mode),xp.asnumpy(1-pupil)),origin='lower',cmap='RdBu')
    plt.colorbar()
    plt.axis('off')
    plt.title(title)

def display_modes(pupil, Mat, N:int=8):
    nModes = xp.shape(Mat)[1]
    plt.figure(figsize=(2*N,10))
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

# Devices
pyr = wt.wfs1
zwfs = wt.wfs2

# 3-sided pyramid
subapertureSize = 60
subapPixSep = 60
rebin = wt.wfs1.oversampling*wt.pupilSizeInPixels/max(wt.ccd1.detector_shape)
vertex_angle = 2*xp.pi*wt.wfs1.lambdaInM/wt.pupilSizeInM*((xp.floor(subapertureSize+1.0)+subapPixSep)/(2*xp.cos(xp.pi/6)))*rebin/2
pyr3 = Pyr3WFS(vertex_angle=vertex_angle,oversampling=wt.wfs1.oversampling,sensorLambda=wt.wfs1.lambdaInM)
sc3 = SlopeComputer(pyr3,wt.ccd1,{'modulationInLambdaOverD':0.0})
args = {'zero_phase': ef_amp.astype(xp.cfloat), 'lambdaOverD': lambdaOverD, 'Npix': subapertureSize, 'centerObscurationInPixels': 0.0}
sc3.calibrate_sensor(tn=wt._tn, prefix_str='pyr3_',**args)


def n_norm(vec,n:int=2):
    if n == 1:
        norm = xp.sum(xp.abs(vec))
    else:
        norm = xp.sqrt(xp.sum(vec**2))
    return norm

def plot_sensitivites(Nmodes:int,amp:float,rMods,dotSizes,n:int=1,
                      use_full_frame:bool=False,use_fourier_modes:bool=False):

    if use_fourier_modes:
        mode_basis = FM
    else:
        mode_basis = KL

    # ROI masks
    if use_full_frame is True:
        pyr_roi = xp.ones(wt.ccd1.detector_shape).astype(bool)
        zwfs_roi = xp.ones(wt.ccd1.detector_shape).astype(bool)
        pyr3_roi = xp.ones(wt.ccd1.detector_shape).astype(bool)
    else:
        pyr_roi = (xp.sum(1-wt.sc1._roi_masks,axis=0)).astype(bool)
        zwfs_roi = (1-wt.sc2._roi_masks).astype(bool)
        pyr3_roi = (xp.sum(1-sc3._roi_masks,axis=0)).astype(bool)

    # Initialize variables
    zsens = xp.zeros([len(dotSizes),Nmodes])
    pyrsens = xp.zeros([len(rMods),Nmodes])
    pyr3sens = xp.zeros([Nmodes])
    combsens = xp.zeros([Nmodes])

    zsens_shot = xp.zeros([len(dotSizes),Nmodes])
    pyrsens_shot = xp.zeros([len(rMods),Nmodes])
    pyr3sens_shot = xp.zeros([Nmodes])
    combsens_shot = xp.zeros([Nmodes])

    # Throughput
    pyr_thrp = xp.zeros(len(rMods))
    pyr_ref = xp.zeros([len(rMods),int(xp.sum(pyr_roi))])
    for k,rMod in enumerate(rMods):
        pyr.set_modulation_angle(rMod,verbose=False)
        pyr_thrp[k], pyr_ref[k,:] = get_throughput(pyr,pyr_roi)

    
    pyr3.set_modulation_angle(0.0,verbose=False)
    pyr3_thrp, pyr3_ref = get_throughput(pyr3,pyr3_roi)

    zwfs_thrp = xp.zeros(len(dotSizes))
    zwfs_ref = xp.zeros([len(dotSizes),int(xp.sum(zwfs_roi))])
    for k,dotSize in enumerate(dotSizes):
        zwfs.change_dot_parameters(dotDelayInradians=0.5,dotSizeInLambdaOverD=dotSize)
        zwfs_thrp[k], zwfs_ref[k,:] = get_throughput(zwfs,zwfs_roi)
    print(f'Throughputs: ZWFS = {zwfs_thrp[0]*1e+2:1.1f}%, PWFS = {pyr_thrp[0]*1e+2:1.1f}%, 3PWFS = {pyr3_thrp*1e+2:1.1f}%')
    print(f'Number of pixels: ZWFS = {xp.sum(zwfs_roi):1.0f}, PWFS = {xp.sum(pyr_roi)}, 3PWFS = {xp.sum(pyr3_roi)}')

    Nsubap = 60
    if use_full_frame:
        Nsubap = xp.sqrt(xp.sum(pyr_roi))/5

    display_modes(1-wt.cmask, xp.asnumpy(mode_basis[:Nmodes,:].T), N=8)

    plt.figure(figsize=(15,4.5))
    plt.subplot(1,3,1)
    plt.imshow(masked_array(xp.asnumpy(reshape_on_mask(pyr_ref[0,:],(1-pyr_roi).astype(bool))),mask=xp.asnumpy(1-pyr_roi)),origin='lower',cmap='RdBu')
    plt.subplot(1,3,2)
    plt.imshow(masked_array(xp.asnumpy(reshape_on_mask(pyr3_ref,(1-pyr3_roi).astype(bool))),mask=xp.asnumpy(1-pyr3_roi)),origin='lower',cmap='RdBu')
    plt.subplot(1,3,3)
    plt.imshow(masked_array(xp.asnumpy(reshape_on_mask(zwfs_ref[0,:],(1-zwfs_roi).astype(bool))),mask=xp.asnumpy(1-zwfs_roi)),origin='lower',cmap='RdBu')
    plt.colorbar(shrink=0.7)
    plt.show()

    flux = xp.sum(ef_amp)

    for mode_id in range(Nmodes):
        print(f'\rMode {mode_id+1}/{Nmodes}', end='\r', flush=True)
        mode = mode_basis[mode_id,:]
        in_ef = ef_amp * xp.exp(1j*reshape_on_mask(mode*amp*m2rad,wt.cmask),dtype=xp.cfloat)

        # 3PWFS case
        pyr3_signal = get_signal(in_ef,pyr3,pyr3_roi,norm_amp=amp*m2rad)/flux
        pyr3sens[mode_id] = n_norm(pyr3_signal,n=n)*Nsubap
        # pyr3sens_shot[mode_id] = n_norm(pyr3_signal/(pyr3_ref),n=n)*pyr3_thrp
        pyr3sens_shot[mode_id] = n_norm(pyr3_signal/xp.sqrt(pyr3_ref/flux),n=n)

        # pyWFS case
        for k,rMod in enumerate(rMods):
            pyr.set_modulation_angle(rMod,verbose=False)
            pyr_signal = get_signal(in_ef,pyr,pyr_roi,norm_amp=amp*m2rad)/flux
            pyrsens[k,mode_id] = n_norm(pyr_signal,n=n)*Nsubap
            # pyrsens_shot[k,mode_id] = n_norm(pyr_signal/(pyr_ref[k,:]),n=n)*pyr_thrp[k]
            pyrsens_shot[k,mode_id] = n_norm(pyr_signal/xp.sqrt(pyr_ref[k,:]/flux),n=n)

        # zWFS case
        for k,dotSize in enumerate(dotSizes):
            zwfs.change_dot_parameters(dotDelayInradians=0.5,dotSizeInLambdaOverD=dotSize)
            z_signal = get_signal(in_ef,zwfs,zwfs_roi,norm_amp=amp*m2rad)/flux
            zsens[k,mode_id] = n_norm(z_signal,n=n)*Nsubap
            # zsens_shot[k,mode_id] = n_norm(z_signal/(zwfs_ref[k,:]),n=n)*zwfs_thrp[k]
            zsens_shot[k,mode_id] = n_norm(z_signal/xp.sqrt(zwfs_ref[k,:]/flux),n=n)

        # Combined case
        pyr.set_modulation_angle(0.0,verbose=False)
        pyrsig = get_signal(in_ef,pyr,pyr_roi,norm_amp=amp*m2rad)/flux
        zwfs.change_dot_parameters(dotDelayInradians=0.5,dotSizeInLambdaOverD=2.0)
        zsig = get_signal(in_ef,zwfs,zwfs_roi,norm_amp=amp*m2rad)/flux
        combsens[mode_id] = n_norm(xp.hstack([pyrsig,zsig]),n=n)*Nsubap/xp.sqrt(2)
        combsens_shot[mode_id] = n_norm(xp.hstack([pyrsig/xp.sqrt(pyr_ref[0,:]/flux),zsig/xp.sqrt(zwfs_ref[0,:]/flux)]),n=n)/xp.sqrt(2)

    x = xp.asnumpy(xp.arange(Nmodes)+1)

    plt.figure(figsize=(14,5))
    plt.subplot(1,2,2)
    plt.plot(x,xp.asnumpy(pyrsens[0,:]),'-.',label='Unmod pyWFS')
    plt.plot(x,xp.asnumpy(pyr3sens),'-.',label='Unmod 3pyWFS')
    plt.plot(x,xp.asnumpy(zsens[0,:]),'-.',label=f'zWFS')
    plt.plot(x,xp.asnumpy(zsens[2,:]),'-.',label=f'z2WFS')
    plt.grid()
    plt.legend()
    if use_fourier_modes is False:
        plt.xscale('log')
        plt.xlabel('KL mode')
    else:
        plt.xlabel('Fourier mode')
    plt.xlim([1,Nmodes])
    plt.title(f'RON sensitivity\n(60x60 subapertures, '+r'$\lambda$='+f'{wt.wfs2.lambdaInM*1e+9:1.0f}nm)')
    plt.subplot(1,2,1)
    plt.plot(x,xp.asnumpy(pyrsens_shot[0,:]),'-.',label='Unmod pyWFS')
    plt.plot(x,xp.asnumpy(pyr3sens_shot),'-.',label='Unmod 3pyWFS')
    plt.plot(x,xp.asnumpy(zsens_shot[0,:]),'-.',label=f'zWFS')
    plt.plot(x,xp.asnumpy(zsens_shot[2,:]),'-.',label=f'z2WFS')
    plt.grid()
    plt.legend()
    if use_fourier_modes is False:
        plt.xscale('log')
        plt.xlabel('KL mode')
    else:
        plt.xlabel('Fourier mode')
    plt.xlim([1,Nmodes])
    plt.title(f'Shot noise sensitivity\n(60x60 subapertures, '+r'$\lambda$='+f'{wt.wfs2.lambdaInM*1e+9:1.0f}nm)')

    plt.figure(figsize=(14,5))
    plt.subplot(1,2,2)
    for k,rMod in enumerate(rMods):
        plt.plot(x,xp.asnumpy(pyrsens[k,:]),'-.',label=f'pyWFS {rMod:1.1f}'+r' $\lambda/D$')
    plt.grid()
    plt.legend()
    if use_fourier_modes is False:
        plt.xscale('log')
        plt.xlabel('KL mode')
    else:
        plt.xlabel('Fourier mode')
    plt.xlim([1,Nmodes])
    plt.title(f'RON sensitivity\n(60x60 subapertures, '+r'$\lambda$='+f'{wt.wfs2.lambdaInM*1e+9:1.0f}nm)')
    plt.subplot(1,2,1)
    for k,rMod in enumerate(rMods):
        plt.plot(x,xp.asnumpy(pyrsens_shot[k,:]),'-.',label=f'pyWFS {rMod:1.1f}'+r' $\lambda/D$')
    plt.grid()
    plt.legend()
    if use_fourier_modes is False:
        plt.xscale('log')
        plt.xlabel('KL mode')
    else:
        plt.xlabel('Fourier mode')
    plt.xlim([1,Nmodes])
    plt.title(f'Shot noise sensitivity\n(60x60 subapertures, '+r'$\lambda$='+f'{wt.wfs2.lambdaInM*1e+9:1.0f}nm)')

    plt.figure(figsize=(14,5))
    plt.subplot(1,2,2)
    plt.plot(x,xp.asnumpy(pyrsens[0,:]),':',label='Unmod pyWFS')
    for k,dotSize in enumerate(dotSizes):
        plt.plot(x,xp.asnumpy(zsens[k,:]),'-.',label=f'zWFS {dotSize:1.1f}'+r' $\lambda/D$')
    plt.grid()
    plt.legend()
    if use_fourier_modes is False:
        plt.xscale('log')
        plt.xlabel('KL mode')
    else:
        plt.xlabel('Fourier mode')
    plt.xlim([1,Nmodes])
    plt.title(f'RON sensitivity\n(60x60 subapertures, '+r'$\lambda$='+f'{wt.wfs2.lambdaInM*1e+9:1.0f}nm)')
    plt.subplot(1,2,1)
    plt.plot(x,xp.asnumpy(pyrsens_shot[0,:]),':',label='Unmod pyWFS')
    for k,dotSize in enumerate(dotSizes):
        plt.plot(x,xp.asnumpy(zsens_shot[k,:]),'-.',label=f'zWFS {dotSize:1.1f}'+r' $\lambda/D$')
    plt.grid()
    plt.legend()
    if use_fourier_modes is False:
        plt.xscale('log')
        plt.xlabel('KL mode')
    else:
        plt.xlabel('Fourier mode')
    plt.xlim([1,Nmodes])
    plt.title(f'Shot noise sensitivity\n(60x60 subapertures, '+r'$\lambda$='+f'{wt.wfs2.lambdaInM*1e+9:1.0f}nm)')

    plt.figure(figsize=(14,5))
    plt.subplot(1,2,2)
    plt.plot(x,xp.asnumpy(pyrsens[0,:]),'-.',label='Unmod pyWFS')
    plt.plot(x,xp.asnumpy(zsens[-1,:]),'-.',label=f'z2WFS')
    plt.plot(x,xp.asnumpy(combsens),':',label=f'Combined')
    plt.grid()
    plt.legend()
    if use_fourier_modes is False:
        plt.xscale('log')
        plt.xlabel('KL mode')
    else:
        plt.xlabel('Fourier mode')
    plt.xlim([1,Nmodes])
    plt.title(f'RON sensitivity\n(60x60 subapertures, '+r'$\lambda$='+f'{wt.wfs2.lambdaInM*1e+9:1.0f}nm)')
    plt.subplot(1,2,1)
    plt.plot(x,xp.asnumpy(pyrsens_shot[0,:]),'-.',label='Unmod pyWFS')
    plt.plot(x,xp.asnumpy(zsens_shot[-1,:]),'-.',label=f'z2WFS')
    plt.plot(x,xp.asnumpy(combsens_shot),':',label=f'Combined')
    plt.grid()
    plt.legend()
    if use_fourier_modes is False:
        plt.xscale('log')
        plt.xlabel('KL mode')
    else:
        plt.xlabel('Fourier mode')
    plt.xlim([1,Nmodes])
    plt.title(f'Shot noise sensitivity\n(60x60 subapertures, '+r'$\lambda$='+f'{wt.wfs2.lambdaInM*1e+9:1.0f}nm)')

    plt.show()


if __name__ == '__main__':
    rMods = xp.array([0.0,0.5,2.0,4.0,6.0])
    dotSizes = xp.array([1.0,1.5,2.0])
    plot_sensitivites(Nmodes=400,amp=15e-9,rMods=rMods,dotSizes=dotSizes,n=2,use_full_frame=True,use_fourier_modes=True)


