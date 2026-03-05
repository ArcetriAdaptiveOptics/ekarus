import os
import xupy as xp
import matplotlib.pyplot as plt

from ekarus.e2e.woofer_tweeter_ao_class import WooferTweeterAO
from ekarus.analytical.turbulence_layers import TurbulenceLayers
from ekarus.e2e.utils.root import atmopath

from ekarus.e2e.utils import my_fits_package as myfits
from ekarus.e2e.utils.image_utils import reshape_on_mask
from numpy.ma import masked_array


# Initialization
wt = WooferTweeterAO('test_wfs')
KL,_ = wt.define_KL_modes(wt.dm,zern_modes=2)

lambdaInM = wt.wfs1.lambdaInM
D = wt.pupilSizeInM
ef_amp = 1-wt.cmask
m2rad = 2*xp.pi/lambdaInM
amp = 10e-9

zern_sc = wt.sc2
pyr_sc = wt.sc1

# Generate phase-screens
Nscreens = 48
L0 = 25
r0 = 10e-2
atmo_dir = os.path.join(atmopath,'optical_gains')
if not os.path.exists(atmo_dir):
    os.mkdir(atmo_dir)
atmo_path = os.path.join(atmo_dir,f'pix{wt.pupilSizeInPixels:1.0f}_{Nscreens:1.0f}layers.fits')

layers = TurbulenceLayers(xp.repeat(xp.array([r0]),Nscreens), L0, savepath=atmo_path)
layers.generate_phase_screens(wt.pupilSizeInPixels, wt.pupilSizeInM)
layers.rescale_phasescreens() # rescale in meters
layers._L0 = L0
layers._r0 = r0


def filter_screens(r0, Nmodes:int, corr_level:float, show:bool=False):
    layers.update_r0(r0)
    screens = layers.phase_screens
    N = len(screens)
    corr_screens = xp.zeros([N,int(xp.sum(1-wt.cmask))])
    modes2phase = KL[:Nmodes,:].T
    phase2modes = xp.linalg.pinv(modes2phase)
    if show:
        psd = xp.zeros(KL.shape[0])
        psd_corr = xp.zeros(KL.shape[0])
        phi2m = xp.linalg.pinv(KL.T)
    for j in range(N):
        phase = screens[j][~wt.cmask]
        phase -= xp.mean(phase)
        modes = phase2modes @ phase
        modes *= corr_level
        rec_phase = modes2phase @ modes
        res_phase = phase - rec_phase
        corr_screens[j,:] = res_phase
        if show:
            modes = phi2m @ phase
            psd += modes**2
            corr_modes = phi2m @ res_phase
            psd_corr += corr_modes**2
    res_phase_rms = xp.mean(xp.sqrt(xp.mean(corr_screens**2,axis=0)))
    SR = xp.exp(-(res_phase_rms/lambdaInM*2*xp.pi)**2)
    seeing = 0.98*500e-9/r0*180/xp.pi*3600
    prefix = f'SR{SR*1e+2:1.1f}%_{seeing:1.2f}seeing_'
    mask_cube = xp.asnumpy(xp.stack([wt.cmask for _ in range(N)]))
    res_phases = xp.stack([reshape_on_mask(corr_screens[i, :], wt.cmask) for i in range(N)])
    ma_res_screens = masked_array(xp.asnumpy(res_phases), mask=mask_cube)
    myfits.save_fits(os.path.join(atmo_dir,prefix+'res_screens.fits'),ma_res_screens)
    if show:
        psd = xp.sqrt(psd/N)
        psd_corr = xp.sqrt(psd_corr/N)
        plot_data([psd*1e+9,psd_corr*1e+9],labels=[f'{seeing:1.2f}" turbulence',f'AO correction'],
                  ylabel='RMS [nm]',title='Phase screens PSD')
    return corr_screens, prefix


def plot_data(data:list, labels:list, title:str='', ylabel:str=''):
    N = len(data)
    L = len(data[0])
    x = xp.asnumpy(xp.arange(L)+1)
    plt.figure()
    for i in range(N):
        plt.plot(x,xp.asnumpy(data[i]),label=labels[i])
    plt.legend()
    plt.grid()
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel(ylabel)
    plt.xlabel('KL mode #')
    plt.title(title)


def compute_pyr_ogs(corr_screens, Nmodes:int, rMod:float=0.0, no_fitting_ogs:bool=False, show:bool=False, prefix=''):
    wt.wfs1.set_modulation_angle(rMod)
    imprefix = f'rMod{rMod:1.1f}_pyr_' 
    saveprefix = prefix+imprefix
    ogs = wt.calibrate_optical_gains_from_precorrected_screens(corr_screens, wt.sc1, KL, return_lo_ogs=no_fitting_ogs, imprefix=imprefix,
                                                              lambdaInM=lambdaInM, ampsInM=10e-9, save_prefix=saveprefix, Nmodes=Nmodes)
    if no_fitting_ogs:
        pyr_ogs = ogs[0]
        lo_ogs = ogs[1]
    else:
        pyr_ogs = ogs
        lo_ogs = None
    if show:
        if lo_ogs is not None:
            plot_data(data=[pyr_ogs,lo_ogs],labels=['','no fitting error'],title='Optical gains')
        else:
            plot_data(data=[pyr_ogs],labels=[''],title='Optical gains')
    return pyr_ogs, lo_ogs


def compute_zwfs_ogs(corr_screens, Nmodes:int, dotDelay:float=0.5, dotSize:float=2.0, show:bool=False,prefix=''):
    wt.wfs2.change_dot_parameters(dotDelayInradians=dotDelay,dotSizeInLambdaOverD=dotSize)
    imprefix = f'delay{dotDelay:1.2f}_z{dotSize:1.0f}wfs_' 
    saveprefix = prefix+imprefix
    zogs = wt.calibrate_optical_gains_from_precorrected_screens(corr_screens, wt.sc2, KL,
                                                              lambdaInM=lambdaInM, ampsInM=10e-9,  imprefix=imprefix,
                                                              save_prefix=saveprefix, Nmodes=Nmodes)
    if show:
        plot_data(data=[zogs],labels=[''],title='Optical gains')
    return zogs


if __name__ == '__main__':

    r0 = 10e-2
    Nmodes = 620
    corr_level = 1.0-xp.linspace(0.01,0.9,Nmodes)

    corr_screens, prefix = filter_screens(r0, Nmodes=Nmodes, corr_level=corr_level, show=True)
    pyr_ogs, lo_ogs = compute_pyr_ogs(corr_screens, Nmodes=Nmodes, no_fitting_ogs=True, show=True, prefix=prefix)
    pyr1_ogs, lo_ogs = compute_pyr_ogs(corr_screens, Nmodes=Nmodes, rMod=1, show=True, prefix=prefix)
    pyr2_ogs, lo_ogs = compute_pyr_ogs(corr_screens, Nmodes=Nmodes, rMod=2, show=True, prefix=prefix)
    z2ogs = compute_zwfs_ogs(corr_screens, Nmodes=Nmodes, prefix=prefix)
    zogs = compute_zwfs_ogs(corr_screens, Nmodes=Nmodes, dotSize=1.0, prefix=prefix)

    seeing = 0.98*500e-9/r0*180/xp.pi*3600
    title = f'Optical gains\n({seeing:1.2f}" seeing, {Nmodes:1.0f} KL modes corrected)'
    plot_data(data=[pyr_ogs,pyr1_ogs,pyr2_ogs,zogs,z2ogs],labels=['pyWFS',r'pyWFS 1 $\lambda/D$',r'pyWFS 2 $\lambda/D$','zWFS','z2WFS'])
    plt.show()