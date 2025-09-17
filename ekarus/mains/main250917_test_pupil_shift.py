import matplotlib.pyplot as plt
from numpy.ma import masked_array
import os.path as op

from ekarus.e2e.utils.image_utils import showZoomCenter, myimshow, reshape_on_mask
from ekarus.e2e.test_pupil_shift_class import SingleStageAO

import ekarus.e2e.utils.my_fits_package as myfits

try:
    import cupy as xp
    xptype = xp.float32
    print('Now using GPU acceleration!')
except:
    import numpy as xp
    xptype = xp.float64


def main(tn:str='pupil_shift', wedgeAngInSubaps = 39/24*0.2, show:bool=False):

    print('Initializing devices ...')
    ssao = SingleStageAO(tn, xp=xp)

    lambdaInM, starMagnitude = ssao._config.read_target_pars()

    print('Defining the detector subaperture masks ...')
    ssao.define_subaperture_masks(ssao.slope_computer, lambdaInM)

    print('Obtaining the Karhunen-Loeve mirror modes ...')
    KL, m2c = ssao.define_KL_modes(ssao.dm, ssao.pyr.oversampling, zern_modes=5)

    print('Calibrating the KL modes ...')
    Rec, IM = ssao.calibrate_modes(ssao.slope_computer, KL, lambdaInM, amps=0.2)

    print('Initializing turbulence ...')
    ssao.initialize_turbulence()

    print('Running the loop ...')
    sig2, input_sig2 = ssao.run_loop(lambdaInM, starMagnitude, Rec, m2c, save_telemetry=True)

    sig = xp.zeros([7,ssao.Nits])
    sig[0,:] = input_sig2
    sig[1,:] = sig2

    # Post-processing and plotting
    print('Plotting results ...')
    if show:
        subap_masks = xp.sum(ssao.slope_computer._subaperture_masks,axis=0)
        plt.figure()
        myimshow(subap_masks,title='Subaperture masks')
        plt.show()

        KL = myfits.read_fits(op.join(ssao.savepath,'KLmodes.fits'))
        N=9
        plt.figure(figsize=(2*N,7))
        for i in range(N):
            plt.subplot(4,N,i+1)
            ssao.dm.plot_surface(KL[i,:],title=f'KL Mode {i}')
            plt.subplot(4,N,i+1+N)
            ssao.dm.plot_surface(KL[i+N,:],title=f'KL Mode {i+N}')
            plt.subplot(4,N,i+1+N*2)
            ssao.dm.plot_surface(KL[-i-1-N,:],title=f'KL Mode {xp.shape(KL)[0]-i-1-N}')
            plt.subplot(4,N,i+1+N*3)
            ssao.dm.plot_surface(KL[-i-1,:],title=f'KL Mode {xp.shape(KL)[0]-i-1}')

        IM = myfits.read_fits(op.join(ssao.savepath,'IM.fits'))
        IM_std = xp.std(IM,axis=0)
        plt.figure()
        plt.plot(IM_std,'-o')
        plt.grid()
        plt.title('Interaction matrix standard deviation')

        screen = ssao.get_phasescreen_at_time(0)
        if xp.__name__ == 'cupy':
            screen = screen.get()
        plt.figure()
        plt.imshow(screen, cmap='RdBu')
        plt.colorbar()
        plt.title('Atmo screen')

    # Testing pupil shifts
    phi = xp.linspace(0, xp.pi/2, 5)
    for k in range(len(phi)):
        wedgeX = wedgeAngInSubaps * xp.cos(phi[k])
        wedgeY = wedgeAngInSubaps * xp.sin(phi[k])
        print(f'Now applying a shift of ({wedgeX:1.2f},{wedgeY:1.2f})')
        wedgeShift = (wedgeX, wedgeY)
        sig[k+2,:], _ = ssao.run_loop(lambdaInM, starMagnitude, Rec, m2c, wedgeShift=wedgeShift, save_telemetry=True)


    # Plotting
    masked_input_phases, _, masked_residual_phases, detector_frames, _ = ssao.load_telemetry_data()
    last_res_phase = xp.array(masked_residual_phases[-1,:,:])
    residual_phase = last_res_phase[~ssao.cmask]
    oversampling = 4
    padding_len = int(ssao.cmask.shape[0]*(oversampling-1)/2)
    psf_mask = xp.pad(ssao.cmask, padding_len, mode='constant', constant_values=1)
    electric_field_amp = 1-psf_mask
    input_phase = reshape_on_mask(residual_phase, psf_mask, xp=xp)
    electric_field = electric_field_amp * xp.exp(-1j*xp.asarray(input_phase*(2*xp.pi)/lambdaInM))
    field_on_focal_plane = xp.fft.fftshift(xp.fft.fft2(electric_field))
    psf = abs(field_on_focal_plane)**2
    # psf = abs(ssao.pyr.field_on_focal_plane)**2

    cmask = ssao.cmask.get() if xp.__name__ == 'cupy' else ssao.cmask.copy()
    if xp.__name__ == 'cupy': # Convert to numpy for plotting
        input_sig2 = input_sig2.get()
        sig2 = sig2.get()

    plt.figure(figsize=(9,9))
    plt.subplot(2,2,1)
    myimshow(masked_array(masked_input_phases[-1],cmask), \
        title=f'Atmosphere phase [m]\nStrehl ratio = {xp.exp(-input_sig2[-1]):1.3f}',\
        cmap='RdBu',shrink=0.8)
    plt.axis('off')

    pixelsPerMAS = lambdaInM/ssao.pupilSizeInM/ssao.pyr.oversampling*180/xp.pi*3600*1000
    plt.subplot(2,2,2)
    showZoomCenter(psf, pixelsPerMAS, shrink=0.8, \
        title = f'Corrected PSF\nStrehl ratio = {xp.exp(-sig2[-1]):1.3f}',cmap='inferno') 

    plt.subplot(2,2,3)
    myimshow(detector_frames[-1], title = 'Detector image', shrink=0.8)

    plt.subplot(2,2,4)
    ssao.dm.plot_position()
    plt.title('Mirror command [m]')
    plt.axis('off')

    plt.figure()
    for k in range(sig.shape[0]):
        if hasattr(sig,'get'):
            sig = sig.get()
        if k == 0:
            label = 'input'
        elif k == 1:
            label = 'reference'
        else:
            label = f'shift {wedgeAngInSubaps*24/39*1.5:1.2f} [mm], angle: {phi[k-2]*180/xp.pi:1.0f} [deg]'
        plt.plot(sig[k],'-o',label=label)
    plt.legend()
    plt.grid()
    plt.ylabel(r'$\sigma^2 [rad^2]$')
    plt.xlabel('# iteration')
    plt.gca().set_yscale('log')
    plt.xlim([-0.5,ssao.Nits-0.5])

    plt.show()

    return ssao, sig

