import matplotlib.pyplot as plt
from numpy.ma import masked_array

from ekarus.e2e.utils.image_utils import showZoomCenter, myimshow
from ekarus.e2e.single_stage_ao_class import SingleStageAO

try:
    import cupy as xp
    xptype = xp.float32
    print('Now using GPU acceleration!')
except:
    import numpy as xp
    xptype = xp.float64

GoodSeeing = '20250908_130000'
BadSeeing = '20250909_160000'

def main(tn:str=GoodSeeing, lambdaInM=1000e-9, starMagnitude=0, modulationAngleInLambdaOverD=3, show:bool=False):
    print('Initializing devices ...')
    ssao = SingleStageAO(tn, xp=xp)

    ssao.set_wavelength(lambdaInM=lambdaInM)
    ssao.set_star_magnitude(starMagnitude)
    ssao.set_modulation_angle(modulationAngleInLambdaOverD)

    # 1. Define the subapertures using a piston input wavefront
    print('Defining the detector subaperture masks ...')
    ssao.define_subaperture_masks()
    if show:
        try:
            detector_image = ssao.ccd.last_frame
            if xp.__name__ == 'cupy':
                detector_image = detector_image.get()
            plt.figure()
            plt.imshow(detector_image,origin='lower')
            plt.colorbar()
            plt.title('Subaperture masks')
            plt.show()
        except:
            pass


    # 2. Define the system modes
    print('Obtaining the Karhunen-Loeve mirror modes ...')
    KL, m2c = ssao.define_KL_modes(zern_modes = 5)
    if show:
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


    # 3. Calibrate the system
    print('Calibrating the KL modes ...')
    Rec, IM = ssao.calibrate_modes(KL, amps=0.2)
    if show:
        IM_std = xp.std(IM,axis=0)
        if xp.__name__ == 'cupy':
            IM_std = IM_std.get()
        plt.figure()
        plt.plot(IM_std,'-o')
        plt.grid()
        plt.title('Interaction matrix standard deviation')
        plt.show()


    # 4. Get atmospheric phase screen
    print('Initializing turbulence ...')
    ssao.initialize_turbulence(N=20)
    screen = ssao.get_phase_screen(dt=0)
    if show:
        if xp.__name__ == 'cupy':
            screen = screen.get()
        plt.figure()
        plt.imshow(screen, cmap='RdBu')
        plt.colorbar()
        plt.title('Atmo screen')


    # 5. Perform the iteration
    print('Running the loop ...')
    electric_field_amp = 1-ssao.cmask

    dt = 1/1500 #1e-3
    g = 0.05

    Nits = 300
    Nmodes = 467

    Nsteps_delay = 2 # accounts for CCD reading time + time for DM to reach cmd steady-state

    # Define variables
    mask_len = int(xp.sum(1-ssao.dm.mask))
    dm_shape = xp.zeros(mask_len, dtype=xptype)
    dm_cmd = xp.zeros(ssao.dm.Nacts, dtype=xptype)
    dm_phase = xp.zeros_like(ssao.cmask, dtype=xptype)

    # Save telemetry
    dm_cmds = xp.zeros([Nits+Nsteps_delay,ssao.dm.Nacts])
    dm_phases = xp.zeros([Nits,mask_len])
    residual_phases = xp.zeros([Nits,mask_len])
    input_phases = xp.zeros([Nits,mask_len])
    detector_images = xp.zeros([Nits,ssao.ccd.detector_shape[0],ssao.ccd.detector_shape[1]])

    for i in range(Nits):
        print(f'\rIteration {i+1}/{Nits}', end='')
        tt = dt*i
        input_phase = ssao.get_phase_screen(tt)
        input_phase -= xp.mean(input_phase[~ssao.cmask])

        dm_shape = ssao.dm.IFF @ dm_cmds[i,:]*(2*xp.pi)/lambdaInM # convert to radians
        dm_phase[~ssao.cmask] = dm_shape
        dm_phase = xp.reshape(dm_phase, ssao.cmask.shape)

        residual_phase = input_phase - dm_phase

        modes = ssao.perform_loop_iteration(dt, residual_phase, Rec)
        cmd = m2c[:,0:Nmodes] @ modes[0:Nmodes]
        dm_cmd += cmd*g
        dm_cmds[i+Nsteps_delay,:] = dm_cmd*lambdaInM/(2*xp.pi) # convert to meters
        # dm_cmd += cmd*g
        # dm_shape = ssao.dm.IFF @ dm_cmd

        # Save telemetry
        input_phases[i,:] = input_phase[~ssao.cmask]
        dm_phases[i,:] = dm_phase[~ssao.cmask]
        detector_images[i,:,:] = ssao.ccd.last_frame
        dm_cmds[i,:] = dm_cmd
        residual_phases[i,:] = residual_phase[~ssao.cmask]
    print('')


    #################### Post-processing ##############################
    electric_field_amp = 1-ssao.cmask
    electric_field = electric_field_amp * xp.exp(-1j*residual_phase)
    field_on_focal_plane = xp.fft.fftshift(xp.fft.fft2(electric_field))
    psf = abs(field_on_focal_plane)**2

    sig2 = xp.std(residual_phases,axis=-1)**2
    input_sig2 = xp.std(input_phases,axis=-1)**2 

    print('Saving telemetry to .fits ...')
    cmask = ssao.cmask.get() if xp.__name__ == 'cupy' else ssao.cmask.copy()
    masked_input_phases = xp.zeros([Nits,ssao.cmask.shape[0],ssao.cmask.shape[1]], dtype=xptype)
    masked_dm_phases = xp.zeros([Nits,ssao.cmask.shape[0],ssao.cmask.shape[1]], dtype=xptype)
    masked_residual_phases = xp.zeros([Nits,ssao.cmask.shape[0],ssao.cmask.shape[1]], dtype=xptype)

    aux = xp.zeros_like(ssao.cmask,dtype=xptype)

    if xp.__name__ == 'cupy':
        masked_input_phases = masked_input_phases.get()
        masked_dm_phases = masked_dm_phases.get()
        masked_residual_phases = masked_residual_phases.get()

        input_phases = input_phases.get()
        dm_phases = dm_phases.get()
        residual_phases = residual_phases.get()
        dm_cmds = dm_cmds.get()
        detector_images = detector_images.get()

        aux = aux.get()

    for i in range(Nits):
        aux[~cmask] = input_phases[i,:]
        aux = (aux).reshape(cmask.shape)
        masked_input_phases[i,:,:] = masked_array(aux, cmask)

        aux[~cmask] = dm_phases[i,:]
        aux = (aux).reshape(cmask.shape)
        masked_dm_phases[i,:,:]= masked_array(aux, cmask)

        aux[~cmask] = residual_phases[i,:]
        aux = (aux).reshape(cmask.shape)
        masked_residual_phases[i,:,:]= masked_array(aux, cmask)

    import os
    atmo_phases_path = os.path.join(ssao.savepath,'AtmoPhases.fits')
    dm_phases_path = os.path.join(ssao.savepath,'DMphases.fits')
    res_phases_path = os.path.join(ssao.savepath,'ResPhases.fits')
    detector_imgs_path = os.path.join(ssao.savepath,'DetectorFrames.fits')
    dm_cmds_path = os.path.join(ssao.savepath,'DMcommands.fits')

    from ekarus.e2e.utils import my_fits_package as myfits
    myfits.save_fits(atmo_phases_path, masked_input_phases)
    myfits.save_fits(dm_phases_path, masked_dm_phases)
    myfits.save_fits(res_phases_path, masked_residual_phases)
    myfits.save_fits(detector_imgs_path, detector_images)
    myfits.save_fits(dm_cmds_path, dm_cmds)

    # print(f'Number of collected photons: {xp.sum(detector_image):1.0f}')

    ########################## Plotting ###############################
    if xp.__name__ == 'cupy': # Convert to numpy for plotting
        psf = psf.get()
        input_sig2 = input_sig2.get()
        sig2 = sig2.get()

    plt.figure(figsize=(9,9))
    plt.subplot(2,2,1)
    # plt.imshow(masked_array(input_phase, mask = cmask),origin='lower',cmap='RdBu')
    # plt.colorbar()
    # plt.title(f'Input: Strehl ratio = {xp.exp(-input_sig2[-1]):1.3f}')
    myimshow(masked_input_phases[-1],title=f'Input: Strehl ratio = {xp.exp(-input_sig2[-1]):1.3f}',cmap='RdBu',shrink=0.8)
    w = ssao.pupilSizeInPixels + 10
    H,W = ssao.cmask.shape
    plt.xlim([W//2-w//2, W//2+w//2])
    plt.ylim([H//2-w//2, H//2+w//2])

    pixelsPerMAS = ssao.pixelsPerRadian*180/xp.pi*3600*1000

    plt.subplot(2,2,2)
    showZoomCenter(psf, pixelsPerMAS, \
    title = f'PSF: Strehl ratio = {xp.exp(-sig2[-1]):1.3f}',cmap='inferno', \
    xlabel='[mas]', ylabel='[mas]', shrink=0.8)

    plt.subplot(2,2,3)
    # plt.imshow(detector_image,origin='lower')
    # plt.colorbar()
    # plt.title('Detector image')
    myimshow(detector_images[-1], title = 'Detector image', shrink=0.8)

    plt.subplot(2,2,4)
    ssao.dm.plot_position(dm_cmds[-1])
    plt.title('Mirror command')

    plt.figure(figsize=(2.4*Nits/10,2.4))
    plt.plot(input_sig2,'-o',label='open loop')
    plt.plot(sig2,'-o',label='closed loop')
    plt.legend()
    plt.grid()
    plt.ylabel(r'$\sigma^2 [rad^2]$')
    plt.xlabel('# iteration')
    x_ticks = (xp.arange(Nits//10)*10).get() if xp.__name__ == 'cupy' else xp.arange(Nits//10)*10
    plt.xticks(x_ticks)
    plt.xlim([-0.5,Nits-0.5])

    plt.figure()
    plt.plot(xp.exp(-xp.array(sig2)).get(),'-o')
    plt.grid()
    plt.title(f'magV={starMagnitude},gain={g:1.3f},mod={modulationAngleInLambdaOverD:1.1f}')
    plt.ylabel('Strehl ratio')
    plt.xlabel('Iteration number')

    plt.show()

    return ssao

