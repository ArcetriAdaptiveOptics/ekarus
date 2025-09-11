import matplotlib.pyplot as plt
from numpy.ma import masked_array

from ekarus.e2e.utils.image_utils import showZoomCenter, myimshow, image_grid
from ekarus.e2e.single_stage_ao_class import SingleStageAO

try:
    import cupy as xp
    xptype = xp.float32
    print('Now using GPU acceleration!')
except:
    import numpy as xp
    xptype = xp.float64


def main(tn:str='exampleTN', Nits:int=500, integratorGain=0.05, lambdaInM=1000e-9, starMagnitude=3, modulationAngleInLambdaOverD=3, show:bool=False):

    # Loop parameters
    loopFrequencyInHz = 1500
    nModes2Correct = 467
    delayInS = 2/loopFrequencyInHz # CCD reading time + time for DM to reach steady-state

    print('Initializing devices ...')
    ssao = SingleStageAO(tn, xp=xp)

    ssao.set_wavelength(lambdaInM=lambdaInM)
    ssao.set_star_magnitude(starMagnitude)
    ssao.set_modulation_angle(modulationAngleInLambdaOverD)
    print(f'Now modulating {ssao.wfs.modulationAngle*180/xp.pi*3600:1.2f} [arcsec] with {ssao.wfs.modulationNsteps:1.0f} modulation steps')

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

    
    # 3.5 Define loop parameters
    dt = 1/loopFrequencyInHz
    modal_gains = xp.zeros(xp.shape(m2c)[1])
    modal_gains[:nModes2Correct] = 1
    nDelaySteps = int(delayInS * loopFrequencyInHz)
    endTime = Nits*dt

    # 4. Get atmospheric phase screen
    print('Initializing turbulence ...')
    ssao.initialize_turbulence(endTime)
    screen = ssao.get_phasescreen_at_time(0)
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

    # Define variables
    mask_len = int(xp.sum(1-ssao.dm.mask))
    dm_shape = xp.zeros(mask_len, dtype=xptype)
    dm_cmd = xp.zeros(ssao.dm.Nacts, dtype=xptype)
    # dm_phase = xp.zeros_like(ssao.cmask, dtype=xptype)
    delta_phase_in_rad = xp.zeros_like(ssao.cmask, dtype=xptype)

    # Save telemetry
    dm_cmds = xp.zeros([Nits+nDelaySteps,ssao.dm.Nacts])
    dm_phases = xp.zeros([Nits,mask_len])
    residual_phases = xp.zeros([Nits,mask_len])
    input_phases = xp.zeros([Nits,mask_len])
    detector_images = xp.zeros([Nits,ssao.ccd.detector_shape[0],ssao.ccd.detector_shape[1]])

    X,Y = image_grid(ssao.cmask.shape, recenter=True, xp=xp)
    tiltX = X[~ssao.cmask]/(ssao.cmask.shape[0]//2)
    tiltY = Y[~ssao.cmask]/(ssao.cmask.shape[1]//2)
    TTmat = xp.stack((tiltX.T,tiltY.T),axis=1)
    pinvTTmat = xp.linalg.pinv(TTmat)

    tt_offload = 0 # tip-tilt offload percentage

    for i in range(Nits):
        print(f'\rIteration {i+1}/{Nits}', end='')
        sim_time = dt*i

        atmo_phase = ssao.get_phasescreen_at_time(sim_time)
        input_phase = atmo_phase[~ssao.cmask]*lambdaInM/(2*xp.pi) # convert to meters
        input_phase -= xp.mean(input_phase) # remove piston

        # tilt_coeffs = pinvTTmat @ input_phase # get tilt coefficients
        # input_phase -= tt_offload * TTmat @ tilt_coeffs # remove tt_offload% of the tilt

        dm_shape = ssao.dm.IFF @ dm_cmds[i,:]
        residual_phase = input_phase - dm_shape

        delta_phase_in_rad[~ssao.cmask] = residual_phase*(2*xp.pi)/lambdaInM
        delta_phase_in_rad = xp.reshape(delta_phase_in_rad, ssao.cmask.shape)

        modes = ssao.perform_loop_iteration(dt, delta_phase_in_rad, Rec)
        modes *= modal_gains
        cmd = m2c @ modes # cmd = m2c[:,0:nModes2Correct] @ modes[0:nModes2Correct]
        dm_cmd += cmd*integratorGain
        dm_cmds[i+nDelaySteps,:] = dm_cmd*lambdaInM/(2*xp.pi) # convert to meters

        # Save telemetry
        input_phases[i,:] = input_phase
        dm_phases[i,:] = dm_shape
        detector_images[i,:,:] = ssao.ccd.last_frame
        dm_cmds[i,:] = dm_cmd
        residual_phases[i,:] = residual_phase
    print('')


    #################### Post-processing ##############################
    electric_field_amp = 1-ssao.cmask
    electric_field = electric_field_amp * xp.exp(-1j*delta_phase_in_rad)
    field_on_focal_plane = xp.fft.fftshift(xp.fft.fft2(electric_field))
    psf = abs(field_on_focal_plane)**2

    sig2 = xp.std(residual_phases*(2*xp.pi)/lambdaInM,axis=-1)**2
    input_sig2 = xp.std(input_phases*(2*xp.pi)/lambdaInM,axis=-1)**2 

    print('Saving telemetry to .fits ...')
    cmask = ssao.cmask.get() if xp.__name__ == 'cupy' else ssao.cmask.copy()
    masked_input_phases = xp.zeros([Nits,ssao.cmask.shape[0],ssao.cmask.shape[1]], dtype=xptype)
    masked_dm_phases = xp.zeros([Nits,ssao.cmask.shape[0],ssao.cmask.shape[1]], dtype=xptype)
    masked_residual_phases = xp.zeros([Nits,ssao.cmask.shape[0],ssao.cmask.shape[1]], dtype=xptype)
    logPSFs = xp.zeros([Nits,ssao.cmask.shape[0],ssao.cmask.shape[1]], dtype=xptype)

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

        electric_field = electric_field_amp * xp.exp(-1j*xp.asarray(aux*(2*xp.pi)/lambdaInM))
        field_on_focal_plane = xp.fft.fftshift(xp.fft.fft2(electric_field))
        psf = abs(field_on_focal_plane)**2
        logPSFs[i,:,:] = xp.log(psf)

    import os
    atmo_phases_path = os.path.join(ssao.savepath,'AtmoPhases.fits')
    dm_phases_path = os.path.join(ssao.savepath,'DMphases.fits')
    res_phases_path = os.path.join(ssao.savepath,'ResPhases.fits')    
    psfs_path = os.path.join(ssao.savepath,'logPSFs.fits')
    detector_imgs_path = os.path.join(ssao.savepath,'DetectorFrames.fits')
    dm_cmds_path = os.path.join(ssao.savepath,'DMcommands.fits')

    from ekarus.e2e.utils import my_fits_package as myfits
    myfits.save_fits(atmo_phases_path, masked_input_phases)
    myfits.save_fits(dm_phases_path, masked_dm_phases)
    myfits.save_fits(res_phases_path, masked_residual_phases)
    myfits.save_fits(detector_imgs_path, detector_images)
    myfits.save_fits(dm_cmds_path, dm_cmds)
    myfits.save_fits(psfs_path, logPSFs)

    # print(f'Number of collected photons: {xp.sum(detector_image):1.0f}')

    ########################## Plotting ###############################
    if xp.__name__ == 'cupy': # Convert to numpy for plotting
        psf = psf.get()
        input_sig2 = input_sig2.get()
        sig2 = sig2.get()

    plt.figure(figsize=(9,9))
    plt.subplot(2,2,1)
    myimshow(masked_input_phases[-1], \
        title=f'Atmosphere phase [m]\nStrehl ratio = {xp.exp(-input_sig2[-1]):1.3f}',\
        cmap='RdBu',shrink=0.8)
    w = ssao.pupilSizeInPixels + 10
    H,W = ssao.cmask.shape
    plt.xlim([W//2-w//2, W//2+w//2])
    plt.ylim([H//2-w//2, H//2+w//2])
    plt.axis('off')

    pixelsPerMAS = ssao.pixelsPerRadian*180/xp.pi*3600*1000
    plt.subplot(2,2,2)
    showZoomCenter(psf, pixelsPerMAS, shrink=0.8, \
        title = f'Corrected PSF\nStrehl ratio = {xp.exp(-sig2[-1]):1.3f}',cmap='inferno') 

    plt.subplot(2,2,3)
    myimshow(detector_images[-1], title = 'Detector image', shrink=0.8)

    plt.subplot(2,2,4)
    ssao.dm.plot_position(dm_cmds[-1])
    plt.title('Mirror command [m]')
    plt.axis('off')

    plt.figure(figsize=(2*Nits/10,2.4))
    plt.plot(input_sig2,'-o',label='open loop')
    plt.plot(sig2,'-o',label='closed loop')
    plt.legend()
    plt.grid()
    plt.ylabel(r'$\sigma^2 [rad^2]$')
    plt.xlabel('# iteration')
    plt.gca().set_yscale('log')
    # x_ticks = (xp.arange(Nits//10)*10).get() if xp.__name__ == 'cupy' else xp.arange(Nits//10)*10
    # plt.xticks(x_ticks)
    plt.xlim([-0.5,Nits-0.5])

    plt.show()

    return ssao

