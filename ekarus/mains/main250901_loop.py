import matplotlib.pyplot as plt
from numpy.ma import masked_array

from ekarus.e2e.utils.image_utils import showZoomCenter, myimshow, image_grid, reshape_on_mask
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
    ssao.pyr.set_modulation_angle(modulationAngleInLambdaOverD)
    print(f'Now modulating {ssao.pyr.modulationAngleInLambdaOverD:1.0f} [lambda/D] with {ssao.pyr.modulationNsteps:1.0f} modulation steps')

    # 1. Define the subapertures using a piston input wavefront
    print('Defining the detector subaperture masks ...')
    ssao.define_subaperture_masks()
    if show:
        try:
            detector_image = ssao.ccd.last_frame
            plt.figure()
            myimshow(detector_image,title='Subaperture masks')
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
    dm_cmd = xp.zeros(ssao.dm.Nacts, dtype=xptype)
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

    ttOffloadFrequency = 100 # [Hz]


    for i in range(Nits):
        print(f'\rIteration {i+1}/{Nits}', end='')
        sim_time = dt*i

        atmo_phase = ssao.get_phasescreen_at_time(sim_time)
        input_phase = atmo_phase[~ssao.cmask]*lambdaInM/(2*xp.pi) # convert to meters
        input_phase -= xp.mean(input_phase) # remove piston

        # Tilt offloading
        if i>0 and i % int(loopFrequencyInHz/ttOffloadFrequency) <= 1e-6:
            tt_coeffs = modes[:2]*lambdaInM/(2*xp.pi)
            input_phase -= TTmat @ tt_coeffs

        residual_phase = input_phase - ssao.dm.surface

        # delta_phase_in_rad[~ssao.cmask] = residual_phase*(2*xp.pi)/lambdaInM
        # delta_phase_in_rad = xp.reshape(delta_phase_in_rad, ssao.cmask.shape)
        delta_phase_in_rad = reshape_on_mask(residual_phase*(2*xp.pi)/lambdaInM, ssao.cmask, xp=xp)

        modes = ssao.perform_loop_iteration(dt, delta_phase_in_rad, Rec)
        modes *= modal_gains
        cmd = m2c @ modes # cmd = m2c[:,0:nModes2Correct] @ modes[0:nModes2Correct]
        dm_cmd += cmd*integratorGain
        ssao.dm.set_position(dm_cmd*lambdaInM/(2*xp.pi), absolute=True)
        dm_cmds[i+nDelaySteps,:] = dm_cmd*lambdaInM/(2*xp.pi)
        ssao.dm.set_position(dm_cmd*lambdaInM/(2*xp.pi), absolute=True)
        dm_cmds[i+nDelaySteps,:] = dm_cmd*lambdaInM/(2*xp.pi)

        # Save telemetry
        input_phases[i,:] = input_phase
        dm_phases[i,:] = ssao.dm.surface # dm_shape
        dm_phases[i,:] = ssao.dm.surface # dm_shape
        detector_images[i,:,:] = ssao.ccd.last_frame
        dm_cmds[i,:] = dm_cmd
        residual_phases[i,:] = residual_phase
    print('')


    #################### Post-processing ##############################
    sig2 = xp.std(residual_phases*(2*xp.pi)/lambdaInM,axis=-1)**2
    input_sig2 = xp.std(input_phases*(2*xp.pi)/lambdaInM,axis=-1)**2 

    print('Saving telemetry to .fits ...')
    oversampling = 4
    padding_len = int(ssao.cmask.shape[0]*(oversampling-1)/2)
    psf_mask = xp.pad(ssao.cmask, padding_len, mode='constant', constant_values=1)
    electric_field_amp = 1-psf_mask

    cmask = ssao.cmask.get() if xp.__name__ == 'cupy' else ssao.cmask.copy()
    psf_mask = psf_mask.get() if xp.__name__ == 'cupy' else psf_mask.copy()

    masked_input_phases = xp.zeros([Nits,ssao.cmask.shape[0],ssao.cmask.shape[1]], dtype=xptype)
    masked_dm_phases = xp.zeros([Nits,ssao.cmask.shape[0],ssao.cmask.shape[1]], dtype=xptype)
    masked_residual_phases = xp.zeros([Nits,ssao.cmask.shape[0],ssao.cmask.shape[1]], dtype=xptype)
    logPSFs = xp.zeros([Nits,psf_mask.shape[0],psf_mask.shape[1]], dtype=xptype)

    if xp.__name__ == 'cupy':
        masked_input_phases = masked_input_phases.get()
        masked_dm_phases = masked_dm_phases.get()
        masked_residual_phases = masked_residual_phases.get()

        input_phases = input_phases.get()
        dm_phases = dm_phases.get()
        residual_phases = residual_phases.get()
        dm_cmds = dm_cmds.get()
        detector_images = detector_images.get()

    def get_masked_array(vec, mask):
        aux = reshape_on_mask(vec, mask)
        ma_vec = masked_array(aux, mask)
        return ma_vec


    for i in range(Nits):
        masked_input_phases[i,:,:] = get_masked_array(input_phases[i,:], cmask)
        masked_dm_phases[i,:,:] = get_masked_array(dm_phases[i,:], cmask)
        masked_residual_phases[i,:,:] = get_masked_array(residual_phases[i,:], cmask)

        input_phase = reshape_on_mask(input_phases[i,:], psf_mask)
        electric_field = electric_field_amp * xp.exp(-1j*xp.asarray(input_phase*(2*xp.pi)/lambdaInM))
        field_on_focal_plane = xp.fft.fftshift(xp.fft.fft2(electric_field))
        psf = abs(field_on_focal_plane)**2
        logPSFs[i,:,:] = xp.log(psf)

    # Can this be done passing a dictionary of {filenames: MAs} to save?
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
    psf = xp.exp(logPSFs[-1])

    plt.figure(figsize=(9,9))
    plt.subplot(2,2,1)
    myimshow(masked_input_phases[-1], \
        title=f'Atmosphere phase [m]\nStrehl ratio = {xp.exp(-input_sig2[-1]):1.3f}',\
        cmap='RdBu',shrink=0.8)
    # w = ssao.pupilSizeInPixels + 10
    # H,W = ssao.cmask.shape
    # plt.xlim([W//2-w//2, W//2+w//2])
    # plt.ylim([H//2-w//2, H//2+w//2])
    plt.axis('off')

    pixelsPerMAS = ssao.lambdaInM/ssao.pupilSizeInM*180/xp.pi*3600*1000
    plt.subplot(2,2,2)
    showZoomCenter(psf, pixelsPerMAS, shrink=0.8, \
        title = f'Corrected PSF\nStrehl ratio = {xp.exp(-sig2[-1]):1.3f}',cmap='inferno') 

    plt.subplot(2,2,3)
    myimshow(detector_images[-1], title = 'Detector image', shrink=0.8)

    plt.subplot(2,2,4)
    ssao.dm.plot_position(dm_cmds[-1])
    plt.title('Mirror command [m]')
    plt.axis('off')

    if xp.__name__ == 'cupy': # Convert to numpy for plotting
        input_sig2 = input_sig2.get()
        sig2 = sig2.get()

    plt.figure()#figsize=(1.7*Nits/10,3))
    plt.plot(input_sig2,'-o',label='open loop')
    plt.plot(sig2,'-o',label='closed loop')
    plt.legend()
    plt.grid()
    plt.ylabel(r'$\sigma^2 [rad^2]$')
    plt.xlabel('# iteration')
    plt.gca().set_yscale('log')
    # x_ticks = (xp.linspace(0,Nits,21,endpoint=True)).get() if xp.__name__ == 'cupy' else xp.linspace(0,Nits,21,endpoint=True)
    # x_ticks = (xp.linspace(0,Nits,21,endpoint=True)).get() if xp.__name__ == 'cupy' else xp.linspace(0,Nits,21,endpoint=True)
    # plt.xticks(x_ticks)
    plt.xlim([-0.5,Nits-0.5])

    plt.show()

    return ssao

