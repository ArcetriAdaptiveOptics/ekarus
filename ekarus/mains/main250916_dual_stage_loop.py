import matplotlib.pyplot as plt
from numpy.ma import masked_array

from ekarus.e2e.utils.image_utils import showZoomCenter, myimshow, reshape_on_mask
from ekarus.e2e.cascading_ao_class import CascadingAO

try:
    import cupy as xp
    xptype = xp.float32
    print('Now using GPU acceleration!')
except:
    import numpy as xp
    xptype = xp.float64


def main(tn:str='example_dual_stage', show:bool=False):

    print('Initializing devices ...')
    cascao = CascadingAO(tn, xp=xp)

    lambdaInM, starMagnitude = cascao._config.read_target_pars()

    # 1. Define the subapertures using a piston input wavefront
    print('Defining the detector subaperture masks for CCD1 ...')
    cascao.define_subaperture_masks(cascao.slope_computer1, lambdaInM, save_prefix='CCD1_')
    print('Defining the detector subaperture masks for CCD2 ...')
    cascao.define_subaperture_masks(cascao.slope_computer2, lambdaInM, save_prefix='CCD2_')
    if show:
        try:
            detector1_image = cascao.ccd1.last_frame
            plt.figure()
            myimshow(detector1_image,title='Subaperture masks')
            plt.show()

            detector2_image = cascao.ccd2.last_frame
            plt.figure()
            myimshow(detector2_image,title='Subaperture masks')
            plt.show()
        except:
            pass


    # 2. Define the system modes
    print('Obtaining the Karhunen-Loeve mirror modes for DM1 ...')
    KL1, m2c_dm1 = cascao.define_KL_modes(cascao.dm1, cascao.pyr1.oversampling, zern_modes=5, save_prefix='DM1_')
    print('Obtaining the Karhunen-Loeve mirror modes for DM2 ...')
    KL2, m2c_dm2 = cascao.define_KL_modes(cascao.dm2, cascao.pyr2.oversampling, zern_modes=5, save_prefix='DM2_')
    if show:
        N=9
        plt.figure(figsize=(2*N,7))
        for i in range(N):
            plt.subplot(4,N,i+1)
            cascao.dm1.plot_surface(KL1[i,:],title=f'KL Mode {i}')
            plt.subplot(4,N,i+1+N)
            cascao.dm1.plot_surface(KL1[i+N,:],title=f'KL Mode {i+N}')
            plt.subplot(4,N,i+1+N*2)
            cascao.dm1.plot_surface(KL1[-i-1-N,:],title=f'KL Mode {xp.shape(KL1)[0]-i-1-N}')
            plt.subplot(4,N,i+1+N*3)
            cascao.dm1.plot_surface(KL1[-i-1,:],title=f'KL Mode {xp.shape(KL1)[0]-i-1}')

        plt.figure(figsize=(2*N,7))
        for i in range(N):
            plt.subplot(4,N,i+1)
            cascao.dm2.plot_surface(KL2[i,:],title=f'KL Mode {i}')
            plt.subplot(4,N,i+1+N)
            cascao.dm2.plot_surface(KL2[i+N,:],title=f'KL Mode {i+N}')
            plt.subplot(4,N,i+1+N*2)
            cascao.dm2.plot_surface(KL2[-i-1-N,:],title=f'KL Mode {xp.shape(KL2)[0]-i-1-N}')
            plt.subplot(4,N,i+1+N*3)
            cascao.dm2.plot_surface(KL2[-i-1,:],title=f'KL Mode {xp.shape(KL2)[0]-i-1}')

    # 3. Calibrate the system
    print('Calibrating the KL modes for DM1 ...')
    Rec1, IM1 = cascao.calibrate_modes(cascao.slope_computer1, KL1, lambdaInM, amps=0.2, save_prefix='DM1_')
    print('Calibrating the KL modes for DM2 ...')
    Rec2, IM2 = cascao.calibrate_modes(cascao.slope_computer2, KL2, lambdaInM, amps=0.2, save_prefix='DM2_')
    if show:
        IM1_std = xp.std(IM1,axis=0)
        IM2_std = xp.std(IM2,axis=0)
        if xp.__name__ == 'cupy':
            IM1_std = IM1_std.get()
            IM2_std = IM2_std.get()
        plt.figure()
        plt.plot(IM1_std,'-o')
        plt.plot(IM2_std,'-o')
        plt.grid()
        plt.title('Interaction matrix standard deviation')
        plt.show()

    # 4. Get atmospheric phase screen
    print('Initializing turbulence ...')
    cascao.initialize_turbulence()
    if show:
        screen = cascao.get_phasescreen_at_time(0)
        if xp.__name__ == 'cupy':
            screen = screen.get()
        plt.figure()
        plt.imshow(screen, cmap='RdBu')
        plt.colorbar()
        plt.title('Atmo screen')

    # 5. Perform the iteration
    print('Running the loop ...')
    dm2_sig2, dm1_sig2, input_sig2 = cascao.run_loop(lambdaInM, starMagnitude, Rec1, Rec2, m2c_dm1, m2c_dm2, save_telemetry=True)

    # 6. Post-processing and plotting
    print('Plotting results ...')
    atmo_phases, _, res1_phases, det1_frames, _, _, res2_phases, det2_frames, _ = cascao.load_telemetry_data()

    oversampling = 4
    padding_len = int(cascao.cmask.shape[0]*(oversampling-1)/2)
    psf_mask = xp.pad(cascao.cmask, padding_len, mode='constant', constant_values=1)
    electric_field_amp = 1-psf_mask

    last_res1_phase = xp.array(res1_phases[-1,:,:])
    residual1_phase = last_res1_phase[~cascao.cmask]
    input_phase = reshape_on_mask(residual1_phase, psf_mask, xp=xp)
    electric_field = electric_field_amp * xp.exp(-1j*xp.asarray(input_phase*(2*xp.pi)/lambdaInM))
    field_on_focal_plane = xp.fft.fftshift(xp.fft.fft2(electric_field))
    psf1 = abs(field_on_focal_plane)**2

    last_res2_phase = xp.array(res2_phases[-1,:,:])
    residual2_phase = last_res2_phase[~cascao.cmask]
    input_phase = reshape_on_mask(residual2_phase, psf_mask, xp=xp)
    electric_field = electric_field_amp * xp.exp(-1j*xp.asarray(input_phase*(2*xp.pi)/lambdaInM))
    field_on_focal_plane = xp.fft.fftshift(xp.fft.fft2(electric_field))
    psf2 = abs(field_on_focal_plane)**2

    pixelsPerMAS = lambdaInM/cascao.pupilSizeInM/oversampling*180/xp.pi*3600*1000

    # cmask = cascao.cmask.get() if xp.__name__ == 'cupy' else cascao.cmask.copy()
    if xp.__name__ == 'cupy': # Convert to numpy for plotting
        dm1_sig2 = dm1_sig2.get()
        dm2_sig2 = dm2_sig2.get()
        input_sig2 = input_sig2.get()

    shrink = 0.45
    plt.figure(figsize=(9,13.5))
    plt.subplot(2,3,1)
    myimshow(det1_frames[-1], title = 'Detector 1 image', shrink=shrink)
    # myimshow(masked_array(atmo_phases[-1],cmask), \
    #     title=f'Atmosphere phase [m]\nStrehl ratio = {xp.exp(-input_sig2[-1]):1.3f}',\
    #     cmap='RdBu',shrink=0.8)
    # plt.axis('off')
    plt.subplot(2,3,2)    
    cascao.dm1.plot_position(shrink=shrink)
    plt.title('DM1 command [m]')
    plt.axis('off')

    plt.subplot(2,3,3)
    showZoomCenter(psf1, pixelsPerMAS, shrink=shrink, \
        title = f'PSF after DM1\nStrehl ratio = {xp.exp(-dm1_sig2[-1]):1.3f}',cmap='inferno') 

    plt.subplot(2,3,4)
    myimshow(det2_frames[-1], title = 'Detector 2 image', shrink=shrink)

    plt.subplot(2,3,5)    
    cascao.dm2.plot_position(shrink=shrink)
    plt.title('DM2 command [m]')
    plt.axis('off')

    plt.subplot(2,3,6)
    showZoomCenter(psf2, pixelsPerMAS, shrink=shrink, \
        title = f'PSF after DM2\nStrehl ratio = {xp.exp(-dm2_sig2[-1]):1.3f}',cmap='inferno') 


    plt.figure()#figsize=(1.7*Nits/10,3))
    plt.plot(input_sig2,'-o',label='open loop')
    plt.plot(dm1_sig2,'-o',label='after DM1')
    plt.plot(dm2_sig2,'-o',label='after DM2')
    plt.legend()
    plt.grid()
    plt.ylabel(r'$\sigma^2 [rad^2]$')
    plt.xlabel('# iteration')
    plt.gca().set_yscale('log')
    plt.xlim([-0.5,cascao.Nits-0.5])

    plt.show()

    return cascao

