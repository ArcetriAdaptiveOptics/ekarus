import matplotlib.pyplot as plt
import os.path as op

from ekarus.e2e.utils.image_utils import showZoomCenter, myimshow, reshape_on_mask
from ekarus.e2e.cascading_ao_class import CascadingAO

import ekarus.e2e.utils.my_fits_package as myfits   

import xupy as xp
np = xp.np


def main(tn:str='example_dual_stage', show:bool=False):

    print('Initializing devices ...')
    cascao = CascadingAO(tn)

    cascao.initialize_turbulence()
    cascao.sc1.calibrate_sensor(tn, prefix_str='Pyr1_',
                            piston=1-cascao.cmask, 
                            lambdaOverD = cascao.pyr1.lambdaInM/cascao.pupilSizeInM,
                            Npix = cascao.subap1Size) 

    cascao.sc2.calibrate_sensor(tn, prefix_str='Pyr2_',
                    piston=1-cascao.cmask, 
                    lambdaOverD = cascao.pyr2.lambdaInM/cascao.pupilSizeInM,
                    Npix = cascao.subap2Size) 


    KL, m2c = cascao.define_KL_modes(cascao.dm1, zern_modes=5, save_prefix='DM1_')
    cascao.pyr1.set_modulation_angle(cascao.sc1.modulationAngleInLambdaOverD)
    Rec, _ = cascao.compute_reconstructor(cascao.sc1, KL, cascao.pyr1.lambdaInM, amps=0.2, save_prefix='SC1_')
    cascao.sc1.load_reconstructor(Rec,m2c)

    KL, m2c = cascao.define_KL_modes(cascao.dm2, zern_modes=5, save_prefix='DM2_')
    cascao.pyr2.set_modulation_angle(cascao.sc2.modulationAngleInLambdaOverD)
    Rec, _ = cascao.compute_reconstructor(cascao.sc2, KL, cascao.pyr2.lambdaInM, amps=0.2, save_prefix='SC2_')
    cascao.sc2.load_reconstructor(Rec,m2c)

    print('Running the loop ...')
    lambdaInM = 1000e-9
    dm2_sig2, dm1_sig2, input_sig2 = cascao.run_loop(lambdaInM, cascao.starMagnitude, save_prefix='')

    # Post-processing and plotting
    print('Plotting results ...')
    if show:
        subap1_masks = xp.sum(cascao.sc1._subaperture_masks,axis=0)
        plt.figure()
        myimshow(subap1_masks,title='Subaperture masks CCD1')
        subap2_masks = xp.sum(cascao.sc2._subaperture_masks,axis=0)
        plt.figure()
        myimshow(subap2_masks,title='Subaperture masks CCD2')

        KL = myfits.read_fits(op.join(cascao.savecalibpath,'DM1_KLmodes.fits'))
        N=9
        plt.figure(figsize=(2*N,7))
        for i in range(N):
            plt.subplot(4,N,i+1)
            cascao.dm1.plot_surface(KL[i,:],title=f'KL Mode {i}')
            plt.subplot(4,N,i+1+N)
            cascao.dm1.plot_surface(KL[i+N,:],title=f'KL Mode {i+N}')
            plt.subplot(4,N,i+1+N*2)
            cascao.dm1.plot_surface(KL[-i-1-N,:],title=f'KL Mode {xp.shape(KL)[0]-i-1-N}')
            plt.subplot(4,N,i+1+N*3)
            cascao.dm1.plot_surface(KL[-i-1,:],title=f'KL Mode {xp.shape(KL)[0]-i-1}')

        KL = myfits.read_fits(op.join(cascao.savecalibpath,'DM2_KLmodes.fits'))
        N=9
        plt.figure(figsize=(2*N,7))
        for i in range(N):
            plt.subplot(4,N,i+1)
            cascao.dm2.plot_surface(KL[i,:],title=f'KL Mode {i}')
            plt.subplot(4,N,i+1+N)
            cascao.dm2.plot_surface(KL[i+N,:],title=f'KL Mode {i+N}')
            plt.subplot(4,N,i+1+N*2)
            cascao.dm2.plot_surface(KL[-i-1-N,:],title=f'KL Mode {xp.shape(KL)[0]-i-1-N}')
            plt.subplot(4,N,i+1+N*3)
            cascao.dm2.plot_surface(KL[-i-1,:],title=f'KL Mode {xp.shape(KL)[0]-i-1}')

        IM1 = myfits.read_fits(op.join(cascao.savecalibpath,'SC1_IM.fits'))
        IM1_std = xp.std(IM1,axis=0)
        IM2 = myfits.read_fits(op.join(cascao.savecalibpath,'SC2_IM.fits'))
        IM2_std = xp.std(IM2,axis=0)
        if xp.on_gpu:
            IM1_std = IM1_std.get()
            IM2_std = IM2_std.get()
        plt.figure()
        plt.plot(IM1_std,'-o')
        plt.plot(IM2_std,'-o')
        plt.grid()
        plt.title('Interaction matrix standard deviation')

        screen = cascao.get_phasescreen_at_time(0.5)
        if xp.on_gpu:
            screen = screen.get()
        plt.figure()
        plt.imshow(screen, cmap='RdBu')
        plt.colorbar()
        plt.title('Atmo screen')

    atmo_phases, _, res1_phases, det1_frames, rec1_modes, _, _, res2_phases, det2_frames, rec2_modes, _ = cascao.load_telemetry_data()

    oversampling = 8
    padding_len = int(cascao.cmask.shape[0]*(oversampling-1)/2)
    psf_mask = xp.pad(cascao.cmask, padding_len, mode='constant', constant_values=1)
    electric_field_amp = 1-psf_mask

    last_res1_phase = xp.array(res1_phases[-1,:,:])
    residual1_phase = last_res1_phase[~cascao.cmask]
    input_phase = reshape_on_mask(residual1_phase, psf_mask)
    electric_field = electric_field_amp * xp.exp(-1j*xp.asarray(input_phase*(2*xp.pi)/lambdaInM))
    field_on_focal_plane = xp.fft.fftshift(xp.fft.fft2(electric_field))
    psf1 = abs(field_on_focal_plane)**2

    last_res2_phase = xp.array(res2_phases[-1,:,:])
    residual2_phase = last_res2_phase[~cascao.cmask]
    input_phase = reshape_on_mask(residual2_phase, psf_mask)
    electric_field = electric_field_amp * xp.exp(-1j*xp.asarray(input_phase*(2*xp.pi)/lambdaInM))
    field_on_focal_plane = xp.fft.fftshift(xp.fft.fft2(electric_field))
    psf2 = abs(field_on_focal_plane)**2

    pixelsPerMAS = lambdaInM/cascao.pupilSizeInM/oversampling*180/xp.pi*3600*1000

    # cmask = cascao.cmask.get() if xp.__name__ == 'cupy' else cascao.cmask.copy()
    if xp.on_gpu: # Convert to numpy for plotting
        dm1_sig2 = dm1_sig2.get()
        dm2_sig2 = dm2_sig2.get()
        input_sig2 = input_sig2.get()
        rec1_modes = rec1_modes.get()
        rec2_modes = rec2_modes.get()

    shrink = 0.75
    plt.figure()#figsize=(9,13.5))
    plt.subplot(2,3,1)
    myimshow(det1_frames[-1], title = 'Detector 1 image', shrink=shrink)
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

    tvec = np.arange(cascao.Nits)*cascao.dt*1e+3
    plt.figure()#figsize=(1.7*Nits/10,3))
    plt.plot(tvec,input_sig2,'-o',label='open loop')
    plt.plot(tvec,dm1_sig2,'-o',label='after DM1')
    plt.plot(tvec,dm2_sig2,'-o',label='after DM2')
    plt.grid()
    plt.xlim([0.0,tvec[-1]*1e+3])
    plt.xlabel('Time [ms]')
    plt.ylabel(r'$\sigma^2 [rad^2]$')
    plt.gca().set_yscale('log')
    plt.xlim([-0.5,cascao.Nits-0.5])

    plt.figure()
    plt.plot(tvec,rec1_modes[:,:10],'-o')
    plt.grid()
    plt.xlim([0.0,tvec[-1]*1e+3])
    plt.xlabel('Time [ms]')
    plt.ylabel('amplitude [m]')
    plt.title('Reconstructor 1 modes\n(first 10)')
    plt.show()

    plt.figure()
    plt.plot(tvec,rec2_modes[:,:10],'-o')
    plt.grid()
    plt.xlim([0.0,tvec[-1]*1e+3])
    plt.xlabel('Time [ms]')
    plt.ylabel('amplitude [m]')
    plt.title('Reconstructor 2 modes\n(first 10)')
    plt.show()


    plt.show()

    return cascao

if __name__ == '__main__':
    cascao = main(show=True)