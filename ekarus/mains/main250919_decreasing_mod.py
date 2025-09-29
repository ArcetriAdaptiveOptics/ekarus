import xupy as xp
# import numpy as np
masked_array = xp.masked_array

import matplotlib.pyplot as plt

from ekarus.e2e.utils.image_utils import showZoomCenter, myimshow #, reshape_on_mask
from ekarus.e2e.single_stage_ao_class import SingleStageAO

# import os.path as op
# import ekarus.e2e.utils.my_fits_package as myfits


def main(tn:str='example_decreasing_mod'):

    ssao = SingleStageAO(tn)

    ssao.initialize_turbulence()

    KL, m2c = ssao.define_KL_modes(ssao.dm, zern_modes=5)

    ssao.pyr.set_modulation_angle(ssao.sc.modulationAngleInLambdaOverD)
    Rec, _ = ssao.compute_reconstructor(ssao.sc, KL, ssao.pyr.lambdaInM, amps=0.2)
    ssao.pyr.set_modulation_angle(0.0)
    mod0_Rec, _ = ssao.compute_reconstructor(ssao.sc, KL, ssao.pyr.lambdaInM, amps=0.002, save_prefix='mod0_')

    print('Running the loop ...')
    try:
        if ssao.recompute is True:
            raise FileNotFoundError('Recompute is True')
        masked_input_phases, _, masked_residual_phases, _, _, _ = ssao.load_telemetry_data()
        m2rad = 2 * xp.pi / ssao.pyr.lambdaInM
        residual_phases = xp.zeros([ssao.Nits,int(xp.sum(1-ssao.cmask))])
        input_phases = xp.zeros([ssao.Nits,int(xp.sum(1-ssao.cmask))])
        for i in range(ssao.Nits):
            ma_res_phase = masked_residual_phases[i,:,:]
            residual_phases[i,:] = ma_res_phase[~ssao.cmask]
            ma_in_phase = masked_input_phases[i,:,:]
            input_phases[i,:] = ma_in_phase[~ssao.cmask]
        sig2 = xp.std(residual_phases * m2rad, axis=-1) ** 2
        input_sig2 = xp.std(input_phases * m2rad, axis=-1) ** 2
    except:
        ssao.sc.load_reconstructor(mod0_Rec,m2c)
        ssao.pyr.set_modulation_angle(0.0)
        mod0_sig2, _ = ssao.run_loop(ssao.pyr.lambdaInM, ssao.starMagnitude, save_prefix='mod0_')

        ssao.sc.load_reconstructor(Rec,m2c)
        ssao.pyr.set_modulation_angle(ssao.sc.modulationAngleInLambdaOverD)
        sig2, input_sig2 = ssao.run_loop(ssao.pyr.lambdaInM, ssao.starMagnitude, save_prefix='')

    masked_input_phases, _, masked_residual_phases, detector_frames, _, dm_commands = ssao.load_telemetry_data()
    _, _, mod0_ma_residual_phases, mod0_det_frames, _, mod0_dm_commands = ssao.load_telemetry_data(save_prefix='mod0_')

    lambdaRef = ssao.pyr.lambdaInM
    pixelsPerMAS = ssao.pyr.lambdaInM/ssao.pupilSizeInM/ssao.pyr.oversampling*180/xp.pi*3600*1000

    psf = ssao.get_psf_from_frame(xp.array(masked_residual_phases[-1,:,:]), lambdaRef, oversampling=8)
    mod0_psf = ssao.get_psf_from_frame(xp.array(mod0_ma_residual_phases[-1,:,:]), lambdaRef, oversampling=8)

    cmask = ssao.cmask.get() if xp.on_gpu else ssao.cmask.copy()
    if xp.on_gpu: # Convert to numpy for plotting
        input_sig2 = input_sig2.get()
        sig2 = sig2.get()
        mod0_sig2 = mod0_sig2.get()


    plt.figure(figsize=(9,9))
    plt.subplot(2,2,1)
    myimshow(masked_array(masked_input_phases[-1],cmask), \
        title=f'Atmosphere phase [m]\nSR = {xp.exp(-input_sig2[-1]):1.3f} @ {lambdaRef*1e+9:1.0f} [nm]',\
        cmap='RdBu',shrink=0.8)
    plt.axis('off')
    plt.subplot(2,2,2)
    showZoomCenter(psf, pixelsPerMAS, shrink=0.8, \
        title = f'Corrected PSF\nSR = {xp.exp(-sig2[-1]):1.3f} @ {lambdaRef*1e+9:1.0f} [nm]',cmap='inferno') 
    plt.subplot(2,2,3)
    myimshow(detector_frames[-1], title = 'Detector frame', shrink=0.8)
    plt.subplot(2,2,4)
    ssao.dm.plot_position(dm_commands[-1])
    plt.title('Mirror command [m]')
    plt.axis('off')

    plt.figure(figsize=(9,9))
    plt.subplot(2,2,1)
    myimshow(masked_array(masked_input_phases[-1],cmask), \
        title=f'Atmosphere phase [m]\nSR = {xp.exp(-input_sig2[-1]):1.3f} @ {lambdaRef*1e+9:1.0f} [nm]',\
        cmap='RdBu',shrink=0.8)
    plt.axis('off')
    plt.subplot(2,2,2)
    showZoomCenter(mod0_psf, pixelsPerMAS, shrink=0.8, \
        title = f'Corrected PSF\nSR = {xp.exp(-sig2[-1]):1.3f} @ {lambdaRef*1e+9:1.0f} [nm]',cmap='inferno') 
    plt.subplot(2,2,3)
    myimshow(mod0_det_frames[-1], title = 'Detector frame', shrink=0.8)
    plt.subplot(2,2,4)
    ssao.dm.plot_position(mod0_dm_commands[-1])
    plt.title('Mirror command [m]')
    plt.axis('off')

    tvec = xp.arange(ssao.Nits)*ssao.dt*1e+3
    tvec = tvec.get() if xp.on_gpu else tvec.copy()
    plt.figure()#figsize=(1.7*Nits/10,3))
    plt.plot(tvec,input_sig2,'-o',label='open loop')
    plt.plot(tvec,sig2,'-o',label=f'closed loop, modulation: {ssao.sc.modulationAngleInLambdaOverD} $lambda/D$')
    plt.plot(tvec,mod0_sig2,'-o',label='closed loop, no modulation')
    plt.legend()
    plt.grid()
    plt.xlim([0.0,tvec[-1]])
    plt.xlabel('Time [ms]')
    plt.ylabel(r'$\sigma^2 [rad^2]$')
    plt.gca().set_yscale('log')

    _,S,_ = xp.linalg.svd(Rec,full_matrices=False)
    _,mod0_S,_ = xp.linalg.svd(mod0_Rec,full_matrices=False)
    print(xp.sum(Rec-mod0_Rec)**2)
    if xp.on_gpu:
        S = S.get()
        mod0_S = mod0_S.get()
    plt.figure()
    plt.plot(S,'-o',label=f'modulation: {ssao.sc.modulationAngleInLambdaOverD} $lambda/D$')
    plt.plot(mod0_S,'-x',label=f'modulation: 0.0 $lambda/D$')
    plt.legend()
    plt.xlabel('Mode number')
    plt.grid()
    plt.title('Reconstructor singular values')
    
    plt.show()

    return ssao

if __name__ == '__main__':
    ssao = main()