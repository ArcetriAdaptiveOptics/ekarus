import matplotlib.pyplot as plt
from numpy.ma import masked_array

from ekarus.e2e.utils.image_utils import showZoomCenter
from ekarus.e2e.single_stage_ao_class import SingleStageAO

try:
    import cupy as xp
    xptype = xp.float32
    print('Now using GPU acceleration!')
except:
    import numpy as xp
    xptype = xp.float64

# def myimshow(image, title = ''):
#     if hasattr(image,'get'):
#         image = image.get()
#     plt.imshow(image,origin='lower')
#     plt.colorbar()
#     plt.title(title)


tn = '20250829_080000'

print('Initializing devices ...')
ssao = SingleStageAO(tn)

show = False # boolean to show initialization outputs
lambdaInM = 1000e-9
starMagnitude = 3

alpha = 3*lambdaInM/ssao.pupilSizeInM # modulation angle

ssao.set_wavelength(lambdaInM=lambdaInM)
ssao.set_star_magnitude(starMagnitude)


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
Rec, IM = ssao.calibrate_modes(KL, amps = 0.2, modulation_angle = alpha)
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
print('Generating phase screens ...')
screens = ssao.generate_phase_screens(N=20)
screen = screens[0]
if show:
    plt.figure()
    plt.imshow(screen, cmap='RdBu')
    plt.colorbar()
    plt.title('Atmo screen')



# 5. Perform the iteration
print('Running the loop ...')
electric_field_amp = 1-ssao.cmask

dt = 1e-3
g = 10

Nits = 150#300 
Nmodes = 468#100
wind_speed = 10
wind_angle = xp.pi/4


mask_len = int(xp.sum(1-ssao.dm.mask))
dm_shape = xp.zeros(mask_len, dtype=xptype)
dm_cmd = xp.zeros(ssao.dm.Nacts, dtype=xptype)
dm_phase = xp.zeros_like(ssao.cmask, dtype=xptype)

# Save telemetry
dm_cmds = xp.zeros([Nits,ssao.dm.Nacts])
dm_phases = xp.zeros([Nits,mask_len])
phases = xp.zeros([Nits,mask_len])
input_phases = xp.zeros([Nits,mask_len])
reconstructed_phases = xp.zeros([Nits,mask_len])
ccd_images = xp.zeros([Nits,ssao.ccd.detector_shape[0],ssao.ccd.detector_shape[1]])

# Rec = m2c @ Rec
# from ekarus.e2e.utils.turbulence_generator import move_mask_on_phasescreen

init_scramble = xp.random.randn(ssao.dm.Nacts)*20*ssao.pupilSizeInM/ssao.lambdaInM*2*xp.pi
init_phase = xp.zeros_like(ssao.cmask, dtype=xptype)
init_phase[~ssao.cmask] = ssao.dm.IFF @ init_scramble
input_phase = xp.reshape(init_phase, ssao.cmask.shape)

for i in range(Nits):
    print(f'\rIteration {i+1}/{Nits}', end='')
    tt = dt*i
    #input_phase = move_mask_on_phasescreen(screen, ssao.cmask, tt, wind_speed, wind_angle, ssao.pixelsPerMeter, xp=xp)

    dm_phase[~ssao.cmask] = dm_shape
    dm_phase = xp.reshape(dm_phase, ssao.cmask.shape)

    phase = input_phase - dm_phase
    # phase -= xp.mean(phase)

    modes = ssao.perform_loop_iteration(dt, phase, Rec, modulation_angle=alpha)
    cmd = m2c[:,0:Nmodes] @ modes[0:Nmodes]
    dm_cmd += cmd*0.05
    dm_shape = ssao.dm.IFF @ dm_cmd

    # Save telemetry
    input_phases[i,:] = input_phase[~ssao.cmask]
    dm_phases[i,:] = dm_phase[~ssao.cmask]
    ccd_images[i,:,:] = ssao.ccd.last_frame
    dm_cmds[i,:] = dm_cmd
    phases[i,:] = phase[~ssao.cmask]
print('')

#################### Post-processing ##############################
electric_field_amp = 1-ssao.cmask
electric_field = electric_field_amp * xp.exp(-1j*phase)
field_on_focal_plane = xp.fft.fftshift(xp.fft.fft2(electric_field))
psf = abs(field_on_focal_plane)**2

sig2 = xp.std(input_phases-dm_phases,axis=-1)**2 #xp.mean((input_phases-dm_phases)**2, axis=-1)
input_sig2 = xp.std(input_phases,axis=-1)**2 #xp.mean((input_phases)**2, axis=-1)

print('Saving telemetry to .fits ...')
cmask = ssao.cmask.get() if xp.__name__ == 'cupy' else ssao.cmask.copy()
# from ekarus.e2e.utils.image_utils import reshape_on_mask

masked_input_phases = xp.zeros([Nits,ssao.cmask.shape[0],ssao.cmask.shape[1]], dtype=xptype)
masked_dm_phases = xp.zeros([Nits,ssao.cmask.shape[0],ssao.cmask.shape[1]], dtype=xptype)
masked_phases = xp.zeros([Nits,ssao.cmask.shape[0],ssao.cmask.shape[1]], dtype=xptype)

aux = xp.zeros_like(ssao.cmask,dtype=xptype)

if xp.__name__ == 'cupy':
    masked_input_phases = masked_input_phases.get()
    masked_dm_phases = masked_dm_phases.get()
    masked_phases = masked_phases.get()
    input_phases = input_phases.get()
    dm_phases = dm_phases.get()
    phases = phases.get()
    reconstructed_phases = reconstructed_phases.get()
    aux = aux.get()

for i in range(Nits):
    aux[~cmask] = input_phases[i,:]
    aux = (aux).reshape(cmask.shape)
    masked_input_phases[i,:,:] = masked_array(aux, cmask)

    aux[~cmask] = dm_phases[i,:]
    aux = (aux).reshape(cmask.shape)
    masked_dm_phases[i,:,:]= masked_array(aux, cmask)

    aux[~cmask] = phases[i,:]
    aux = (aux).reshape(cmask.shape)
    masked_phases[i,:,:]= masked_array(aux, cmask)

import os
atmo_phases_path = os.path.join(ssao.savepath,'AtmoPhase.fits')
dm_phases_path = os.path.join(ssao.savepath,'DMphase.fits')
err_phases_path = os.path.join(ssao.savepath,'DeltaPhase.fits')

from ekarus.e2e.utils import my_fits_package as myfits
myfits.save_fits(atmo_phases_path, masked_input_phases)
myfits.save_fits(dm_phases_path, masked_dm_phases)
myfits.save_fits(err_phases_path, masked_phases-masked_dm_phases)

ccd_image = ssao.ccd.last_frame
print(f'Number of collected photons: {xp.sum(ccd_image):1.0f}')

########################## Plotting ###############################
if xp.__name__ == 'cupy': # Convert to numpy for plotting
    input_phase = input_phase.get()
    phase = phase.get()
    ccd_image = ccd_image.get()
    dm_cmd = dm_cmd.get()
    dm_phase = dm_phase.get()
    psf = psf.get()
    input_sig2 = input_sig2.get()
    sig2 = sig2.get()

    
plt.figure(figsize=(12,12))
plt.subplot(2,2,1)
plt.imshow(masked_array(input_phase, mask = cmask),origin='lower',cmap='RdBu')
plt.colorbar()
plt.title(f'Input: Strehl ratio = {xp.exp(-input_sig2[-1]):1.3f}')
w = ssao.pupilSizeInPixels + 10
H,W = ssao.cmask.shape
plt.xlim([W//2-w//2, W//2+w//2])
plt.ylim([H//2-w//2, H//2+w//2])

# plt.subplot(2,2,2)
# plt.imshow(masked_array(phase, mask = cmask),origin='lower')
# plt.colorbar()
# plt.title(f'Output: Strehl ratio = {xp.exp(-sig2[-1]):1.3f}')


plt.subplot(2,2,2)
showZoomCenter(psf, ssao.pixelScale*180/xp.pi*3600, title = f'PSF: Strehl ratio = {xp.exp(-sig2[-1]**2):1.3f}',cmap='inferno')

plt.subplot(2,2,3)
plt.imshow(ccd_image,origin='lower')
plt.colorbar()
plt.title('Detector image')

# plt.subplot(2,2,4)
# plt.imshow(masked_array(dm_phase, mask = cmask),origin='lower')
# plt.colorbar()
# plt.title('DM phase')

plt.subplot(2,2,4)
ssao.dm.plot_position(dm_cmd*lambdaInM/(2*xp.pi))
plt.title('Mirror command')

plt.figure(figsize=(4*Nits/10,3))
plt.plot(input_sig2,'-o',label='open loop')
plt.plot(sig2,'-o',label='closed loop')
plt.legend()
plt.grid()
plt.ylabel(r'$\sigma^2 [rad^2]$')
plt.xlabel('# iteration')
x_ticks = (xp.arange(Nits//4)*4).get() if xp.__name__ == 'cupy' else xp.arange(Nits//4)*4
plt.xticks(x_ticks)
plt.xlim([-0.5,Nits-0.5])

plt.show()
