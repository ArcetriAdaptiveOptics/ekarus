import matplotlib.pyplot as plt
import os
from numpy.ma import masked_array

from ekarus.e2e.utils import my_fits_package as myfits

from ekarus.e2e.alpao_deformable_mirror import ALPAODM
from ekarus.e2e.pyramid_wfs import PyramidWFS
from ekarus.e2e.detector import Detector
from ekarus.e2e.slope_computer import SlopeComputer

from ekarus.e2e.scao_class import SCAO
from ekarus.e2e.utils.turbulence_generator import generate_phasescreens, move_mask_on_phasescreen

from ekarus.e2e.utils.image_utils import showZoomCenter

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


# System data
lambdaInM = 1000e-9
pupilSizeInM = 32e-3
oversampling = 4
Npix = 128

pix2rad = lambdaInM/pupilSizeInM/oversampling 
rad2arcsec = 180/xp.pi*3600

alpha = 3*lambdaInM/pupilSizeInM # modulation angle
apex_angle = 0.005859393655500168/(2*xp.pi)*30/32
detector_shape = (256,256)
subaperture_size = 63.5

# Telescope data
throughput = 0.06
telescopeSizeInM = 1.8
telescopeArea = telescopeSizeInM**2/4*xp.pi

# Atmospheric data
r0 = 5e-2
L0 = 8

show = False

# Sensor, detector, DM
print('Initializing devices ...')
wfs = PyramidWFS(apex_angle, xp=xp)
ccd = Detector(detector_shape=detector_shape, xp=xp)
dm = ALPAODM(468, Npix=Npix, xp=xp)
slope_computer = SlopeComputer(wfs_type = 'PyrWFS', xp=xp)

scao = SCAO(wfs, ccd, slope_computer, dm, Npix, pupilSizeInM, oversampling=oversampling, telescope_area=telescopeArea, throughput=throughput, xp=xp)
scao.set_wavelength(lambdaInM=lambdaInM)
scao.set_star_magnitude(6)

# Define save directory and data dictionary
basepath = os.getcwd()
dir_path = os.path.join(basepath,'ekarus/mains/250802_sim_data/')

hdr_dict = {'WAVELEN': lambdaInM*1e+9, 'PUP_SIZE': pupilSizeInM, 'OVERSAMP': oversampling, 'N_PIX': Npix, \
'MOD_ANG': alpha, 'APEX_ANG': apex_angle, 'CCD_SIZE': max(detector_shape), 'SUBAPPIX': subaperture_size, \
'FR_PAR': r0, 'ATMO_LEN': L0, 'TEL_THR': throughput}

try:
    os.mkdir(dir_path)
except FileExistsError:
    pass


# 1. Define the subapertures using a piston input wavefront
print('Defining the detector subaperture masks ...')
subap_path = os.path.join(dir_path,'SubapertureMasks.fits')
try:
    subaperture_masks = myfits.read_fits(subap_path, isBool=True)
    if xp.__name__ == 'cupy':
        subaperture_masks = xp.asarray(subaperture_masks)
    slope_computer._subaperture_masks = subaperture_masks
except FileNotFoundError:
    piston = 1-scao.cmask
    modulated_intensity = wfs.modulate(piston, 10*alpha, pix2rad)
    detector_image = ccd.image_on_detector(modulated_intensity)
    slope_computer.calibrate_sensor(subaperture_image = detector_image, Npix = subaperture_size)
    subaperture_masks = slope_computer._subaperture_masks
    if xp.__name__ == 'cupy':
        detector_image = detector_image.get()
    plt.figure()
    plt.imshow(detector_image,origin='lower')
    plt.colorbar()
    plt.title('Subaperture masks')
    plt.show()
    myfits.save_fits(subap_path, (subaperture_masks).astype(xp.uint8), hdr_dict)

# 2. Define the system modes
print('Obtaining the Karhunen-Loeve mirror modes ...')

KL_path = os.path.join(dir_path,'KLmatrix.fits')
m2c_path = os.path.join(dir_path,'m2c.fits')

try:
    KL = myfits.read_fits(KL_path)
    m2c = myfits.read_fits(m2c_path)
    if xp.__name__ == 'cupy':
        KL = xp.asarray(KL, dtype=xp.float32)
        m2c = xp.asarray(m2c, dtype=xp.float32)
except FileNotFoundError:
    KL, m2c = scao.define_KL_modal_base(r0, L0, zern2remove = 5)
    myfits.save_fits(KL_path, KL, hdr_dict)
    myfits.save_fits(m2c_path, m2c, hdr_dict)

if show:
    N=9
    plt.figure(figsize=(2*N,7))
    for i in range(N):
        plt.subplot(4,N,i+1)
        dm.plot_surface(KL[i,:],title=f'KL Mode {i}')
        plt.subplot(4,N,i+1+N)
        dm.plot_surface(KL[i+N,:],title=f'KL Mode {i+N}')
        plt.subplot(4,N,i+1+N*2)
        dm.plot_surface(KL[-i-1-N,:],title=f'KL Mode {xp.shape(KL)[0]-i-1-N}')
        plt.subplot(4,N,i+1+N*3)
        dm.plot_surface(KL[-i-1,:],title=f'KL Mode {xp.shape(KL)[0]-i-1}')

# 3. Calibrate the system
print('Calibrating the KL modes ...')

IM_path = os.path.join(dir_path,'IMmatrix.fits')
Rec_path = os.path.join(dir_path,'Rec.fits')

try:
    IM = myfits.read_fits(IM_path)
    Rec = myfits.read_fits(Rec_path)
    if xp.__name__ == 'cupy':
        IM = xp.asarray(IM, dtype=xp.float32)
        Rec = xp.asarray(Rec, dtype=xp.float32)

except FileNotFoundError:
    IM, Rec = scao.calibrate_modes(KL, amps = 0.2, modulation_angle=alpha)
    IM_std = xp.std(IM,axis=0)
    if xp.__name__ == 'cupy':
        IM_std = IM_std.get()
    plt.figure()
    plt.plot(IM_std,'-o')
    plt.grid()
    plt.title('Interaction matrix standard deviation')
    plt.show()
    myfits.save_fits(IM_path, IM, hdr_dict)
    myfits.save_fits(Rec_path, Rec, hdr_dict)


# 4. Get atmospheric phase screen
print('Generating phase screens ...')

Nscreens = 1
N = 10

screenPixels = Npix*oversampling*N
screenMeters = N*oversampling*telescopeSizeInM

pixelsPerMeter = screenPixels/screenMeters

atmo_path = os.path.join(dir_path, 'AtmoScreens.fits')
try:
    screens = myfits.read_fits(atmo_path)
    if xp.__name__ == 'cupy':
        screens = xp.asarray(screens, dtype=xp.float32)
except FileNotFoundError:
    screens = generate_phasescreens(lambdaInM, r0, L0, Nscreens, \
     screenSizeInPixels=screenPixels, screenSizeInMeters=screenMeters, \
     savepath=atmo_path, xp=xp, dtype=xptype)
    myfits.save_fits(atmo_path, screens, hdr_dict)


screen = screens[0]/4

dt = 1e-3
wind_speed = 20
wind_angle = 1/4 * xp.pi

if show:
    plt.figure()
    plt.imshow(screen)
    plt.colorbar()
    plt.title('Atmo screen')


# 5. Perform the iteration
print('Running the loop ...')
compression = telescopeSizeInM/pupilSizeInM
electric_field_amp = 1-scao.cmask

g = 2
Nits = 200

Nmodes = 300

mask_len = int(xp.sum(1-dm.mask))
dm_shape = xp.zeros(mask_len, dtype=xptype)
dm_cmd = xp.zeros(dm.Nacts, dtype=xptype)
dm_phase = xp.zeros_like(scao.cmask, dtype=xptype)

# Save telemetry
dm_cmds = xp.zeros([Nits,dm.Nacts])
dm_phases = xp.zeros([Nits,mask_len])
phases = xp.zeros([Nits,mask_len])
input_phases = xp.zeros([Nits,mask_len])
reconstructed_phases = xp.zeros([Nits,mask_len])
ccd_images = xp.zeros([Nits,ccd.detector_shape[0],ccd.detector_shape[1]])

scao.set_integration_time(timeStep=dt)

# Nzern = 10
# from ekarus.analytical.zernike_generator import ZernikeGenerator
# zg = ZernikeGenerator(scao.cmask.get(), Npix)
# zern_mat = xp.stack([xp.array(zg.getZernike(i+2),dtype=xp.float32) for i in range(Nzern)])
# zern_mix = xp.zeros([zern_mat.shape[1],zern_mat.shape[2]])
# zern_amps = xp.array([0.2,-0.4,0.3,0.12,-0.15,0.1,0.13,-0.11,0.1,0.05])*100
# for i in range(Nzern):
#     zern_mix += zern_mat[i,:,:] * zern_amps[i]
# zern_mix += xp.ones_like(zern_mix)*0

for i in range(Nits):
    print(f'\rIteration {i+1}/{Nits}', end='')
    tt = dt*i
    input_phase = move_mask_on_phasescreen(screen, scao.cmask, tt, wind_speed, wind_angle, pixelsPerMeter, xp=xp)
    # input_phase *= compression
    # input_phase = zern_mix

    dm_phase[~scao.cmask] = dm_shape
    dm_phase = xp.reshape(dm_phase, scao.cmask.shape)

    phase = input_phase - dm_phase

    cmd = scao.perform_loop_iteration(phase, Rec[:Nmodes,:], m2c=m2c[:,:Nmodes], modulation_angle=alpha)
    dm_cmd += cmd*g
    dm_shape = dm.IFF @ dm_cmd
    # dm_shape += dm.IFF @ cmd

    ccd_image = scao.ccd.last_frame

    # Save telemetry
    input_phases[i,:] = input_phase[~scao.cmask]
    dm_phases[i,:] = dm_phase[~scao.cmask]
    ccd_images[i,:,:] = ccd_image
    dm_cmds[i,:] = dm_cmd
    phases[i,:] = phase[~scao.cmask]
print('')

#################### Post-processing ##############################
electric_field_amp = 1-scao.cmask
electric_field = electric_field_amp * xp.exp(-1j*phase)
field_on_focal_plane = xp.fft.fftshift(xp.fft.fft2(electric_field))
psf = abs(field_on_focal_plane)**2

sig2 = xp.std(input_phases-dm_phases,axis=-1)**2 #xp.mean((input_phases-dm_phases)**2, axis=-1)
input_sig2 = xp.std(input_phases,axis=-1)**2 #xp.mean((input_phases)**2, axis=-1)

print('Saving telemetry to .fits ...')
cmask = scao.cmask.get() if xp.__name__ == 'cupy' else scao.cmask.copy()
# from ekarus.e2e.utils.image_utils import reshape_on_mask

masked_input_phases = xp.zeros([Nits,scao.cmask.shape[0],scao.cmask.shape[1]], dtype=xptype)
masked_dm_phases = xp.zeros([Nits,scao.cmask.shape[0],scao.cmask.shape[1]], dtype=xptype)
masked_phases = xp.zeros([Nits,scao.cmask.shape[0],scao.cmask.shape[1]], dtype=xptype)

aux = xp.zeros_like(scao.cmask,dtype=xptype)

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

atmo_phases_path = os.path.join(dir_path,'AtmoPhase.fits')
dm_phases_path = os.path.join(dir_path,'DMphase.fits')
err_phases_path = os.path.join(dir_path,'DeltaPhase.fits')

myfits.save_fits(atmo_phases_path, masked_input_phases, hdr_dict)
myfits.save_fits(dm_phases_path, masked_dm_phases, hdr_dict)
myfits.save_fits(err_phases_path, masked_phases-masked_dm_phases, hdr_dict)

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
plt.imshow(masked_array(input_phase, mask = cmask),origin='lower')
plt.colorbar()
plt.title(f'Input: Strehl ratio = {xp.exp(-input_sig2[-1]):1.3f}')

# plt.subplot(2,2,2)
# plt.imshow(masked_array(phase, mask = cmask),origin='lower')
# plt.colorbar()
# plt.title(f'Output: Strehl ratio = {xp.exp(-sig2[-1]):1.3f}')

plt.subplot(2,2,2)
showZoomCenter(psf, pix2rad*rad2arcsec, title = f'PSF: Strehl ratio = {xp.exp(-sig2[-1]**2):1.3f}')

plt.subplot(2,2,3)
plt.imshow(ccd_image,origin='lower')
plt.colorbar()
plt.title('Detector image')

# plt.subplot(2,2,4)
# plt.imshow(masked_array(dm_phase, mask = cmask),origin='lower')
# plt.colorbar()
# plt.title('DM phase')

plt.subplot(2,2,4)
dm.plot_position(dm_cmd*lambdaInM/(2*xp.pi))
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
