import numpy as np
import matplotlib.pyplot as plt
import os

from ekarus.e2e.utils import my_fits_package as myfits

from ekarus.e2e.alpao_deformable_mirror import ALPAODM
from ekarus.e2e.pyramid_wfs import PyramidWFS
from ekarus.e2e.detector import Detector
from ekarus.e2e.slope_computer import SlopeComputer

from ekarus.e2e.scao_class import SCAO
from ekarus.e2e.utils.turbulence_generator import generate_phasescreens, move_mask_on_phasescreen

from ekarus.e2e.utils.image_utils import showZoomCenter #, reshape_on_mask#, get_circular_mask


# TODO: add cupy acceleration

# System data
lambdaInM = 1000e-9
pupilSizeInM = 32e-3
oversampling = 4
Npix = 128

pix2rad = lambdaInM/pupilSizeInM/oversampling 
rad2arcsec = 180/np.pi*3600

alpha = 3*lambdaInM/pupilSizeInM # modulation angle
apex_angle = 0.005859393655500168/(2*np.pi)*30/32
detector_shape = (256,256)
subaperture_size = 63.5

# Atmospheric data
r0 = 5e-2
L0 = 8

show = False

# Sensor, detector, DM
print('Initializing sensor, detector and deformable mirror ...')
wfs = PyramidWFS(apex_angle)
ccd = Detector(detector_shape=detector_shape)
dm = ALPAODM(468, Npix=Npix)
slope_computer = SlopeComputer(wfs_type = 'PyrWFS')

scao = SCAO(wfs, ccd, slope_computer, dm, Npix, pupilSizeInM, oversampling=oversampling)

# Define save directory and data dictionary
basepath = os.getcwd()
dir_path = os.path.join(basepath,'ekarus/mains/250802_sim_data/')

hdr_dict = {'WAVELEN': lambdaInM*1e+9, 'PUP_SIZE': pupilSizeInM, 'OVERSAMP': oversampling, 'N_PIX': Npix, \
'MOD_ANG': alpha, 'APEX_ANG': apex_angle, 'CCD_SIZE': np.max(detector_shape), 'SUBAPPIX': subaperture_size, \
'FR_PAR': r0, 'ATMO_LEN': L0}

try:
    os.mkdir(dir_path)
except FileExistsError:
    pass


# 1. Define the subapertures using a piston input wavefront
print('Defining the detector subaperture masks ...')

subap_path = os.path.join(dir_path,'SubapertureMasks.fits')
try:
    slope_computer._subaperture_masks = myfits.read_fits(subap_path, isBool=True)
except FileNotFoundError:
    piston = 1-scao.cmask
    modulated_intensity = wfs.modulate(piston, 10*alpha, pix2rad)
    detector_image = ccd.image_on_detector(modulated_intensity)
    plt.figure()
    plt.imshow(detector_image,origin='lower')
    plt.colorbar()
    plt.show()
    slope_computer.calibrate_sensor(subaperture_image = detector_image, Npix = subaperture_size)
    myfits.save_fits(subap_path, (slope_computer._subaperture_masks).astype(np.uint8), hdr_dict)

# 2. Define the system modes
print('Obtaining the Karhunen-Loeve mirror modes ...')

KL_path = os.path.join(dir_path,'KLmatrix.fits')
m2c_path = os.path.join(dir_path,'m2c.fits')

try:
    KL = myfits.read_fits(KL_path)
    m2c = myfits.read_fits(m2c_path)
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
        dm.plot_surface(KL[-i-1-N,:],title=f'KL Mode {np.shape(KL)[0]-i-1-N}')
        plt.subplot(4,N,i+1+N*3)
        dm.plot_surface(KL[-i-1,:],title=f'KL Mode {np.shape(KL)[0]-i-1}')

# 3. Calibrate the system
print('Calibrating the KL modes ...')

IM_path = os.path.join(dir_path,'IMmatrix.fits')
Rec_path = os.path.join(dir_path,'Rec.fits')

try:
    IM = myfits.read_fits(IM_path)
    Rec = myfits.read_fits(Rec_path)

except FileNotFoundError:
    IM, Rec = scao.calibrate_modes(KL, lambdaInM, alpha, amps = 0.1)

    myfits.save_fits(IM_path, IM, hdr_dict)
    myfits.save_fits(Rec_path, Rec, hdr_dict)


# 4. Get atmospheric phase screen
print('Generating phase screens ...')

Nscreens = 1
N = 10
telescopeSizeInM = 1.8

screenPixels = Npix*oversampling*N
screenMeters = N*oversampling*telescopeSizeInM

pixelsPerMeter = screenPixels/screenMeters

atmo_path = os.path.join(dir_path, 'AtmoScreens.fits')
try:
    screens = myfits.read_fits(atmo_path)
except FileNotFoundError:
    screens = generate_phasescreens(lambdaInM, r0, L0, Nscreens, \
     screenSizeInPixels=screenPixels, screenSizeInMeters=screenMeters, savepath=atmo_path)
    myfits.save_fits(atmo_path, screens, hdr_dict)


screen = screens[0]

dt = 1e-3
wind_speed = 20
wind_angle = np.pi/4

if show:
    plt.figure()
    plt.imshow(screen)
    plt.colorbar()
    plt.title('Atmo screen')


# 5. Perform the iteration
print('Running the loop ...')
g = 1e-3
mask_shape = (Npix*oversampling,Npix*oversampling)
dm_shape = np.zeros(np.sum(1-dm.mask))
dm_cmd = np.zeros(dm.Nacts)

compression = telescopeSizeInM/pupilSizeInM

Nits = 100
electric_field_amp = 1-scao.cmask

input_res = np.zeros(Nits)
res = np.zeros(Nits)

alpha *= 4 # do I decrease it as I converge?

for i in range(Nits):
    print(f'\rIteration {i+1}/{Nits}', end='')
    tt = dt*i
    input_phase = move_mask_on_phasescreen(screen, scao.cmask, tt, wind_speed, wind_angle, pixelsPerMeter)
    input_phase *= 1/compression

    dm_phase = np.zeros_like(dm.mask, dtype=np.float32)
    dm_phase[~dm.mask] = dm_shape #/lambdaInM*(2*np.pi) #/pupilSizeInM*(2*np.pi)
    dm_phase = np.pad(np.reshape(dm_phase,dm.mask.shape), (Npix*(oversampling-1))//2)

    phase = input_phase - dm_phase
    input_res[i] = np.std(input_phase[~scao.cmask])
    res[i] = np.std(phase[~scao.cmask])

    cmd, ccd_image = scao.perform_loop_iteration(phase, Rec, lambdaInM, alpha, m2c=m2c)
    dm_cmd += cmd*g
    dm_shape += dm.IFF @ dm_cmd
    
    electric_field = electric_field_amp * np.exp(1j*phase)
    field_on_focal_plane = np.fft.fftshift(np.fft.fft2(electric_field))
    psf = np.abs(field_on_focal_plane)**2

    
plt.figure(figsize=(12,12))
plt.subplot(2,2,1)
plt.imshow(np.ma.masked_array(input_phase, mask = scao.cmask),origin='lower')
plt.colorbar()
plt.title(f'Input: $\sigma^2$ = {input_res[-1]**2:1.3f} [$rad^2$]')

plt.subplot(2,2,2)
showZoomCenter(psf, pix2rad*rad2arcsec, title = f'PSF: $\sigma^2$ = {res[-1]**2:1.3f} [$rad^2$]')

plt.subplot(2,2,3)
plt.imshow(ccd_image,origin='lower')
plt.colorbar()
plt.title('Detector image')

# plt.subplot(2,2,4)
# plt.imshow(np.ma.masked_array(phase, mask = scao.cmask),origin='lower')
# plt.colorbar()
# plt.title('Corrected phase')

plt.subplot(2,2,4)
dm.plot_position(dm_cmd)
plt.title('Mirror command')



plt.figure(figsize=(4*Nits/10,3))
plt.plot(input_res,'-o',label='open loop')
plt.plot(res,'-o',label='closed loop')
plt.legend()
plt.grid()
plt.ylabel('$\sigma^2$ [$rad^2$]')
plt.xlabel('# iteration')
plt.xticks(np.arange(Nits))
plt.xlim([-0.5,Nits-0.5])


plt.show()


# input_phases = []
# psfs = []
# ccd_images = []
# dm_cmds = []

# Nit = 10

# for i in range(Nit):
#     tt = dt*i
#     input_phase = move_mask_on_phasescreen(screen, scao.cmask.mask(), tt, wind_speed, wind_angle, pixelsPerMeter)

#     dm_phase = np.zeros_like(dm.mask,dtype=np.float32)
#     dm_phase[~dm.mask] = dm_shape/pupilSizeInM*(2*np.pi)
#     dm_phase = np.pad(dm_phase, (Npix*(oversampling-1))//2)

#     phase = input_phase - dm_phase

#     cmd, ccd_image = scao.perform_loop_iteration(phase, Rec, lambdaInM, m2c)
#     dm_cmd += cmd*g
#     dm_shape += dm.IFF @ dm_cmd
    
#     electric_field = scao.cmask.mask() * np.exp(1j*phase)
#     field_on_focal_plane = np.fft.fftshift(np.fft.fft2(electric_field))
#     psf = np.abs(field_on_focal_plane**2)
    
#     input_phases.append(np.ma.masked_array(input_phase, mask = scao.cmask.mask()))
#     psfs.append(psf)
#     ccd_images.append(ccd_image)
#     dm_cmds.append(dm_cmd)

# fig = plt.figure(figsize=(12,12))
# plt.subplot(2,2,1)
# plt.imshow(input_phases[0],origin='lower')
# plt.colorbar()
# plt.title('Input phase')

# plt.subplot(2,2,2)
# showZoomCenter(psfs[0],pix2rad*rad2arcsec, title='PSF')

# plt.subplot(2,2,3)
# plt.imshow(ccd_images[0],origin='lower')
# plt.colorbar()
# plt.title('Detector image')

# plt.subplot(2,2,4)
# plt.scatter(dm.act_coords[0],dm.act_coords[1],c=dm_cmds[0])
# plt.axis('equal')
# plt.colorbar()
# plt.title('Mirror command')

# from matplotlib.animation import FuncAnimation

# def animate(frame):
#     ff = frame+1

#     plt.subplot(2,2,1)
#     plt.imshow(input_phases[ff],origin='lower')

#     plt.subplot(2,2,2)
#     showZoomCenter(psfs[ff],pix2rad*rad2arcsec, title='PSF')

#     plt.subplot(2,2,3)
#     plt.imshow(ccd_images[ff],origin='lower')

#     plt.subplot(2,2,4)
#     plt.scatter(dm.act_coords[0],dm.act_coords[1],c=dm_cmds[ff])

# ani = FuncAnimation(fig, animate, interval = 30, frames = Nit-1)
# plt.show()





