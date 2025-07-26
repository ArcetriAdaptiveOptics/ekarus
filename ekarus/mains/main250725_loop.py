import numpy as np
import matplotlib.pyplot as plt
import os

from ekarus.e2e.utils import my_fits_package as myfits

from ekarus.e2e.alpao_deformable_mirror import ALPAODM
from ekarus.e2e.pyramid_wfs import PyramidWFS
from ekarus.e2e.detector import Detector

from ekarus.userscripts import *


# TODO: add detector noise
# TODO: add cupy acceleration

# System data
lambdaInM = 1000e-9
pupilSizeInM = 32e-3
oversampling = 4
Npix = 128

pix2rad = lambdaInM/pupilSizeInM/oversampling 

alpha = 3*lambdaInM/pupilSizeInM # modulation angle
apex_angle = 0.005859393655500168/(2*np.pi)*30/32
detector_shape = (256,256)
subaperture_size = 63.5

# Atmospheric data
r0 = 5e-2
L0 = 8

# Sensor, detector, DM
print('Initializing sensor, detector and deformable mirror ...')
wfs = PyramidWFS(apex_angle)
ccd = Detector(detector_shape=detector_shape)
dm = ALPAODM(468)

# Define save directory and data dictionary
basepath = os.getcwd()
dir_path = os.path.join(basepath,'ekarus/mains/250725_sim_data/')

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
    ccd.subapertures = myfits.read_fits(subap_path, isBool=True)
except FileNotFoundError:
    ccd.subapertures, modulated_intensity = define_subaperture_masks(wfs, ccd, Npix, oversampling, \
                                             lambdaInM, pupilSizeInM, subapertureSizeInPixels=subaperture_size)
    myfits.save_fits(subap_path, (ccd.subapertures).astype(np.uint8), hdr_dict)

    ccd_intensity = ccd.resize_on_detector(modulated_intensity)
    plt.figure()
    plt.imshow(ccd_intensity-(ccd.subapertures[0]+ccd.subapertures[1]+\
                              ccd.subapertures[2]+ccd.subapertures[3]), origin='lower')
    plt.title('Subaperture masks')


# 2. Define the system modes
print('Obtaining the Karhunen-Loeve mirror modes ...')

KL_path = os.path.join(dir_path,'KLmatrix.fits')
m2c_path = os.path.join(dir_path,'m2c.fits')

try:
    KL = myfits.read_fits(KL_path)
    m2c = myfits.read_fits(m2c_path)
except FileNotFoundError:
    KL, m2c = make_KL_modal_base_from_dm(dm, Npix, oversampling, pupilSizeInM, r0, L0, zern2remove = 5)
    myfits.save_fits(KL_path, KL, hdr_dict)
    myfits.save_fits(m2c_path, m2c, hdr_dict)

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
    IM = calibrate_modes(wfs, ccd, KL, Npix, oversampling, lambdaInM, pupilSizeInM, amps = 0.1)
    # slope_STD = np.std(IM, axis=0)
    # IM = calibrate_modes(wfs, ccd, KL, Npix, oversampling, lambdaInM, pupilSizeInM, amps = 1/slope_STD)

    print('Computing the reconstructor ...')
    U,S,Vt = np.linalg.svd(IM, full_matrices=False)
    Rec = (Vt.T*1/S) @ U.T

    plt.figure()
    plt.plot(1/S,'o')
    plt.grid()
    plt.yscale('log')
    plt.title('Reconstructor eigenvalues')

    myfits.save_fits(IM_path, IM, hdr_dict)
    myfits.save_fits(Rec_path, Rec, hdr_dict)


# 4. Get atmospheric phase screen
print('Generating phase screens ...')

Nscreens = 1

atmo_path = os.path.join(dir_path, 'MyPhaseScreens.fits')
try:
    screen = myfits.read_fits(atmo_path)
except FileNotFoundError:
    screen = turbulence_generator(lambdaInM, r0, L0, Nscreens, \
     screenSizeInPixels=1024, screenSizeInMeters=1.0, savepath=atmo_path)
    myfits.save_fits(atmo_path, screen, hdr_dict)

input_phase = screen[0,:Npix*oversampling,:Npix*oversampling] * 1e-3

plt.figure()
plt.imshow(input_phase)
plt.colorbar()
plt.title('Atmo screen')


# 5. Perform the iteration
dm_cmd, ccd_image, pup_mask = perform_loop_iteration(wfs, ccd, dm, Rec, m2c, input_phase, Npix, oversampling, lambdaInM, pupilSizeInM)
dm_phase = dm.IFF @ dm_cmd

input_field = pup_mask * np.exp(1j*input_phase)
field_on_focal_plane = np.fft.fftshift(np.fft.fft2(input_field))
psf = np.abs(field_on_focal_plane)**2

plt.figure(figsize=(9,9))
plt.subplot(2,2,1)
plt.imshow(np.ma.masked_array(input_phase, mask = pup_mask),origin='lower')
plt.colorbar()
plt.title('Input phase')

plt.subplot(2,2,2)
plt.imshow(np.log(psf),origin='lower')
plt.colorbar()
plt.title('log PSF')

plt.subplot(2,2,3)
plt.imshow(ccd_image,origin='lower')
plt.colorbar()
plt.title('Detector image')

plt.subplot(2,2,4)
dm.plot_position(dm_cmd)
plt.title('Mirror command')


plt.show()





