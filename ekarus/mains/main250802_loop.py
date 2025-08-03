import numpy as np
import matplotlib.pyplot as plt
import os

from ekarus.e2e.utils import my_fits_package as myfits

from ekarus.e2e.alpao_deformable_mirror import ALPAODM
from ekarus.e2e.pyramid_wfs import PyramidWFS
from ekarus.e2e.detector import Detector

from ekarus.e2e.scao_class import SCAO
from ekarus.e2e.utils.turbulence_generator import turbulence_generator


def imageShow(image2d, pixelSize=1, title='', xlabel='', ylabel='', zlabel='', shrink=1.0):
    sz=image2d.shape
    plt.imshow(image2d, extent=[-sz[0]/2*pixelSize, sz[0]/2*pixelSize,
                                -sz[1]/2*pixelSize, sz[1]/2*pixelSize],origin='lower')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    cbar= plt.colorbar(shrink=shrink)
    cbar.ax.set_ylabel(zlabel)

def showZoomCenter(image, pixelSize, **kwargs):
    '''show log(image) zoomed around center'''
    imageHalfSizeInPoints= image.shape[0]/2
    roi= [int(imageHalfSizeInPoints*0.8), int(imageHalfSizeInPoints*1.2)]
    imageZoomedLog= np.log(image[roi[0]: roi[1], roi[0]:roi[1]])
    imageShow(imageZoomedLog, pixelSize=pixelSize, **kwargs)


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
L0 = 25

# Sensor, detector, DM
print('Initializing sensor, detector and deformable mirror ...')
wfs = PyramidWFS(apex_angle)
ccd = Detector(detector_shape=detector_shape)
dm = ALPAODM(468)

scao = SCAO(wfs, ccd, dm, Npix, pupilSizeInM, modulationAngleInLambdaoverD=3, oversampling=oversampling)

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
    scao.ccd.subapertures = myfits.read_fits(subap_path, isBool=True)
except FileNotFoundError:
    scao.define_subaperture_masks(lambdaInM, subapertureSizeInPixels=subaperture_size)
    myfits.save_fits(subap_path, (ccd.subapertures).astype(np.uint8), hdr_dict)


# 2. Define the system modes
print('Obtaining the Karhunen-Loeve mirror modes ...')

KL_path = os.path.join(dir_path,'KLmatrix.fits')
m2c_path = os.path.join(dir_path,'m2c.fits')

try:
    scao.KL = myfits.read_fits(KL_path)
    scao.m2c = myfits.read_fits(m2c_path)
except FileNotFoundError:
    scao.define_KL_modal_base(r0, L0, zern2remove = 5)
    myfits.save_fits(KL_path, scao.KL, hdr_dict)
    myfits.save_fits(m2c_path, scao.m2c, hdr_dict)

KL = scao.KL
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
    scao.IM = myfits.read_fits(IM_path)
    scao.Rec = myfits.read_fits(Rec_path)

except FileNotFoundError:
    scao.calibrate_modes(KL, lambdaInM, amps = 0.1)

    myfits.save_fits(IM_path, scao.IM, hdr_dict)
    myfits.save_fits(Rec_path, scao.Rec, hdr_dict)


# 4. Get atmospheric phase screen
print('Generating phase screens ...')

Nscreens = 1
Npupils = 10

atmo_path = os.path.join(dir_path, 'AtmoScreens.fits')
try:
    screen = myfits.read_fits(atmo_path)
except FileNotFoundError:
    screen = turbulence_generator(lambdaInM, r0, L0, Nscreens, \
     screenSizeInPixels=Npix*oversampling*Npupils, screenSizeInMeters=Npupils*oversampling*pupilSizeInM, savepath=atmo_path)
    myfits.save_fits(atmo_path, screen, hdr_dict)

input_phase = screen[0,:Npix*oversampling,:Npix*oversampling]
input_phase *= 1e-2

plt.figure()
plt.imshow(input_phase)
plt.colorbar()
plt.title('Atmo screen')


# 5. Perform the iteration
phase = input_phase.copy()
phase -= np.mean(phase[~scao.cmask.mask()])
g = 0.005

for i in range(10):
    dm_cmd, ccd_image = scao.perform_loop_iteration(phase, lambdaInM)
    dm_cmd *= g
    dm_shape = dm.IFF @ dm_cmd
    
    dm_phase = np.zeros_like(dm.mask,dtype=np.float32)
    dm_phase[~dm.mask] = dm_shape/pupilSizeInM*(2*np.pi)
    dm_phase = np.pad(dm_phase, (Npix*(oversampling-1))//2)

    input_field = scao.cmask.mask() * np.exp(1j*phase)
    field_on_focal_plane = np.fft.fftshift(np.fft.fft2(input_field))
    psf = np.abs(field_on_focal_plane)**2

    plt.figure(figsize=(12,12))
    plt.subplot(2,2,1)
    plt.imshow(np.ma.masked_array(phase, mask = scao.cmask.mask()),origin='lower')
    plt.colorbar()
    plt.title('Input phase')

    plt.subplot(2,2,2)
    showZoomCenter(psf, pix2rad, title=r'Zoomed PSF (D=%g m, $\lambda$=%g $\mu m$)' % (pupilSizeInM, lambdaInM*1e6))
    # plt.imshow(np.log(psf),origin='lower')
    # plt.colorbar()
    # plt.title('log PSF')

    plt.subplot(2,2,3)
    plt.imshow(ccd_image,origin='lower')
    plt.colorbar()
    plt.title('Detector image')

    plt.subplot(2,2,4)
    dm.plot_position(dm_cmd)
    plt.title('Mirror command')

    plt.show()

    phase -= dm_phase





