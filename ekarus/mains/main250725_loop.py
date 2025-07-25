import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits as pyfits
import os

from arte.math.toccd import toccd
from arte.types.mask import CircularMask
from arte.atmo.phase_screen_generator import PhaseScreenGenerator

from ekarus.e2e.alpao_deformable_mirror import ALPAODM
from ekarus.e2e.pyramid_wfs import PyramidWFS
from ekarus.e2e.detector import Detector

from ekarus.e2e.utils.zernike_coefficients import create_field_from_zernike_coefficients
from ekarus.e2e.utils.image_utils import compute_pixel_size
from ekarus.e2e.utils.kl_modes import make_modal_base_from_ifs_fft

# 1. Define the system
lambda0 = 1000e-9
pupil_diameter = 30e-3
oversampling = 4
Npix = 128
pix2rad = compute_pixel_size(wavelength=lambda0, pupil_diameter_in_m=pupil_diameter, padding=oversampling)

# Pyramid WFS
apex_angle = 0.005859393655500168 # vertex angle in radians
wfs = PyramidWFS(apex_angle)

# CCD
detector_shape = (256,256)
ccd = Detector(detector_shape=detector_shape)

# DM 
print('Initializing ALPAO DM ...')
dm = ALPAODM(468)

# Modulation angle
alpha = 3*lambda0/pupil_diameter


basepath = os.getcwd()
try:
    dir_path = os.path.join(basepath,'ekarus/mains/250725_sim_data/')
    os.mkdir(dir_path)
except FileExistsError:
    pass

# 2. Define the subapertures using a piston input wavefront
print('Defining the detector subaperture masks ...')
mask = CircularMask((oversampling * Npix, oversampling * Npix), maskRadius=Npix // 2)
piston =  create_field_from_zernike_coefficients(mask, 1, 1)

modulated_intensity = wfs.modulate(piston, 10*alpha, pix2rad)

subap_path = os.path.join(basepath,'ekarus/mains/250725_sim_data/SubapertureMasks.fits')
centers_path = os.path.join(basepath,'ekarus/mains/250725_sim_data/SubapertureCenters.fits')
try:
    subap_hdu = pyfits.open(subap_path)
    ccd.subapertures = (np.array(subap_hdu[0].data)).astype(bool)
    centers_hdu = pyfits.open(centers_path)
    ccd.subaperture_centers = np.array(centers_hdu[0].data)
except FileNotFoundError:
    ccd.define_subaperture_masks(modulated_intensity, Npix = 63.5)
    hdr = pyfits.Header()
    hdr['SHAPE'] = ccd.detector_shape
    hdr['PYR_ANG'] = wfs.apex_angle
    pyfits.writeto(subap_path, (ccd.subapertures).astype(np.uint8), hdr, overwrite=True)
    pyfits.writeto(centers_path, ccd.subaperture_centers, hdr, overwrite=True)

ccd_intensity = toccd(modulated_intensity,ccd.detector_shape)
# TO DO: Add detector noise!!!

plt.figure()
plt.imshow(ccd_intensity-(ccd.subapertures[0]+ccd.subapertures[1]+ccd.subapertures[2]+ccd.subapertures[3])/4, origin='lower')
plt.scatter(ccd.subaperture_centers[:,0],ccd.subaperture_centers[:,1],color='red')
plt.title('Subaperture masks')


# 3. Calibrate the system
r0 = 5e-2
L0 = 8

KL_path = os.path.join(basepath,'ekarus/mains/250725_sim_data/KLmatrix.fits')
m2c_path = os.path.join(basepath,'ekarus/mains/250725_sim_data/m2c.fits')

print('Obtaining the Karhunen-Loeve mirror modes ...')
try:
    kl_hdu = pyfits.open(KL_path)
    KL = np.array(kl_hdu[0].data)
    m2c_hdu = pyfits.open(m2c_path)
    m2c = np.array(m2c_hdu[0].data)
except FileNotFoundError:
    KL, m2c, singular_values = make_modal_base_from_ifs_fft(1-dm.mask, pupil_diameter, \
    dm.IFF.T, r0, L0, zern_modes=7, zern_mask = mask, oversampling=oversampling, verbose = True)
    hdr = pyfits.Header()
    hdr['N_ACTS'] = dm.Nacts
    hdr['PUP_DIAM'] = pupil_diameter
    hdr['FR_PAR'] = r0
    hdr['ATMO_LEN'] = L0
    pyfits.writeto(KL_path, KL, hdr, overwrite=True)
    pyfits.writeto(m2c_path, m2c, hdr, overwrite=True)


N=9
plt.figure(figsize=(3*N,10))

for i in range(N):
    plt.subplot(4,N,i+1)
    dm.plot_surface(KL[i,:],title=f'KL Mode {i}')
    plt.subplot(4,N,i+1+N)
    dm.plot_surface(KL[i+N,:],title=f'KL Mode {i+N}')
    plt.subplot(4,N,i+1+N*2)
    dm.plot_surface(KL[-i-1-N,:],title=f'KL Mode {np.shape(KL)[0]-i-1-N}')
    plt.subplot(4,N,i+1+N*3)
    dm.plot_surface(KL[-i-1,:],title=f'KL Mode {np.shape(KL)[0]-i-1}')

print('Calibrating the KL modes ...')

IM_path = os.path.join(basepath,'ekarus/mains/250725_sim_data/IMmatrix.fits')
Rec_path = os.path.join(basepath,'ekarus/mains/250725_sim_data/Rec.fits')


try:
    im_hdu = pyfits.open(IM_path)
    IM= np.array(im_hdu[0].data)
    Rec_hdu = pyfits.open(Rec_path)
    Rec = np.array(Rec_hdu[0].data)

except FileNotFoundError:

    slope_len = int(np.sum(1-ccd.subapertures[0])*2)
    IM = np.zeros((slope_len,np.shape(KL)[0]))
    amp = 0.1

    for i in range(np.shape(KL)[0]):

        kl_phase = np.zeros(mask.mask().shape)
        kl_phase[~mask.mask()] = KL[i,:]*amp
        kl_phase = np.reshape(kl_phase,mask.mask().shape)
        input_field = np.exp(1j*kl_phase) * mask.asTransmissionValue()
        modulated_intensity = wfs.modulate(input_field, alpha, pix2rad)
        push_slope = ccd.compute_slopes(modulated_intensity)/amp

        input_field = np.conj(input_field)
        modulated_intensity = wfs.modulate(input_field, alpha, pix2rad)
        pull_slope = ccd.compute_slopes(modulated_intensity)/amp

        IM[:,i] = (push_slope-pull_slope)/2

    print('Computing the reconstructor ...')
    U,S,Vt = np.linalg.svd(IM, full_matrices=False)
    Sinv = 1/S
    Sinv[Sinv>1e+6] = 0
    Rec = (Vt.T*Sinv) @ U.T

    plt.figure()
    plt.plot(Sinv,'o')
    plt.grid()
    plt.yscale('log')
    plt.title('Reconstructor eigenvalues')

    hdr = pyfits.Header()
    hdr['N_ACTS'] = dm.Nacts
    hdr['PUP_DIAM'] = pupil_diameter
    hdr['FR_PAR'] = r0
    hdr['ATMO_LEN'] = L0
    pyfits.writeto(IM_path, IM, hdr, overwrite=True)
    pyfits.writeto(Rec_path, Rec, hdr, overwrite=True)


# 4. Test the reconstruction
print('Generating phase screens ...')
D = 1.8
m2pix = 1024
Nscreens = 1
phs = PhaseScreenGenerator(screenSizeInPixels=int(D*m2pix), screenSizeInMeters=D, outerScaleInMeters=L0, seed=42)

fname = os.path.join(basepath, 'ekarus/mains/250725_sim_data/MyPhaseScreens.fits')
try:
    phs = phs.load_normalized_phase_screens(fname)
except FileNotFoundError:
    phs.generate_normalized_phase_screens(Nscreens)
    phs.rescale_to(r0At500nm=r0)
    phs.get_in_radians_at(wavelengthInMeters=lambda0)
    phs.save_normalized_phase_screens(fname)

screen = phs._phaseScreens

phase = screen[0,:Npix*oversampling,:Npix*oversampling]
phase *= 0.1/np.max(phase)

plt.figure()
plt.imshow(phase)
plt.colorbar()
plt.title('Atmo screen')

input_field = mask.asTransmissionValue() * np.exp(1j*phase)
meas_intensity = wfs.modulate(input_field, alpha, pix2rad)
slopes = ccd.compute_slopes(meas_intensity)
modes = Rec @ slopes
cmd = m2c @ modes
dm_shape = dm.IFF @ cmd
flat_img = np.zeros(np.size(mask.mask()))
flat_img[~mask.mask().flatten()] = dm_shape
dm_phase = np.reshape(flat_img, mask.mask().shape)

plt.figure(figsize=(12,6))
plt.subplot(1,3,1)
plt.imshow(np.ma.masked_array(phase, mask = mask.mask()),origin='lower')
plt.colorbar(shrink=0.5)
plt.title('Input phase')

plt.subplot(1,3,2)
plt.imshow(np.ma.masked_array(dm_phase, mask = mask.mask()),origin='lower')
plt.colorbar(shrink=0.5)
plt.title('DM phase')

plt.subplot(1,3,3)
plt.imshow(np.ma.masked_array(phase-dm_phase, mask = mask.mask()),origin='lower')
plt.colorbar(shrink=0.5)
plt.title('Output phase')

plt.show()





