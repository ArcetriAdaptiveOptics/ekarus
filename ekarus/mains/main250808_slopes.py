import numpy as np
import matplotlib.pyplot as plt

from ekarus.e2e.pyramid_wfs import PyramidWFS
from ekarus.e2e.detector import Detector
from ekarus.e2e.slope_computer import SlopeComputer

from ekarus.analytical.zernike_generator import ZernikeGenerator
from ekarus.e2e.utils.image_utils import get_circular_mask

# System data
lambdaInM = 1000e-9
pupilSizeInM = 32e-3
oversampling = 4
Npix = 128

pixel_scale = lambdaInM/pupilSizeInM/oversampling
rad2arcsec = 180/np.pi*3600

apex_angle = 0.005859393655500168/(2*np.pi)*30/32
detector_shape = (256,256)
subaperture_size = 63.5

# Initialize devices
wfs = PyramidWFS(apex_angle)
ccd = Detector(detector_shape=detector_shape)
slope_computer = SlopeComputer(wfs_type = 'PyrWFS')

cmask = get_circular_mask((Npix*oversampling,Npix*oversampling), Npix//2)

# Define modulation angle
alpha = 3*lambdaInM/pupilSizeInM 

# Calibrate sensor
piston = 1-cmask
modulated_piston = wfs.modulate(piston, 4*alpha, pixel_scale)
calibration_image = ccd.image_on_detector(modulated_piston)
slope_computer.calibrate_sensor(subaperture_image=calibration_image, Npix=subaperture_size)


N = 4
Nzern = N*(N+1)//2 - 1
zg = ZernikeGenerator(cmask, Npix//2)
amp = 1

for i in range(Nzern):
    noll = i + 2

    phase = zg.getZernike(noll)*amp
    input_field = (1-cmask) * np.exp(1j*phase)

    modulated_intensity = wfs.modulate(input_field, alpha, pixel_scale)
    detector_image = ccd.image_on_detector(modulated_intensity)
    push_slope = slope_computer.compute_slopes(detector_image)/amp

    modulated_intensity = wfs.modulate(np.conj(input_field), alpha, pixel_scale)
    detector_image = ccd.image_on_detector(modulated_intensity)
    pull_slope = slope_computer.compute_slopes(detector_image)/amp

    slopes = (push_slope-pull_slope)/2

    plt.figure(figsize=(16,3))
    plt.subplot(1,4,1)
    plt.imshow(np.ma.masked_array(np.angle(input_field),mask=cmask),origin='lower')
    plt.colorbar()
    plt.title('Input field')

    plt.subplot(1,4,2)
    plt.imshow(modulated_intensity,origin='lower')
    plt.colorbar()
    plt.title('Modulated intensity')

    plt.subplot(1,4,3)
    plt.imshow(detector_image,origin='lower')
    plt.colorbar()
    plt.title('Detector image')

    plt.subplot(1,4,4)
    plt.plot(slopes)
    plt.grid()
    plt.title('Measured slope')


    plt.show()

