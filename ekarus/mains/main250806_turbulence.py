import numpy as np
import matplotlib.pyplot as plt
import os

from ekarus.e2e.utils.image_utils import get_circular_mask
from ekarus.e2e.utils import my_fits_package as myfits
from ekarus.e2e.utils.turbulence_generator import generate_phasescreens, move_mask_on_phasescreen

lambdaInM = 1000e-9
r0 = 5e-2
L0 = 25

screenMeters = 1.8*10
screenPixels = 2048

pixelsPerMeter = screenPixels/screenMeters

Nscreens = 1

basepath = os.getcwd()
dir_path = os.path.join(basepath,'ekarus/mains/250806_atmo_data/')

atmo_path = os.path.join(dir_path, 'AtmoScreens.fits')
try:
    screens = myfits.read_fits(atmo_path)
except FileNotFoundError:
    screens = generate_phasescreens(lambdaInM, r0, L0, Nscreens, \
     screenSizeInPixels=screenPixels, screenSizeInMeters=screenMeters, savepath=atmo_path)
    myfits.save_fits(atmo_path, screens)

screen = screens[0]

plt.figure()
plt.imshow(screen)
plt.colorbar()
plt.title('Atmo screen')

dt = 1e-2
wind_speed = 20
wind_angle = np.pi/2

mask_shape = (512,512)
mask = get_circular_mask(mask_shape,128)

N = 10
shifted_screens = np.zeros([512,512,N])

for i in range(N):
    tt = i*dt
    image = move_mask_on_phasescreen(screen, mask, tt, wind_speed, wind_angle, pixelsPerMeter)
    plt.figure()
    plt.imshow(image,origin='lower')
    plt.title(f'Wind = {wind_speed} [m/s], angle = {wind_angle*180/np.pi} [deg]')
    plt.show()
    shifted_screens[:,:,i] = image