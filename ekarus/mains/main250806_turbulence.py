import numpy as np
import matplotlib.pyplot as plt
import os

from ekarus.e2e.utils.image_utils import get_circular_mask
from ekarus.e2e.utils import my_fits_package as myfits
from ekarus.e2e.utils.turbulence_generator import *

lambdaInM = 1000e-9
r0 = 5e-2
L0 = 25

screenMeters = 1.8*10
screenPixels = 2048

pixelsPerMeter = screenPixels/screenMeters

Nscreens = 1

basepath = os.getcwd()
dir_path = os.path.join(basepath,'ekarus/mains/250806_atmo_data/')

try:
    os.mkdir(dir_path)
except FileExistsError:
    pass

atmo_path = os.path.join(dir_path, 'AtmoScreens.fits')
try:
    screens = myfits.read_fits(atmo_path)
except FileNotFoundError:
    screens = generate_phasescreens(lambdaInM, r0, L0, Nscreens, \
     screenSizeInPixels=screenPixels, screenSizeInMeters=screenMeters, savepath=atmo_path)
    myfits.save_fits(atmo_path, screens)

screen = screens[0]



dt = 1e-2
wind_speed = 7.5
wind_angle = np.pi/4

mask_shape = (400,300)
mx,my = mask_shape
mask = get_circular_mask(mask_shape, 128)


full_mask = np.ones_like(screen)
x0,y0 = update_coordinates_on_phasescreen(screen, mask.shape, 0, wind_speed, wind_angle, pixelsPerMeter)

x0 = int(np.ceil(x0))
y0 = int(np.ceil(y0))

print(x0,y0)

full_mask[(x0-mx):x0,(y0-my):y0] = mask
dx_mask = np.roll(full_mask,128*2,axis=1)
dy_mask = np.roll(full_mask,128*2,axis=0)

plt.figure()
plt.imshow(np.ma.masked_array(screen,full_mask),origin='lower')
plt.imshow(np.ma.masked_array(screen,dx_mask),origin='lower')
plt.imshow(np.ma.masked_array(screen,dy_mask),origin='lower')
plt.show()


N = 21
shifted_screens = np.zeros([400,300,N])
x = np.zeros(N)
y = np.zeros(N)

for i in range(N):
    tt = i*dt
    image = move_mask_on_phasescreen(screen, mask, tt, wind_speed, wind_angle, pixelsPerMeter)
    x[i],y[i] = update_coordinates_on_phasescreen(screen, mask.shape, tt, wind_speed, wind_angle, pixelsPerMeter)
    shifted_screens[:,:,i] = image

print(x,y)


plt.figure()
plt.imshow(screen,origin='lower')
plt.colorbar()
plt.scatter(x,y,c='red')
plt.title('Atmo screen')

Nrows = N//6
Ncols = int(np.ceil(N/Nrows))

plt.figure(figsize=(18,Nrows*3))
for i in range(N):
    plt.subplot(Nrows,Ncols,i+1)
    plt.imshow(shifted_screens[:,:,i],origin='lower')
    plt.title(f'Step {i}')

plt.show()