import numpy as np
import matplotlib.pyplot as plt
import os

from ekarus.e2e.utils.image_utils import get_circular_mask
# from ekarus.e2e.utils import my_fits_package as myfits
from ekarus.analytical.turbulence_layers import TurbulenceLayers

from root import atmopath

lambdaInM = 1000e-9
r0 = 5e-2
L0 = 25

screenMeters = 1.8*10
screenPixels = 2048

wind_speed = 7.5
wind_angle = 0

pixelsPerMeter = screenPixels/screenMeters

Nscreens = 1

basepath = os.getcwd()
dir_path = os.path.join(atmopath,'250806_atmo_data')

if not os.path.exists(dir_path):
    os.makedirs(dir_path)

atmo_path = os.path.join(dir_path, 'AtmoScreens.fits')


# Define turbulence class
atmo = TurbulenceLayers(r0, L0, wind_speed, wind_angle, atmo_path)
atmo.generate_phase_screens(screenSizeInPixels=screenPixels, screenSizeInMeters=screenMeters)
atmo.rescale_phasescreens(lambdaInM=lambdaInM)

dt = 1e-2

# Generate mask
mask_shape = (400,300)
H,W = mask_shape
mask = get_circular_mask(mask_shape, 128)

atmo.update_mask(mask)
full_mask = np.ones(atmo.screen_shape)
x0 = atmo.startX
y0 = atmo.startY
x0 = int(np.ceil(x0))
y0 = int(np.ceil(y0))

full_mask[y0:(H+y0),x0:(W+x0)] = mask
dx_mask = np.roll(full_mask,128*2,axis=1)
dy_mask = np.roll(full_mask,128*2,axis=0)

screen = atmo.phase_screens[0,:,:]

plt.figure()
plt.imshow(np.ma.masked_array(screen,full_mask),origin='lower')
plt.imshow(np.ma.masked_array(screen,dx_mask),origin='lower')
plt.imshow(np.ma.masked_array(screen,dy_mask),origin='lower')

N = 21
shifted_screens = np.zeros([400,300,N])
x = np.zeros(N)
y = np.zeros(N)

for i in range(N):
    tt = i*dt
    image = atmo.move_mask_on_phasescreens(tt)
    shifted_screens[:,:,i] = image


# plt.figure()
# plt.imshow(screen,origin='lower')
# plt.colorbar()
# # plt.scatter(x,y,c='red')
# plt.title('Atmo screen')

Nrows = N//6
Ncols = int(np.ceil(N/Nrows))

plt.figure(figsize=(18,Nrows*3))
for i in range(N):
    plt.subplot(Nrows,Ncols,i+1)
    plt.imshow(shifted_screens[:,:,i],origin='lower')
    plt.title(f'Step {i}')

plt.show()