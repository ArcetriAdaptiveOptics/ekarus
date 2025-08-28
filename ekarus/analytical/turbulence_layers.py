from arte.atmo.phase_screen_generator import PhaseScreenGenerator
import numpy as np

from ekarus.e2e.utils.image_utils import reshape_on_mask
# from functools import lru_cache
# import matplotlib.pyplot as plt

class Turbulence():

    def __init__(self, r0, L0, savepath:str = None, xp = np):

        self.r0 = r0
        self.L0 = L0

        self.savepath = savepath

        self._xp = xp
        self.dtype = self._xp.float32 if self._xp.__name__ == 'cupy' else self._xp.float64
    

    def generate_phasescreens(self, lambdaInM, Nscreens, screenSizeInPixels, screenSizeInMeters):

        phs = PhaseScreenGenerator(screenSizeInPixels, screenSizeInMeters, \
                                outerScaleInMeters=self.L0, seed=42)
        
        phs.generate_normalized_phase_screens(Nscreens)
        phs.rescale_to(r0At500nm=self.r0)
        phs.get_in_radians_at(wavelengthInMeters=lambdaInM)

        if self.savepath is not None :
            phs.save_normalized_phase_screens(self.savepath)

        phase_screens = self._xp.asarray(phs._phaseScreens, dtype=self.dtype)

        self.pixelsPerMeter = screenSizeInPixels/screenSizeInMeters

        return phase_screens


def move_mask_on_phasescreen(self, screen, mask, dt, wind_speed, wind_direction_angle):

    x_start, y_start = get_start_coordinates_on_phasescreen(screen.shape, mask.shape, wind_direction_angle)

    dpix = wind_speed*dt*self.pixelsPerMeter
    x = x_start + dpix*self._xp.cos(wind_direction_angle)
    y = y_start + dpix*self._xp.sin(wind_direction_angle)

    if x > screen.shape[1]-mask.shape[1] or y > screen.shape[0]-mask.shape[0] or x < 0 or y < 0:
        raise ValueError(f'A displacement of {dpix:1.2f} for a time {dt:1.3f} [s] with wind {wind_speed:1.1f} [m/s] yields\
                         ({y:1.0f},{x:1.0f}), which is outside the bounds for a {screen.shape} screen and a {mask.shape} mask.')

    x_round = int(self._xp.floor(x) * (x>=x_start) + self._xp.ceil(x) * (x<x_start))
    y_round = int(self._xp.floor(y) * (y>=y_start) + self._xp.ceil(y) * (y<y_start))

    dx, dy = abs(x-x_round), abs(y-y_round)
    sdx, sdy = int(self._xp.sign(x-x_start)), int(self._xp.sign(y-y_start))

    H,W = mask.shape

    full_mask = self._xp.ones_like(screen, dtype=bool)
    full_mask[y_round:(y_round+H),x_round:(x_round+W)] = mask.copy()
    phase = reshape_on_mask(screen[~full_mask], mask, xp=xp)

    thr = 1e-2

    if dx > thr and dy > thr:
        dx_phase = reshape_on_mask(screen[~self._xp.roll(full_mask,sdx,axis=1)], mask, xp=xp)
        dy_phase = reshape_on_mask(screen[~self._xp.roll(full_mask,sdy,axis=0)], mask, xp=xp)
        dxdy_phase = reshape_on_mask(screen[~self._xp.roll(full_mask,(sdx,sdy),axis=(0,1))], mask, xp=xp)
        phase_mask = (phase * (1-dx) + dx * dx_phase) * (1-dy) + (dy_phase * (1-dx) + dx * dxdy_phase) * dy

    elif dx > thr:
        dx_phase = reshape_on_mask(screen[~self._xp.roll(full_mask,sdx,axis=1)], mask, xp=xp)
        phase_mask = phase * (1-dx) + dx_phase * dx

    elif dy > thr:
        dy_phase = reshape_on_mask(screen[~self._xp.roll(full_mask,sdy,axis=0)], mask, xp=xp)
        phase_mask = phase * (1-dy) + dy_phase * dy

    else:
        phase_mask = phase.copy()

    return phase_mask


def get_start_coordinates_on_phasescreen(screen_shape, mask_shape, wind_direction_angle, xp=np):

    H,W = screen_shape
    h,w = mask_shape

    sin_phi = self._xp.sin(wind_direction_angle)
    cos_phi = self._xp.cos(wind_direction_angle)

    Delta = min(abs((W-w)/(2*cos_phi)), abs((H-h)/(2*sin_phi)))
    x = (W-w)/2 - Delta * cos_phi
    y = (H-h)/2 - Delta * sin_phi

    return x,y


