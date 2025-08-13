from arte.atmo.phase_screen_generator import PhaseScreenGenerator
import numpy as np

# class Turbulence():

#     def __init__(self, r0, L0):

#         self.r0 = r0
#         self.L0 = L0
    

def generate_phasescreens(lambdaInM, r0, L0, Nscreens, screenSizeInPixels, screenSizeInMeters, savepath:str= None, xp=np, dtype=np.float32):

    phs = PhaseScreenGenerator(screenSizeInPixels, screenSizeInMeters, \
                            outerScaleInMeters=L0, seed=42)
    
    phs.generate_normalized_phase_screens(Nscreens)
    phs.rescale_to(r0At500nm=r0)
    phs.get_in_radians_at(wavelengthInMeters=lambdaInM)

    if savepath is not None :
        phs.save_normalized_phase_screens(savepath)

    phase_screens = xp.asarray(phs._phaseScreens, dtype=dtype)

    return phase_screens


def move_mask_on_phasescreen(screen, mask, dt, wind_speed, wind_direction_angle, pixelsPerMeter, xp=np):

    x_start, y_start = get_start_coordinates_on_phasescreen(screen.shape, mask.shape, wind_direction_angle, xp=xp)

    dpix = wind_speed*dt*pixelsPerMeter
    x = x_start + dpix*xp.cos(wind_direction_angle)
    y = y_start + dpix*xp.sin(-wind_direction_angle)

    if x > screen.shape[1]-mask.shape[1] or y > screen.shape[0]-mask.shape[0] or x < 0 or y < 0:
        raise ValueError(f'A displacement of {dpix:1.2f} for a time {dt:1.3f} [s] with wind {wind_speed:1.1f} [m/s] yields\
                         ({y:1.0f},{x:1.0f}), which is outside the bounds for a {screen.shape} screen and a {mask.shape} mask.')

    x_floor = int(xp.floor(x))
    y_floor = int(xp.floor(y))

    H,W = mask.shape

    full_mask = xp.ones_like(screen, dtype=bool)
    full_mask[y_floor:(y_floor+H),x_floor:(x_floor+W)] = mask
    phase = screen[~full_mask]

    thr = 1e-4
    interp_phase = phase.copy()

    if x-x_floor > thr and y-y_floor > thr:
        dx, dy = x-x_floor, y-y_floor
        dx_phase = screen[~xp.roll(full_mask,1,axis=1)]
        dy_phase = screen[~xp.roll(full_mask,1,axis=0)]
        dxdy_phase = screen[~xp.roll(full_mask,(1,1),axis=(0,1))]
        interp_phase = (phase * dx + (1-dx) * dx_phase) * dy + (dy_phase * dx + (1-dx) * dxdy_phase) * (1-dy)

    elif x-x_floor > thr:
        dx = x-x_floor
        dx_phase = screen[~xp.roll(full_mask,1,axis=1)]
        interp_phase *= dx + dx_phase * (1-dx)

    elif y-y_floor > thr:
        dy = y-y_floor
        dy_phase = screen[~xp.roll(full_mask,1,axis=0)]
        interp_phase *= dy + dy_phase * (1-dy)

    phase_mask = xp.zeros(mask.shape)
    phase_mask[~mask] = interp_phase

    return phase_mask


def get_start_coordinates_on_phasescreen(screen_shape, mask_shape, wind_direction_angle, xp=np):

    H,W = screen_shape
    h,w = mask_shape

    sin_phi = xp.sin(-wind_direction_angle)
    cos_phi = xp.cos(wind_direction_angle)

    Delta = min(abs((W-w)/(2*cos_phi)), abs((H-h)/(2*sin_phi)))
    x = (W-w)/2 - Delta * cos_phi
    y = (H-h)/2 - Delta * sin_phi

    return x,y


