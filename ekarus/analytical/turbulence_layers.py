import numpy as np
from arte.atmo.phase_screen_generator import PhaseScreenGenerator
from ekarus.e2e.utils.image_utils import reshape_on_mask

# from ekarus.e2e.utils import my_fits_package as myfits


class TurbulenceLayers():

    def __init__(self, r0s, L0, windSpeeds, windAngles, savepath:str, xp = np):

        self.r0s = r0s
        self.L0 = L0
        self.windAngles = windAngles
        self.windSpeeds = windSpeeds

        self._xp = xp
        self.dtype = self._xp.float32 if self._xp.__name__ == 'cupy' else self._xp.float64

        self._check_same_length(r0s, windSpeeds)
        self._check_same_length(r0s, windAngles)

        self.Nscreens = self._get_len(r0s, xp)
        self.savepath = savepath


    def update_mask(self, mask):
        self.mask = mask
        self._define_start_coordinates_on_phasescreens(mask.shape)


    def generate_phase_screens(self, screenSizeInPixels, screenSizeInMeters):
        
        self.pixelsPerMeter = screenSizeInPixels/screenSizeInMeters
        self._phs = PhaseScreenGenerator(screenSizeInPixels, screenSizeInMeters, \
                            outerScaleInMeters=self.L0, seed=42)
        try:
            self._phs = self._phs.load_normalized_phase_screens(self.savepath)
        except FileNotFoundError:   
            print(f'Generating {self.Nscreens:1.0f} phase-screens ...')
            self._phs.generate_normalized_phase_screens(self.Nscreens)
            self._phs.save_normalized_phase_screens(self.savepath)
        
        self.phase_screens = self._xp.asarray(self._phs._phaseScreens, dtype=self.dtype)
        # self._phs.rescale_to(r0At500nm=self.r0s)
        self._normalization_factors = (1/self.pixelsPerMeter / self.r0s) ** (5. / 6)
        self.screen_shape = self._phs._phaseScreens.shape[1:]
        # print(self.screen_shape)


    def rescale_phasescreens(self, lambdaInM):
        # self._phs.get_in_radians_at(wavelengthInMeters=lambdaInM)
        for k in range(self.Nscreens):
            self.phase_screens[k,:,:] *= 500e-9 / lambdaInM * self._normalization_factors[k]
        
        
    def move_mask_on_phasescreens(self, dt):
        masked_phases = self._xp.zeros([self.Nscreens,self.mask_shape[0],self.mask_shape[1]])
        screen_shape = self.screen_shape
        mask_shape = self.mask.shape

        for k in range(self.Nscreens):

            screen = self.phase_screens[k]
            windSpeed = self.windSpeeds[k]
            windAngle = self.windAngles[k]
            xStart = self.startX[k]
            yStart = self.startY[k]

            dpix = windSpeed*dt*self.pixelsPerMeter
            x = xStart + dpix*self._xp.cos(windAngle)
            y = yStart + dpix*self._xp.sin(windAngle)

            if x > screen_shape[1]-mask_shape[1] or y > screen_shape[0]-mask_shape[0] or x < 0 or y < 0:
                raise ValueError(f'A displacement of {dpix:1.2f} for a time {dt:1.3f} [s] with wind {windSpeed:1.1f} [m/s] yields\
                                ({y:1.0f},{x:1.0f}), which is outside the bounds for a {screen_shape} screen and a {mask_shape} mask.')

            x_round = int(self._xp.floor(x) * (x>=xStart) + self._xp.ceil(x) * (x<xStart))
            y_round = int(self._xp.floor(y) * (y>=yStart) + self._xp.ceil(y) * (y<yStart))

            dx, dy = abs(x-x_round), abs(y-y_round)
            sdx, sdy = int(self._xp.sign(x-xStart)), int(self._xp.sign(y-yStart))

            H,W = mask_shape

            full_mask = self._xp.ones_like(screen, dtype=bool)
            full_mask[y_round:(y_round+H),x_round:(x_round+W)] = self.mask.copy()
            phase = reshape_on_mask(screen[~full_mask], self.mask, xp=self._xp)

            thr = 1e-2

            if dx > thr and dy > thr:
                dx_phase = reshape_on_mask(screen[~self._xp.roll(full_mask,sdx,axis=1)], self.mask, xp=self._xp)
                dy_phase = reshape_on_mask(screen[~self._xp.roll(full_mask,sdy,axis=0)], self.mask, xp=self._xp)
                dxdy_phase = reshape_on_mask(screen[~self._xp.roll(full_mask,(sdx,sdy),axis=(0,1))], self.mask, xp=self._xp)
                phase_mask = (phase * (1-dx) + dx * dx_phase) * (1-dy) + (dy_phase * (1-dx) + dx * dxdy_phase) * dy

            elif dx > thr:
                dx_phase = reshape_on_mask(screen[~self._xp.roll(full_mask,sdx,axis=1)], self.mask, xp=self._xp)
                phase_mask = phase * (1-dx) + dx_phase * dx

            elif dy > thr:
                dy_phase = reshape_on_mask(screen[~self._xp.roll(full_mask,sdy,axis=0)], self.mask, xp=self._xp)
                phase_mask = phase * (1-dy) + dy_phase * dy

            else:
                phase_mask = phase.copy()

            masked_phases[k,:,:] = phase_mask

        return masked_phases
        

    def _define_start_coordinates_on_phasescreens(self, mask_shape):

        self.mask_shape = mask_shape

        H,W = self.screen_shape
        h,w = self.mask_shape

        self.startX = self._xp.zeros(self.Nscreens)
        self.startY = self._xp.zeros(self.Nscreens)

        for k in range(self.Nscreens):

            windAngle = self.windAngles[k]

            sin_phi = max(1e-12, self._xp.sin(windAngle)) # avoid division by 0
            cos_phi = max(1e-12, self._xp.cos(windAngle)) # avoid division by 0

            Delta = min(abs((W-w)/(2*cos_phi)), abs((H-h)/(2*sin_phi)))
            self.startX[k] = (W-w)/2 - Delta * cos_phi
            self.startY[k] = (H-h)/2 - Delta * sin_phi
    
    
    def _check_same_length(self, a, b):
        lenA = self._get_len(a, self._xp)
        lenB = self._get_len(b, self._xp)
        if lenA != lenB:
            raise ValueError(f'Vector {a} of length {lenA} is not compatible with vector {b} of length {lenB}')
    
    @staticmethod
    def _get_len(a, xp=np):
        length = 1
        if not isinstance(a, float):
            length = len(a)
        return length




