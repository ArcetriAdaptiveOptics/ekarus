import numpy as np
from arte.atmo.phase_screen_generator import PhaseScreenGenerator
from ekarus.e2e.utils.image_utils import reshape_on_mask

# from ekarus.e2e.utils import my_fits_package as myfits


class TurbulenceLayers():

    def __init__(self, r0s, L0, windSpeeds, windAngles, savepath:str, xp = np):

        self.r0s = r0s
        self.L0 = L0

        # Wrap angles to [-pi,pi)
        angles = ( windAngles + xp.pi) % (2 * xp.pi ) - xp.pi
        angles += 2*xp.pi * (angles < 0)
        self.windAngles = angles
        self.windSpeeds = windSpeeds

        self._xp = xp
        self.dtype = self._xp.float32 if self._xp.__name__ == 'cupy' else self._xp.float64

        self._check_same_length(r0s, windSpeeds)
        self._check_same_length(r0s, windAngles)

        self.nLayers = self._get_len(r0s)
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
            N, Npix = self.nLayers, screenSizeInPixels
            if N > 1:
                print(f'Generating {N:1.0f} {Npix:1.0f}x{Npix:1.0f} phase-screens ...')
            else:
                print(f'Generating {Npix:1.0f}x{Npix:1.0f} phase-screen ...')
            self._phs.generate_normalized_phase_screens(N)
            self._phs.save_normalized_phase_screens(self.savepath)
        
        self.phase_screens = self._xp.asarray(self._phs._phaseScreens, dtype=self.dtype)
        self._normalization_factors = (1/self.pixelsPerMeter / self.r0s) ** (5. / 6)
        self.screen_shape = self._phs._phaseScreens.shape[1:]


    def rescale_phasescreens(self):
        if self.nLayers > 1:
            for k in range(self.nLayers):
                self.phase_screens[k,:,:] *= 500e-9 / (2*self._xp.pi) * self._normalization_factors[k]
        else:
            self.phase_screens[0,:,:] *= 500e-9 / (2*self._xp.pi) * self._normalization_factors
    

    def rescale_phasescreens_in_radians(self, lambdaInM):
        if self.nLayers > 1:
            for k in range(self.nLayers):
                self.phase_screens[k,:,:] *= 500e-9 / lambdaInM * self._normalization_factors[k]
        else:
            self.phase_screens[0,:,:] *= 500e-9 / lambdaInM * self._normalization_factors
        
        
    def move_mask_on_phasescreens(self, dt):

        masked_phases = self._xp.zeros([self.nLayers,self.mask_shape[0],self.mask_shape[1]])
        if self.nLayers > 1:
            for k in range(self.nLayers):
                masked_phases[k,:,:] = self._get_single_masked_phase(dt, self.phase_screens[k], self.windSpeeds[k], \
                                                                self.windAngles[k], self.startX[k], self.startY[k])
        else:
            masked_phases[0,:,:] = self._get_single_masked_phase(dt, self.phase_screens[0], self.windSpeeds, \
                                                                self.windAngles, self.startX, self.startY)

        return masked_phases
    

    def _get_single_masked_phase(self, dt, screen, windSpeed, windAngle, xStart, yStart):
        screen_shape = self.screen_shape
        mask_shape = self.mask.shape

        dpix = windSpeed*dt*self.pixelsPerMeter
        x = xStart + dpix*self._xp.cos(windAngle)
        y = yStart + dpix*self._xp.sin(windAngle)

        if x > screen_shape[1]-mask_shape[1] or y > screen_shape[0]-mask_shape[0] or x < 0 or y < 0:
            raise ValueError(f'A displacement of {dpix:1.2f} for a time {dt:1.3f} [s] with wind {windSpeed:1.1f} [m/s] yields\
                            ({y:1.0f},{x:1.0f}), which is outside the bounds for a {screen_shape} screen and a {mask_shape} mask.')

        x_round = int(self._xp.floor(x) * (x>=xStart) + self._xp.ceil(x) * (x<xStart))
        y_round = int(self._xp.floor(y) * (y>=yStart) + self._xp.ceil(y) * (y<yStart))

        dx, dy = abs(x-x_round), abs(y-y_round)
        sdx, sdy = int(self._xp.sign(x-x_round)), int(self._xp.sign(y-y_round)) # sdx, sdy = int(self._xp.sign(x-xStart)), int(self._xp.sign(y-yStart))

        H,W = mask_shape

        full_mask = self._xp.ones_like(screen, dtype=bool)
        full_mask[y_round:(y_round+H),x_round:(x_round+W)] = self.mask.copy()
        phase = reshape_on_mask(screen[~full_mask], self.mask, xp=self._xp)

        thr = 1e-8

        if dx > thr and dy > thr:
            dx_phase = reshape_on_mask(screen[~self._xp.roll(full_mask,sdx,axis=1)], self.mask, xp=self._xp)
            dy_phase = reshape_on_mask(screen[~self._xp.roll(full_mask,sdy,axis=0)], self.mask, xp=self._xp)
            dxdy_phase = reshape_on_mask(screen[~self._xp.roll(full_mask,(sdy,sdx),axis=(0,1))], self.mask, xp=self._xp)
            phase_mask = (phase * (1-dx) + dx * dx_phase) * (1-dy) + (dy_phase * (1-dx) + dx * dxdy_phase) * dy

        elif dx > thr:
            dx_phase = reshape_on_mask(screen[~self._xp.roll(full_mask,sdx,axis=1)], self.mask, xp=self._xp)
            phase_mask = phase * (1-dx) + dx_phase * dx

        elif dy > thr:
            dy_phase = reshape_on_mask(screen[~self._xp.roll(full_mask,sdy,axis=0)], self.mask, xp=self._xp)
            phase_mask = phase * (1-dy) + dy_phase * dy

        else:
            phase_mask = phase.copy()
        
        return phase_mask
    

    def _define_start_coordinates_on_phasescreens(self, mask_shape):

        self.mask_shape = mask_shape

        H,W = self.screen_shape
        h,w = self.mask_shape

        self.startX = self._xp.zeros(self.nLayers)
        self.startY = self._xp.zeros(self.nLayers)

        for k in range(self.nLayers):

            if self.nLayers > 1:
                windAngle = self.windAngles[k]
            else: 
                windAngle = self.windAngles

            sin_phi = self._xp.sin(windAngle) 
            cos_phi = self._xp.cos(windAngle) 

            # Avoid division by 0
            if abs(sin_phi) < 1e-12:
                sin_phi = 1e-12*self._xp.sign(sin_phi)
            if abs(cos_phi) < 1e-12:
                cos_phi = 1e-12*self._xp.sign(cos_phi)

            # print(windAngle*180/self._xp.pi, ': ', sin_phi, cos_phi) # debugging

            Delta = min(abs((W-w)/(2*cos_phi)), abs((H-h)/(2*sin_phi)))
            self.startX[k] = (W-w)/2 - Delta * cos_phi
            self.startY[k] = (H-h)/2 - Delta * sin_phi
    
    
    def _check_same_length(self, a, b):
        lenA = self._get_len(a)
        lenB = self._get_len(b)
        if lenA != lenB:
            raise ValueError(f'Vector {a} of length {lenA} is not compatible with vector {b} of length {lenB}')
    
    @staticmethod
    def _get_len(a):
        length = 1
        if isinstance(a, int):
            a = float(a)
        if not isinstance(a, float):
            length = len(a)
        return length




