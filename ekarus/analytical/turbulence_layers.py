import xupy as xp
# import numpy as np

from arte.atmo.phase_screen_generator import PhaseScreenGenerator
from ekarus.e2e.utils.image_utils import reshape_on_mask


class TurbulenceLayers():

    def __init__(self, r0s, L0, windSpeeds, windAngles, savepath:str):
        """ The constructor """

        self.r0s = r0s
        self.L0 = L0

        # Wrap angles to [-pi,pi)
        angles = ( windAngles + xp.pi) % (2 * xp.pi ) - xp.pi
        angles += 2*xp.pi * (angles < 0)
        self.windAngles = angles
        self.windSpeeds = windSpeeds

        self.dtype = xp.float

        self._check_same_length(r0s, windSpeeds)
        self._check_same_length(r0s, windAngles)

        self.nLayers = self._get_len(r0s)
        self.savepath = savepath


    def update_mask(self, mask):
        """ Update the mask and set the start coordinates on the phase screens """
        self.mask = mask
        self._define_start_coordinates_on_phasescreens(mask.shape)


    def generate_phase_screens(self, screenSizeInPixels, screenSizeInMeters):
        """ Generate or load phase screens 
        
        Parameters
        ----------
        screenSizeInPixels : int
            Size of the phase screen in pixels (assumed square)
        screenSizeInMeters : float 
            Size of the phase screen in meters (assumed square)
        """  
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
        
        self.phase_screens = xp.asarray(self._phs._phaseScreens, dtype=self.dtype)
        self._normalization_factors = (1/self.pixelsPerMeter / self.r0s) ** (5. / 6)
        self.screen_shape = self._phs._phaseScreens.shape[1:]


    def rescale_phasescreens(self):
        """ Rescale the phase screens to meters """
        if self.nLayers > 1:
            for k in range(self.nLayers):
                self.phase_screens[k,:,:] *= 500e-9 / (2*xp.pi) * self._normalization_factors[k]
        else:
            self.phase_screens[0,:,:] *= 500e-9 / (2*xp.pi) * self._normalization_factors
    

    def rescale_phasescreens_in_radians(self, lambdaInM):
        """ 
        Rescale the phase screens to radians at a given wavelength.

        Parameters
        ----------
        lambdaInM : float
            Wavelength in meters
        """
        if self.nLayers > 1:
            for k in range(self.nLayers):
                self.phase_screens[k,:,:] *= 500e-9 / lambdaInM * self._normalization_factors[k]
        else:
            self.phase_screens[0,:,:] *= 500e-9 / lambdaInM * self._normalization_factors
        
        
    def move_mask_on_phasescreens(self, dt):
        """ 
        Move the mask on the phase screens according
        to the wind speeds and angles.

        Parameters
        ----------
        dt : float
            Simulation elapsed time in seconds
        
        Returns
        -------
        masked_phases : 3D xp.array
            Masked phases for each layer (nLayers, H, W)
        """

        masked_phases = xp.zeros([self.nLayers,self.mask_shape[0],self.mask_shape[1]])
        if self.nLayers > 1:
            for k in range(self.nLayers):
                masked_phases[k,:,:] = self._get_single_masked_phase(dt, self.phase_screens[k], self.windSpeeds[k], \
                                                                self.windAngles[k], self.startX[k], self.startY[k])
        else:
            masked_phases[0,:,:] = self._get_single_masked_phase(dt, self.phase_screens[0], self.windSpeeds, \
                                                                self.windAngles, self.startX, self.startY)

        return masked_phases
    

    def _get_single_masked_phase(self, dt, screen, windSpeed, windAngle, xStart, yStart):
        """ 
        Get the masked phase for a single phase screen

        Parameters
        ----------
        dt : float
            Simulation elapsed time in seconds
        screen : 2D xp.array
            Phase screen
        windSpeed : float
            Wind speed in m/s
        windAngle : float
            Wind angle in radians
        xStart : float
            Starting x coordinate of the mask on the phase screen
        yStart : float
            Starting y coordinate of the mask on the phase screen

        Returns
        -------
        phase_mask : 2D xp.array
            Masked phase
        """
        screen_shape = self.screen_shape
        mask_shape = self.mask.shape

        dpix = windSpeed*dt*self.pixelsPerMeter
        x = xStart + dpix*xp.cos(windAngle)
        y = yStart + dpix*xp.sin(windAngle)

        if x > screen_shape[1]-mask_shape[1] or y > screen_shape[0]-mask_shape[0] or x < 0 or y < 0:
            raise ValueError(f'A displacement of {dpix:1.2f} for a time {dt:1.3f} [s] with wind {windSpeed:1.1f} [m/s] yields\
                            ({y:1.0f},{x:1.0f}), which is outside the bounds for a {screen_shape} screen and a {mask_shape} mask.')

        x_round = int(xp.floor(x) * (x>=xStart) + xp.ceil(x) * (x<xStart))
        y_round = int(xp.floor(y) * (y>=yStart) + xp.ceil(y) * (y<yStart))

        dx, dy = abs(x-x_round), abs(y-y_round)
        sdx, sdy = int(xp.sign(x-x_round)), int(xp.sign(y-y_round)) # sdx, sdy = int(xp.sign(x-xStart)), int(xp.sign(y-yStart))

        H,W = mask_shape

        full_mask = xp.ones_like(screen, dtype=bool)
        full_mask[y_round:(y_round+H),x_round:(x_round+W)] = self.mask.copy()
        phase = reshape_on_mask(screen[~full_mask], self.mask)

        thr = 1e-8

        if dx > thr and dy > thr:
            dx_phase = reshape_on_mask(screen[~xp.roll(full_mask,sdx,axis=1)], self.mask)
            dy_phase = reshape_on_mask(screen[~xp.roll(full_mask,sdy,axis=0)], self.mask)
            dxdy_phase = reshape_on_mask(screen[~xp.roll(full_mask,(sdy,sdx),axis=(0,1))], self.mask)
            phase_mask = (phase * (1-dx) + dx * dx_phase) * (1-dy) + (dy_phase * (1-dx) + dx * dxdy_phase) * dy

        elif dx > thr:
            dx_phase = reshape_on_mask(screen[~xp.roll(full_mask,sdx,axis=1)], self.mask)
            phase_mask = phase * (1-dx) + dx_phase * dx

        elif dy > thr:
            dy_phase = reshape_on_mask(screen[~xp.roll(full_mask,sdy,axis=0)], self.mask)
            phase_mask = phase * (1-dy) + dy_phase * dy

        else:
            phase_mask = phase.copy()
        
        return phase_mask
    

    def _define_start_coordinates_on_phasescreens(self, mask_shape):
        """
        Define the starting coordinates of the mask on the phase screens
        so that the mask can travel the maximum distance according to
        wind angles without going out of bounds.

        Parameters
        ----------
        mask_shape : tuple
            Shape of the mask (H, W)
        """
        self.mask_shape = mask_shape

        H,W = self.screen_shape
        h,w = self.mask_shape

        self.startX = xp.zeros(self.nLayers)
        self.startY = xp.zeros(self.nLayers)

        for k in range(self.nLayers):

            if self.nLayers > 1:
                windAngle = self.windAngles[k]
            else: 
                windAngle = self.windAngles

            sin_phi = xp.sin(windAngle) 
            cos_phi = xp.cos(windAngle) 

            # Avoid division by 0
            if abs(sin_phi) < 1e-10:
                sin_phi = 1e-10*xp.sign(sin_phi)
            if abs(cos_phi) < 1e-10:
                cos_phi = 1e-10*xp.sign(cos_phi)

            Delta = min(abs((W-w)/(2*cos_phi)), abs((H-h)/(2*sin_phi)))
            self.startX[k] = max(0.0,(W-w)/2 - Delta * cos_phi)
            self.startY[k] = max(0.0,(H-h)/2 - Delta * sin_phi)
    
    
    def _check_same_length(self, a, b):
        lenA = self._get_len(a)
        lenB = self._get_len(b)
        if lenA != lenB:
            raise ValueError(f'Vector {a} of length {lenA} is not \
                              compatible with vector {b} of length {lenB}')
    
    @staticmethod
    def _get_len(a):
        length = 1
        if isinstance(a, int):
            a = float(a)
        if not isinstance(a, float):
            length = len(a)
        return length




