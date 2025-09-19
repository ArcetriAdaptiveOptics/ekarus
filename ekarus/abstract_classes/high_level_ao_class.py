import os
import xupy as xp
np = xp.np

from ekarus.e2e.utils import my_fits_package as myfits
from ekarus.analytical.turbulence_layers import TurbulenceLayers
from ekarus.e2e.utils.read_configuration import ConfigReader
from ekarus.e2e.utils.root import resultspath, calibpath

from ekarus.e2e.utils.image_utils import get_circular_mask, reshape_on_mask
from ekarus.analytical.kl_modes import make_modal_base_from_ifs_fft


class HighLevelAO():

    def __init__(self, tn: str):
        """ The constructor """

        self.savepath = os.path.join(resultspath, tn)
        if not os.path.exists(self.savepath):
            os.mkdir(self.savepath)

        self.savecalibpath = os.path.join(calibpath, tn)
        if not os.path.exists(self.savecalibpath):
            os.mkdir(self.savecalibpath)

        self.dtype = xp.float
        self.atmo_pars = None

        self._config = ConfigReader(tn)
        
        self._read_configuration()
        self._read_loop_parameters()


    def get_photons_per_second(self, starMagnitude: float, B0: float = 1e+10) -> float:
        """
        Compute the number of photons collected per second.
        """
        if starMagnitude is None:
            starMagnitude = self.starMagnitude
        total_flux = B0 * 10**(-starMagnitude/2.5)
        collecting_area = xp.pi/4*self.pupilSizeInM**2
        collected_flux = self.throughput * total_flux * collecting_area
        return collected_flux
    

    def _read_configuration(self):
        """
        Reads the Telescope configuration file and defines the mask.
                """
        telescope_pars = self._config.read_telescope_pars()
        self.pupilSizeInM = telescope_pars['pupilSizeInM']
        self.pupilSizeInPixels = telescope_pars['pupilSizeInPixels']
        self.throughput = telescope_pars['throughput']
        mask_shape = (self.pupilSizeInPixels, self.pupilSizeInPixels)
        self.cmask = get_circular_mask(mask_shape, mask_radius=self.pupilSizeInPixels//2)


    def _read_loop_parameters(self):
        """
        Reads the loop parameters from the configuration file.
        """
        loop_pars = self._config.read_loop_pars()
        self.Nits = loop_pars['nIterations']
        self.starMagnitude = loop_pars['starMagnitude']
        self.modulationAngleInLambdaOverD = loop_pars['modulationAngleInLambdaOverD']
    

    def define_KL_modes(self, dm, oversampling:int=4, zern_modes:int=5, save_prefix:str=''):
        """
        Defines the Karhunen-Loève (KL) modes for the given DM and oversampling.
        
        Parameters
        ----------
        dm : DeformableMirror
            The deformable mirror object.
        oversampling : int
            The oversampling factor.
        zern_modes : int, optional
            The number of Zernike modes to consider, by default 5.
        save_prefix : str, optional
            Prefix for saving the KL modes files, by default ''.
            
        Returns
        -------
        KL : array
            The KL modes.
        m2c : array
            The mode-to-command matrix.
        """
        print('Obtaining the Karhunen-Loeve mirror modes ...')
        KL_path = os.path.join(self.savecalibpath,str(save_prefix)+'KLmodes.fits')
        m2c_path = os.path.join(self.savecalibpath,str(save_prefix)+'m2c.fits')
        try:
            KL = myfits.read_fits(KL_path)
            m2c = myfits.read_fits(m2c_path)
        except FileNotFoundError:
            if self.atmo_pars is None:
                self.atmo_pars = self._config.read_atmo_pars()
            r0s = self.atmo_pars['r0']
            L0 = self.atmo_pars['outerScaleInM']
            if isinstance(r0s, float):
                r0 = r0s
            else:
                r0 = xp.sqrt(xp.sum(r0s**2))
            KL, m2c, _ = make_modal_base_from_ifs_fft(1-dm.mask, self.pupilSizeInPixels, 
                self.pupilSizeInM, dm.IFF.T, r0, L0, zern_modes=zern_modes,
                oversampling=oversampling, verbose=True, xp=xp, dtype=self.dtype)
            hdr_dict = {'r0': r0, 'L0': L0, 'N_ZERN': zern_modes}
            myfits.save_fits(KL_path, KL, hdr_dict)
            myfits.save_fits(m2c_path, m2c, hdr_dict)
        return KL, m2c
    

    def compute_reconstructor(self, slope_computer, MM, lambdaInM, amps, save_prefix:str=''):
        """
        Computes the reconstructor matrix using the provided slope computer and mode matrix.
        
        Parameters
        ----------
        slope_computer : SlopeComputer
            The slope computer object.
        MM : array
            The mirror modes to calibrate/correct (e.g. KL Modes).
        lambdaInM : float | array
            The wavelength(s) at which calibration is performed.
        amps : float or array
            The amplitudes for each mode.
        save_prefix : str, optional
            Prefix for saving the reconstructor files, by default ''.

        Returns
        -------
        Rec : array
            The reconstructor matrix.
        IM : array
            The interaction matrix.
        """
        print('Computing the reconstructor ...')
        IM_path = os.path.join(self.savecalibpath,str(save_prefix)+'IM.fits')
        Rec_path = os.path.join(self.savecalibpath,str(save_prefix)+'Rec.fits')
        try:
            IM = myfits.read_fits(IM_path)
            Rec = myfits.read_fits(Rec_path)
        except FileNotFoundError:
            Nmodes = xp.shape(MM)[0]
            slopes = None
            electric_field_amp = 1-self.cmask
            lambdaOverD = lambdaInM/self.pupilSizeInM
            Nphotons = None # perfect calibration: no noise
            self.pyr.set_modulation_angle(self.modulationAngleInLambdaOverD)
            if isinstance(amps, float):
                amps *= xp.ones(Nmodes)
            for i in range(Nmodes):
                print(f'\rMode {i+1}/{Nmodes}', end='\r', flush=True)
                amp = amps[i]
                mode_phase = reshape_on_mask(MM[i,:]*amp, self.cmask)
                input_field = xp.exp(1j*mode_phase) * electric_field_amp
                push_slope = slope_computer.compute_slopes(input_field, lambdaOverD, Nphotons)/amp #self.get_slopes(input_field, Nphotons)/amp
                input_field = xp.conj(input_field)
                pull_slope = slope_computer.compute_slopes(input_field, lambdaOverD, Nphotons)/amp #self.get_slopes(input_field, Nphotons)/amp
                if slopes is None:
                    slopes = (push_slope-pull_slope)/2
                else:
                    slopes = xp.vstack((slopes,(push_slope-pull_slope)/2))
            IM = slopes.T
            U,S,Vt = xp.linalg.svd(IM, full_matrices=False)
            Rec = (Vt.T*1/S) @ U.T
            myfits.save_fits(IM_path, IM)
            myfits.save_fits(Rec_path, Rec)
        return Rec, IM

    
    def initialize_turbulence(self, N:int=None, dt:float=None):
        """
        Initializes the turbulence layers based on atmospheric parameters.

        The turbulence layers are defined in Meters.

        Parameters
        ----------
        N : int, optional
            The number of pupil lengths for the generated phase screens. 
        dt : float, optional
            Maximum time step, screen length is computed using the maximum wind speeds and simulation time.
        The minimum length for the screen is 20 pupil diameters.
        """
        print('Initializing turbulence ...')
        if self.atmo_pars is None:
            self.atmo_pars = self._config.read_atmo_pars()
        r0s = self.atmo_pars['r0']
        L0 = self.atmo_pars['outerScaleInM']
        windSpeeds = self.atmo_pars['windSpeed']
        windAngles = self.atmo_pars['windAngle']
        if N is None:
            if dt is not None:
                maxTime = dt * self.Nits
                if isinstance(windSpeeds, (int, float)):
                    maxSpeed = windSpeeds
                else:
                    maxSpeed = windSpeeds.max()
                maxLen = maxSpeed*maxTime
                N = int(np.ceil(maxLen/self.pupilSizeInM))
            else:
                N = 20
        N = int(np.max([20,N])) # set minimum N to 20
        screenPixels = N*self.pupilSizeInPixels
        screenMeters = N*self.pupilSizeInM 
        atmo_path = os.path.join(self.savepath, 'AtmospherePhaseScreens.fits')
        self.layers = TurbulenceLayers(r0s, L0, windSpeeds, windAngles, atmo_path)
        self.layers.generate_phase_screens(screenPixels, screenMeters)
        self.layers.rescale_phasescreens() # rescale in meters
        self.layers.update_mask(self.cmask)


    def get_phasescreen_at_time(self, time: float):
        """
        Retrieves the combined phase screen at a specific time.

        Parameters
        ----------
        time : float
            The time at which to retrieve the phase screen.

        Returns
        -------
        masked_phase : array
            The combined phase screen at the specified time.
        """
        masked_phases = self.layers.move_mask_on_phasescreens(time)
        masked_phase = xp.sum(masked_phases,axis=0)
        return masked_phase
    
    
    def save_telemetry_data(self, data_dict, save_prefix:str=''):
        """
        Saves the telemetry data to FITS files.
        """
        for key in data_dict:
            file_path = os.path.join(self.savepath,save_prefix+str(key)+'.fits')
            myfits.save_fits(file_path, data_dict[key])
    
            
    def load_telemetry_data(self, data_keys:list[str]=None, save_prefix:str=''):
        """
        Load telemetry data from FITS files.
        """
        if data_keys is None:
            data_keys = self.telemetry_keys
        loaded_data = []
        for key in data_keys:
            file_path = os.path.join(self.savepath, save_prefix+key+'.fits')
            loaded_data.append(myfits.read_fits(file_path))
        return loaded_data
            








        