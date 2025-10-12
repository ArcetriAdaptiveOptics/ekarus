import os
import xupy as xp
import numpy as np

from ekarus.e2e.utils import my_fits_package as myfits
from ekarus.analytical.turbulence_layers import TurbulenceLayers
from ekarus.e2e.utils.read_configuration import ConfigReader
from ekarus.e2e.utils.root import resultspath, calibpath, atmopath

from ekarus.e2e.utils.image_utils import get_circular_mask, reshape_on_mask, remap_on_new_mask
from ekarus.analytical.kl_modes import make_modal_base_from_ifs_fft
from arte.utils.radial_profile import computeRadialProfile

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
        self._tn = tn
        
        self._read_configuration()
        self._read_loop_parameters()


    def get_photons_per_second(self, starMagnitude: float, B0: float = 1e+10) -> float:
        """ Compute the number of photons collected per second. """
        if starMagnitude is None:
            starMagnitude = self.starMagnitude
        total_flux = B0 * 10**(-starMagnitude/2.5)
        collecting_area = xp.pi/4*(self.pupilSizeInM**2-self.centerObscurationInM**2)
        collected_flux = self.throughput * total_flux * collecting_area
        return collected_flux
    

    def _read_configuration(self):
        """ Reads the Telescope configuration file, defining the mask """
        telescope_pars = self._config.read_telescope_pars()
        self.pupilSizeInM = telescope_pars['pupilSizeInM']
        self.pupilSizeInPixels = telescope_pars['pupilSizeInPixels']
        self.throughput = telescope_pars['throughput']
        mask_shape = (self.pupilSizeInPixels, self.pupilSizeInPixels)
        self.cmask = get_circular_mask(mask_shape, mask_radius=self.pupilSizeInPixels/2)
        try:
            self.centerObscurationInM = telescope_pars['centerObscuration']
            obscSizeInPixels = self.pupilSizeInPixels*self.centerObscurationInM/self.pupilSizeInM
            obs_mask = get_circular_mask(mask_shape, mask_radius=obscSizeInPixels/2)
            self.cmask = (self.cmask + (1-obs_mask)).astype(bool)
        except KeyError:
            self.centerObscurationInM = 0.0



    def _read_loop_parameters(self):
        """
        Reads the loop parameters from the configuration file.
        """
        loop_pars = self._config.read_loop_pars()
        self.Nits = loop_pars['nIterations']
        self.starMagnitude = loop_pars['starMagnitude']
        self.dt = 1/loop_pars['simFreqHz']
        try:
            self.recompute = loop_pars['recompute']
        except KeyError:
            self.recompute = False
    

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
        KL_path = os.path.join(self.savecalibpath,str(save_prefix)+'KLmodes.fits')
        m2c_path = os.path.join(self.savecalibpath,str(save_prefix)+'m2c.fits')
        try:
            if self.recompute is True:
                raise FileNotFoundError('Recompute is True')
            KL = myfits.read_fits(KL_path)
            m2c = myfits.read_fits(m2c_path)
        except FileNotFoundError:
            if self.atmo_pars is None:
                self.atmo_pars = self._config.read_atmo_pars()
            r0s = self.atmo_pars['r0']
            L0 = self.atmo_pars['outerScaleInM']
            r0 = (1/xp.sum(r0s**(-5/3)))**(3/5)
            IFFs = dm.IFF.copy()
            if dm.slaving is not None: # slaving
                IFFs = remap_on_new_mask(dm.IFF, dm.mask, dm.pupil_mask)
                IFFs = IFFs[:,dm.master_ids]
                print(f'SLAVING: downsized IFFs from {dm.IFF.shape} to {IFFs.shape}')
            KL, m2c, _ = make_modal_base_from_ifs_fft(1-self.cmask, self.pupilSizeInPixels, 
                self.pupilSizeInM, IFFs.T, r0, L0, zern_modes=zern_modes,
                oversampling=oversampling, verbose=True, xp=xp, dtype=self.dtype)            
            # KL, m2c, _ = make_modal_base_from_ifs_fft(1-dm.mask, self.pupilSizeInPixels, 
            #     self.pupilSizeInM, dm.IFF.T, r0, L0, zern_modes=zern_modes,
            #     oversampling=oversampling, verbose=True, xp=xp, dtype=self.dtype)
            # if dm.slaving is not None: # slaving
            #     old_shape = KL.shape
            #     KL = remap_on_new_mask(KL, dm.mask, dm.pupil_mask)
            #     print(f'SLAVING: downsized KL from {old_shape} to {KL.shape}')
            hdr_dict = {'r0': r0, 'L0': L0, 'N_ZERN': zern_modes}
            myfits.save_fits(m2c_path, m2c, hdr_dict)
            myfits.save_fits(KL_path, KL, hdr_dict)
        return KL, m2c
    

    def compute_reconstructor(self, slope_computer, MM, lambdaInM, amps, use_diagonal:bool=False, save_prefix:str=''):
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
        IM_path = os.path.join(self.savecalibpath,str(save_prefix)+'IM.fits')
        Rec_path = os.path.join(self.savecalibpath,str(save_prefix)+'Rec.fits')
        try:
            if self.recompute is True:
                raise FileNotFoundError('Recompute is True')
            IM = myfits.read_fits(IM_path)
            Rec = myfits.read_fits(Rec_path)
        except FileNotFoundError:
            Nmodes = min(slope_computer.nModes,xp.shape(MM)[0])
            slopes = None
            electric_field_amp = 1-self.cmask
            lambdaOverD = lambdaInM/self.pupilSizeInM
            Nphotons = None # perfect calibration: no noise
            if isinstance(amps, float):
                amps *= xp.ones(Nmodes)
            for i in range(Nmodes):
                print(f'\rReconstructing mode {i+1}/{Nmodes}', end='\r', flush=True)
                amp = amps[i]
                mode_phase = reshape_on_mask(MM[i,:]*amp, self.cmask)
                # mode_phase = reshape_on_mask(MM[i,:]*amp, self.dm.mask)
                # if self.dm.slaving is not None:
                #     dm_command = self.dm.R[:,self.dm.visible_pix_ids] @ MM[i,:]*amp
                #     mirror_cmd = self.dm.slaving @ dm_command[self.dm.master_ids]
                # else:
                #     mirror_cmd = self.dm.R @ MM[i,:]*amp
                # mode_phase = reshape_on_mask(self.dm.IFF @ mirror_cmd, self.dm.mask)
                input_field = xp.exp(1j*mode_phase) * electric_field_amp
                push_slope = slope_computer.compute_slopes(input_field, lambdaOverD, Nphotons, use_diagonal=use_diagonal)/amp #self.get_slopes(input_field, Nphotons)/amp
                pull_slope = slope_computer.compute_slopes(xp.conj(input_field), lambdaOverD, Nphotons, use_diagonal=use_diagonal)/amp #self.get_slopes(input_field, Nphotons)/amp
                if slopes is None:
                    slopes = (push_slope-pull_slope)/2
                else:
                    slopes = xp.vstack((slopes,(push_slope-pull_slope)/2))
            IM = slopes.T
            U,S,Vt = xp.linalg.svd(IM, full_matrices=False)
            Rec = xp.array((Vt.T*1/S) @ U.T,dtype=self.dtype)
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
        if self.atmo_pars is None:
            self.atmo_pars = self._config.read_atmo_pars()
        r0s = self.atmo_pars['r0']
        L0 = self.atmo_pars['outerScaleInM']
        windSpeeds = self.atmo_pars['windSpeed']
        windAngles = self.atmo_pars['windAngle']
        print(f'Fried parameter is: {(1/xp.sum(r0s**(-5/3)))**(3/5)*1e+2:1.1f} [cm]')
        try:
            recompute_atmo_screens = self.atmo_pars['recompute']
        except KeyError:
            recompute_atmo_screens = False
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
        atmo_dir = os.path.join(atmopath,self._tn)
        if not os.path.exists(atmo_dir):   
            os.mkdir(atmo_dir)
        atmo_path = os.path.join(atmo_dir,'atmospheric_phase_layers.fits')
        self.layers = TurbulenceLayers(r0s, L0, windSpeeds, windAngles, atmo_path)
        self.layers.generate_phase_screens(screenPixels, screenMeters, recompute_atmo_screens)
        self.layers.rescale_phasescreens() # rescale in meters
        self.layers.update_mask(self.cmask)


    def _initialize_pyr_slope_computer(self, pyr_id:str, detector_id:str, slope_computer_id:str):
        """ 
        Initialize devices for PyrWFS slope computation
        """

        from ekarus.e2e.devices.pyramid_wfs import PyramidWFS
        from ekarus.e2e.devices.detector import Detector
        from ekarus.e2e.devices.slope_computer import SlopeComputer

        det_pars = self._config.read_detector_pars(detector_id)
        detector_shape=det_pars["detector_shape"]
        det = Detector(
            detector_shape=detector_shape,
            RON=det_pars["RON"],
            quantum_efficiency=det_pars["quantum_efficiency"],
            beam_split_ratio=det_pars["beam_splitter_ratio"],
        )

        wfs_pars = self._config.read_sensor_pars(pyr_id) 
        subapPixSep = wfs_pars["subapPixSep"]
        oversampling = wfs_pars["oversampling"]
        sensorLambda = wfs_pars["lambdaInM"]
        sensorBandwidth = wfs_pars['bandWidthInM']
        subapertureSize = wfs_pars["subapPixSize"]
        rebin = oversampling*self.pupilSizeInPixels/max(detector_shape)
        apex_angle = 2*xp.pi*sensorLambda/self.pupilSizeInM*(xp.floor(subapertureSize+1.0)+subapPixSep)*rebin/2
        pyr= PyramidWFS(
            apex_angle=apex_angle, 
            oversampling=oversampling, 
            sensorLambda=sensorLambda,
            sensorBandwidth=sensorBandwidth
        )

        # Size checks
        if xp.floor(subapertureSize+1.0)*2+subapPixSep > min(detector_shape):
            raise ValueError(f'Subapertures of size {xp.floor(subapertureSize+1.0):1.0f}  separated by {subapPixSep:1.0f} cannot fit on a  {detector_shape[0]:1.0f}x{detector_shape[1]:1.0f} detector')
        
        if (xp.floor(subapertureSize+1.0)+subapPixSep)*rebin/2 <= self.pupilSizeInPixels//2:
            raise ValueError(f'Pupil center in the subquadrant is {(xp.floor(subapertureSize+1.0)+subapPixSep)*rebin/2:1.0f} meaning pupils of size {self.pupilSizeInPixels:1.0f} would overlap')

        sc_pars = self._config.read_slope_computer_pars(slope_computer_id)
        sc = SlopeComputer(pyr, det, sc_pars)
        sc.calibrate_sensor(self._tn, prefix_str=pyr_id+'_', 
                        recompute=self.recompute,
                        piston=1-self.cmask, 
                        lambdaOverD = sensorLambda/self.pupilSizeInM,
                        Npix = subapertureSize,
                        centerObscurationInPixels = 
                        # self.pupilSizeInPixels*self.centerObscurationInM/self.pupilSizeInM
                        xp.floor(subapertureSize+1.0)*self.centerObscurationInM/self.pupilSizeInM
        ) 
        
        return pyr, det, sc


    
    def perform_loop_iteration(self, phase, dm_cmd, slope_computer, use_diagonal:bool=False, starMagnitude:float=None, slaving=None):
        """
        Performs a single iteration of the AO loop.
        Parameters
        ----------
        phase : array
            The input phase screen in meters.
        dm_cmd : array
            The current DM command in meters.
        slope_computer : SlopeComputer
            The slope computer object.
        starMagnitude : float
            The magnitude of the star being observed.
        
        Returns
        -------
        dm_cmd : xp.array
            The DM command in meters.
        modes : xp.array
            The reconstructed modes in meters.
        """
        lambda_ref = slope_computer._wfs.lambdaInM
        m2rad = 2*xp.pi/lambda_ref
        lambdaOverD = lambda_ref/self.pupilSizeInM
        Nphotons = self.get_photons_per_second(starMagnitude) * slope_computer.dt

        delta_phase_in_rad = reshape_on_mask(phase * m2rad, self.cmask)
        input_field = (1-self.cmask) * xp.exp(1j * delta_phase_in_rad)
        slopes = slope_computer.compute_slopes(input_field, lambdaOverD, Nphotons, use_diagonal=use_diagonal)
        
        modes = slope_computer.Rec @ slopes
        # modes *= slope_computer.modal_gains
        cmd = slope_computer.m2c @ modes

        cmd /= m2rad # convert to meters
        modes /= m2rad  # convert to meters

        if slaving is not None:
            cmd = slaving @ cmd
        dm_cmd += cmd * slope_computer.intGain

        return dm_cmd, modes


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
        self.save_prefix = save_prefix
        for key in data_dict:
            file_path = os.path.join(self.savepath,save_prefix+str(key)+'.fits')
            # print(type(data_dict[key]), xp.shape(data_dict[key].data), xp.shape(data_dict[key][0].mask))
            myfits.save_fits(file_path, data_dict[key])
    
            
    def load_telemetry_data(self, data_keys:list[str]=None, save_prefix:str=''):
        """
        Load telemetry data from FITS files.
        """
        if data_keys is None:
            data_keys = self.telemetry_keys
        loaded_data = []
        for key in data_keys:
            try:
                file_path = os.path.join(self.savepath, save_prefix+key+'.fits')
                loaded_data.append(myfits.read_fits(file_path))
            except FileNotFoundError:
                new_path = os.path.join(self.savepath, key+'.fits')
                print(f'File {file_path} not found, trying {new_path} instead')
                loaded_data.append(myfits.read_fits(new_path))
        return loaded_data
    
    
    def get_contrast(self, residual_phase_in_rad, oversampling:int=12):
        res_phase = xp.asarray(residual_phase_in_rad)
        padding_len = int(self.cmask.shape[0]*(oversampling-1)/2)
        pup_mask = xp.pad(self.cmask, padding_len, mode='constant', constant_values=1)
        phase_2d = reshape_on_mask(res_phase, pup_mask)
        phase_var = reshape_on_mask((res_phase-xp.mean(res_phase))**2, pup_mask)
        perfect_coro_field = (1-pup_mask) * (xp.sqrt(xp.exp(-phase_var))-xp.exp(1j*phase_2d))
        field_on_focal_plane = xp.fft.fftshift(xp.fft.fft2(perfect_coro_field))
        psf = abs(field_on_focal_plane)**2
        perfect_psf = abs(xp.fft.fftshift(xp.fft.fft2((1-pup_mask))))**2
        psd,dist = computeRadialProfile(xp.asnumpy(psf/xp.max(perfect_psf)),psf.shape[0]/2,psf.shape[1]/2)
        pix_dist = dist/oversampling
        return xp.array(psf), xp.array(psd), xp.array(pix_dist)

    
    def _psf_from_frame(self, frame, lambdaInM, oversampling:int=8):
        """
        Computes the PSF from a given frame and mask.

        Parameters
        ----------
        frame : array
            The input frame.
        lambdaInM : float
            The wavelength in meters.
        oversampling : int, optional
            The oversampling factor, by default 8.

        Returns
        -------
        psf : array
            The computed PSF.
        """
        padding_len = int(self.cmask.shape[0]*(oversampling-1)/2)
        psf_mask = xp.pad(self.cmask, padding_len, mode='constant', constant_values=1)
        electric_field_amp = 1-psf_mask
        input_phase = reshape_on_mask(frame[~self.cmask], psf_mask)
        electric_field = electric_field_amp * xp.exp(-1j*xp.asarray(input_phase*(2*xp.pi)/lambdaInM))
        field_on_focal_plane = xp.fft.fftshift(xp.fft.fft2(electric_field))
        psf = abs(field_on_focal_plane)**2
        pixelSize = 1/oversampling
        return psf, pixelSize
    
    @staticmethod
    def phase_rms(vec):
        return xp.sqrt(xp.sum(vec**2)/len(vec))

            








        