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
        
        self._define_pupil_mask()
        self._read_loop_parameters()


    def get_photons_per_second(self, starMagnitude: float, B0: float = 1e+10) -> float:
        """ Compute the number of photons collected per second. """
        if starMagnitude is None:
            starMagnitude = self.starMagnitude
        total_flux = B0 * 10**(-starMagnitude/2.5)
        collecting_area = xp.pi/4*(self.pupilSizeInM**2-self.centerObscurationInM**2)
        collected_flux = self.throughput * total_flux * collecting_area
        return collected_flux
    

    def _define_pupil_mask(self):
        """ Reads the Telescope configuration file, defining the mask """
        telescope_pars = self._config.read_telescope_pars()
        self.pupilSizeInM = telescope_pars['pupilSizeInM']
        self.pupilSizeInPixels = telescope_pars['pupilSizeInPixels']
        self.pixelScale = self.pupilSizeInPixels/self.pupilSizeInM
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
        try:
            spiderWidth = telescope_pars['spiders']['widthInM']
            spiderAngles = telescope_pars['spiders']['angles']
            spiderPixWidth = spiderWidth * self.pixelScale
            self._add_telescope_spiders(spiderPixWidth, spiderAngles)
        except KeyError:
            pass
    

    def define_KL_modes(self, dm, oversampling:int=4, zern_modes:int=2, filt_modes=None, save_prefix:str=''):
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
                self.pupilSizeInM, IFFs.T, r0, L0, zern_modes=zern_modes, filt_modes = filt_modes,
                oversampling=oversampling, if_max_condition_number=100, verbose=True, xp=xp, dtype=self.dtype)         
            hdr_dict = {'r0': r0, 'L0': L0, 'N_ZERN': zern_modes}
            myfits.save_fits(m2c_path, m2c, hdr_dict)
            myfits.save_fits(KL_path, KL, hdr_dict)
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
        IM_path = os.path.join(self.savecalibpath,str(save_prefix)+'IM.fits')
        Rec_path = os.path.join(self.savecalibpath,str(save_prefix)+'Rec.fits')
        try:
            if self.recompute is True:
                raise FileNotFoundError('Recompute is True')
            IM = myfits.read_fits(IM_path)
            Rec = myfits.read_fits(Rec_path)
        except FileNotFoundError:
            slopes = self._get_slopes(slope_computer, MM, lambdaInM, amps)
            IM = slopes.T
            U,S,Vt = xp.linalg.svd(IM, full_matrices=False)
            Rec = xp.array((Vt.T*1/S) @ U.T,dtype=self.dtype)
            myfits.save_fits(IM_path, IM)
            myfits.save_fits(Rec_path, Rec)
        return Rec, IM

    
    def initialize_turbulence(self, tn:str=None, N:int=None, dt:float=None):
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
        r0s = self.atmo_pars['r0']
        L0 = self.atmo_pars['outerScaleInM']
        windSpeeds = self.atmo_pars['windSpeed']
        windAngles = self.atmo_pars['windAngle']
        r0 = (1/xp.sum(r0s**(-5/3)))**(3/5)
        seeing = 0.98*500e-9/r0
        print(f'Fried parameter is: {r0*1e+2:1.1f} [cm] (seeing = {seeing*180/np.pi*3600:1.2f}")')
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
        if tn is not None:
            self.atmo_tn = tn
        else:
            self.atmo_tn = self._tn
        atmo_dir = os.path.join(atmopath,self.atmo_tn)
        if not os.path.exists(atmo_dir):   
            os.mkdir(atmo_dir)
        atmo_path = os.path.join(atmo_dir,'atmospheric_phase_layers.fits')
        self.layers = TurbulenceLayers(r0s, L0, windSpeeds, windAngles, atmo_path)
        self.layers.generate_phase_screens(screenPixels, screenMeters, recompute_atmo_screens)
        self.layers.rescale_phasescreens() # rescale in meters
        self.layers.update_mask(self.cmask)

    
    
    def perform_loop_iteration(self, phase, slope_computer, starMagnitude:float=None, slaving=None):
        """
        Performs a single iteration of the AO loop.
        Parameters
        ----------
        phase : array
            The input phase screen in meters.
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

        slopes = slope_computer.compute_slopes(input_field, lambdaOverD, Nphotons)
        modes = slope_computer.Rec @ slopes
        modes *= slope_computer.modalGains
        cmd = slope_computer.m2c @ modes
        cmd /= m2rad # convert to meters
        modes /= m2rad  # convert to meters

        if slaving is not None:
            cmd = slaving @ cmd

        return cmd, modes


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
    
    
    def get_contrast(self, residual_phases_in_rad, oversampling:int=10):
        """
        Computes the PSF and contrast from the residual phase
        using the formula for a perfect idealized coronograph.
        """        
        N = residual_phases_in_rad.shape[0]
        res_phases = xp.array(residual_phases_in_rad)
        padding_len = int(self.cmask.shape[0]*(oversampling-1)/2)
        pup_mask = xp.pad(self.cmask, padding_len, mode='constant', constant_values=1)
        psf_stack = []
        # psf_stack = xp.zeros([N,self.cmask.shape[0]*oversampling,self.cmask.shape[1]*oversampling])
        field_amp = 1-pup_mask
        for k,res_phase in enumerate(res_phases):
            print(f'\rComputing contrast: processing frame {k+1:1.0f}/{N:1.0f}',end='\r',flush=True)
            phase_2d = reshape_on_mask(res_phase, pup_mask)
            # phase_var = xp.sum((res_phase-xp.mean(res_phase))**2) 
            phase_var = reshape_on_mask((res_phase-xp.mean(res_phase))**2, pup_mask)
            perfect_coro_field = field_amp * (xp.sqrt(xp.exp(-phase_var))-xp.exp(1j*phase_2d))
            coro_focal_plane_ef = xp.fft.fftshift(xp.fft.fft2(perfect_coro_field))
            coro_psf = abs(coro_focal_plane_ef)**2
            input_field = field_amp * xp.exp(1j*phase_2d)
            psf = abs(xp.fft.fftshift(xp.fft.fft2(input_field)))**2
            coro_psf /= xp.max(psf)
            # psf_stack[k] = coro_psf
            psf_stack.append(coro_psf)
            # psf_rms += coro_psf**2
        # psf_rms = xp.sqrt(psf_rms/N)
        psf_stack = xp.array(psf_stack)
        psf_rms = xp.std(psf_stack,axis=0)
        rad_profile,dist = computeRadialProfile(xp.asnumpy(psf_rms),psf_rms.shape[0]/2,psf_rms.shape[1]/2)
        pix_dist = dist/oversampling
        return xp.array(psf_rms), xp.array(rad_profile), xp.array(pix_dist)

    
    def _read_loop_parameters(self):
        """
        Reads the loop parameters from the configuration file.
        """
        self.atmo_pars = self._config.read_atmo_pars()
        loop_pars = self._config.read_loop_pars()
        self.Nits = loop_pars['nIterations']
        self.starMagnitude = loop_pars['starMagnitude']
        self.dt = 1/loop_pars['simFreqHz']
        try:
            self.recompute = loop_pars['recompute']
        except KeyError:
            self.recompute = False


    def _initialize_pyr_slope_computer(self, pyr_id:str, detector_id:str, slope_computer_id:str):
        """ 
        Initialize devices for PyrWFS slope computation
        """

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
        oversampling = wfs_pars["oversampling"]
        sensorLambda = wfs_pars["lambdaInM"]
        sensorBandwidth = wfs_pars['bandWidthInM']
        subapPixSep = wfs_pars["subapPixSep"]
        subapertureSize = wfs_pars["subapPixSize"]
        rebin = oversampling*self.pupilSizeInPixels/max(detector_shape)
        type = wfs_pars['type']
        match type: # TODO Move to PWFS class
            case '4PWFS':
                from ekarus.e2e.devices.pyramid_wfs import PyramidWFS
                apex_angle = 2*xp.pi*sensorLambda/self.pupilSizeInM*(xp.floor(subapertureSize+1.0)+subapPixSep)*rebin/2
                pyr= PyramidWFS(
                    apex_angle=apex_angle, 
                    oversampling=oversampling, 
                    sensorLambda=sensorLambda,
                    sensorBandwidth=sensorBandwidth
                )
            case '3PWFS':
                from ekarus.e2e.devices.pyr3_wfs import Pyr3WFS
                vertex_angle = 2*xp.pi*sensorLambda/self.pupilSizeInM*((xp.floor(subapertureSize+1.0)+subapPixSep)/(2*xp.cos(xp.pi/6)))*rebin/2
                pyr= Pyr3WFS(
                    vertex_angle=vertex_angle, 
                    oversampling=oversampling, 
                    sensorLambda=sensorLambda,
                    sensorBandwidth=sensorBandwidth
                )
            case _:
                raise KeyError('Unrecognized WFS type: Available types are: 4PWFS, 3PWFS')

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
    

    def calibrate_optical_gains_from_precorrected_screens(self, pre_corrected_screens, slope_computer, MM, mode_offset =None,
                                amps:float=0.02, save_prefix:str=''):
        """
        Calibrates the optical gains for each mode using a phase screen at a given time.
        
        Parameters
        ----------
        pre_corrected_screens : cube (Nframes,Npix,Npix)
            The pre-corrected phase screens to use for the calibration.
        slope_computer : SlopeComputer
            The slope computer object.
        MM : array
            The mirror modes to calibrate/correct (e.g. KL Modes).
        lambdaInM : float | array
            The wavelength(s) at which calibration is performed.
        amps : float or array
            The amplitudes for each mode.
        phase_offset : array, optional
            Phase offset to be added to each mode, by default None.
        
        Returns 
        -------
        opt_gains : array
            The computed optical gains.
        """
        og_cl_path = os.path.join(self.savecalibpath,str(save_prefix)+'closed_loop_OG.fits')
        og_pl_path = os.path.join(self.savecalibpath,str(save_prefix)+'perfect_loop_OG.fits')
        try:
            if self.recompute is True:
                raise FileNotFoundError('Recompute is True')
            cl_opt_gains = myfits.read_fits(og_cl_path)
            pl_opt_gains = myfits.read_fits(og_pl_path)
        except FileNotFoundError:
            IM = myfits.read_fits(os.path.join(self.savecalibpath,str(save_prefix)+'IM.fits'))
            Nmodes = slope_computer.nModes
            cl_opt_gains = xp.zeros(Nmodes)
            pl_opt_gains = xp.zeros(Nmodes)
            phase2modes = xp.linalg.pinv(MM.T) 
            field_amp = 1-self.cmask
            lambdaOverD = self.pyr.lambdaInM/self.pupilSizeInM
            N = xp.shape(pre_corrected_screens)[0]
            for i in range(N):
                print(f'\rPhase realization {i+1}/{N}', end='\r', flush=True)
                phi = pre_corrected_screens[int(i),:]
                phi -= xp.mean(phi)
                atmo_phase = reshape_on_mask(phi, self.cmask)
                phi_atmo = phi*2*xp.pi/self.pyr.lambdaInM
                input_field = field_amp * xp.exp(1j*atmo_phase*2*xp.pi/self.pyr.lambdaInM)
                slopes = slope_computer.compute_slopes(input_field, lambdaOverD, None)
                rec_modes = slope_computer.Rec @ slopes
                rec_phi = MM[:slope_computer.nModes,:].T @ rec_modes
                res_phi = phi_atmo - rec_phi
                phi_modes = phase2modes @ phi_atmo
                lo_phi = MM.T @ phi_modes
                ho_phi = phi_atmo - lo_phi
                if mode_offset is not None:
                    phi_atmo += mode_offset
                    res_phi += mode_offset
                    ho_phi += mode_offset
                cl_slopes = self._get_slopes(slope_computer, MM, self.pyr.lambdaInM, amps, phase_offset=res_phi)
                pl_slopes = self._get_slopes(slope_computer, MM, self.pyr.lambdaInM, amps, phase_offset=ho_phi)
                cl_gains = xp.zeros(Nmodes)
                pl_gains = xp.zeros(Nmodes)
                for i in range(Nmodes):
                    calib_slope = IM[:,i]
                    norm = xp.dot(calib_slope,calib_slope)
                    cl_gains[i] = xp.dot(cl_slopes[i,:],calib_slope)/norm
                    pl_gains[i] = xp.dot(pl_slopes[i,:],calib_slope)/norm
                cl_opt_gains += cl_gains/N
                pl_opt_gains += pl_gains/N
            myfits.save_fits(og_cl_path,cl_opt_gains)
            myfits.save_fits(og_pl_path,pl_opt_gains)
        return cl_opt_gains, pl_opt_gains #ol_opt_gains, 
    

    def calibrate_optical_gains(self, N:int, slope_computer, MM, mode_offset =None,
                                amps:float=0.02, save_prefix:str=''):
        """
        Calibrates the optical gains for each mode using a phase screen at a given time.
        
        Parameters
        ----------
        N : int
            The number of random phase screen realizations to use.
        slope_computer : SlopeComputer
            The slope computer object.
        MM : array
            The mirror modes to calibrate/correct (e.g. KL Modes).
        lambdaInM : float | array
            The wavelength(s) at which calibration is performed.
        amps : float or array
            The amplitudes for each mode.
        phase_offset : array, optional
            Phase offset to be added to each mode, by default None.
        
        Returns 
        -------
        opt_gains : array
            The computed optical gains.
        """
        og_cl_path = os.path.join(self.savecalibpath,str(save_prefix)+'closed_lopp_OG.fits')
        # og_ol_path = os.path.join(self.savecalibpath,str(save_prefix)+'open_loop_OG.fits')
        og_pl_path = os.path.join(self.savecalibpath,str(save_prefix)+'perfect_loop_OG.fits')
        try:
            if self.recompute is True:
                raise FileNotFoundError('Recompute is True')
            cl_opt_gains = myfits.read_fits(og_cl_path)
            # ol_opt_gains = myfits.read_fits(og_ol_path)
            pl_opt_gains = myfits.read_fits(og_pl_path)
        except FileNotFoundError:
            IM = myfits.read_fits(os.path.join(self.savecalibpath,str(save_prefix)+'IM.fits'))
            Nmodes = slope_computer.nModes
            cl_opt_gains = xp.zeros(Nmodes)
            # ol_opt_gains = xp.zeros(Nmodes)
            pl_opt_gains = xp.zeros(Nmodes)
            phase2modes = xp.linalg.pinv(MM.T) 
            field_amp = 1-self.cmask
            lambdaOverD = self.pyr.lambdaInM/self.pupilSizeInM
            r0s = self.atmo_pars['r0']
            L0 = self.atmo_pars['outerScaleInM']
            turbulence = TurbulenceLayers(r0s,L0)
            for i in range(N):
                print(f'\rPhase realization {i+1}/{N}', end='\r', flush=True)
                turbulence.generate_phase_screens(screenSizeInMeters=self.pupilSizeInM,screenSizeInPixels=self.pupilSizeInPixels)
                turbulence.rescale_phasescreens()
                atmo_phase = xp.sum(turbulence.phase_screens,axis=0)
                atmo_phase -= xp.mean(atmo_phase[~self.cmask])
                phi = atmo_phase[~self.cmask]
                phi_atmo = phi*2*xp.pi/self.pyr.lambdaInM
                input_field = field_amp * xp.exp(1j*atmo_phase*2*xp.pi/self.pyr.lambdaInM)
                slopes = slope_computer.compute_slopes(input_field, lambdaOverD, None)
                rec_modes = slope_computer.Rec @ slopes
                rec_phi = MM[:slope_computer.nModes,:].T @ rec_modes
                res_phi = phi_atmo - rec_phi
                phi_modes = phase2modes @ phi_atmo
                lo_phi = MM.T @ phi_modes
                ho_phi = phi_atmo - lo_phi
                if mode_offset is not None:
                    phi_atmo += mode_offset
                    res_phi += mode_offset
                    ho_phi += mode_offset
                # ol_slopes = self._get_slopes(slope_computer, MM, self.pyr.lambdaInM, amps, phase_offset=phi_atmo)
                cl_slopes = self._get_slopes(slope_computer, MM, self.pyr.lambdaInM, amps, phase_offset=res_phi)
                pl_slopes = self._get_slopes(slope_computer, MM, self.pyr.lambdaInM, amps, phase_offset=ho_phi)
                cl_gains = xp.zeros(Nmodes)
                # ol_gains = xp.zeros(Nmodes)
                pl_gains = xp.zeros(Nmodes)
                for i in range(Nmodes):
                    # rec_modes = slope_computer.Rec @ ol_slopes[i,:]
                    # ol_gains[i] = rec_modes[i]
                    # rec_modes = slope_computer.Rec @ cl_slopes[i,:]
                    # cl_gains[i] = rec_modes[i]
                    calib_slope = IM[:,i]
                    norm = xp.dot(calib_slope,calib_slope)
                    # ol_gains[i] = xp.dot(ol_slopes[i,:],calib_slope)/norm
                    cl_gains[i] = xp.dot(cl_slopes[i,:],calib_slope)/norm
                    pl_gains[i] = xp.dot(pl_slopes[i,:],calib_slope)/norm
                cl_opt_gains += cl_gains/N
                # ol_opt_gains += ol_gains/N
                pl_opt_gains += pl_gains/N
            # myfits.save_fits(og_ol_path,ol_opt_gains)
            myfits.save_fits(og_cl_path,cl_opt_gains)
            myfits.save_fits(og_pl_path,pl_opt_gains)
        return cl_opt_gains, pl_opt_gains #ol_opt_gains, 
    

    def _get_slopes(self, slope_computer, MM, lambdaInM, amps, phase_offset=None):
        """ 
        Computes the slopes for the given mode matrix MM using push-pull method.

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
        phase_offset : array, optional
            Phase offset to be added to each mode, by default None.
        
        Returns 
        -------
        slopes : array
            The computed slopes for each mode.
        """
        Nmodes = xp.shape(MM)[0] #min(slope_computer.nModes,xp.shape(MM)[0])
        slopes = None
        electric_field_amp = 1-self.cmask
        lambdaOverD = lambdaInM/self.pupilSizeInM
        Nphotons = None # perfect calibration: no noise
        offset = reshape_on_mask(xp.zeros(int(xp.sum(1-self.cmask))), self.cmask)
        if phase_offset is not None:
            offset = reshape_on_mask(phase_offset, self.cmask)
        if isinstance(amps, float):
            amps *= xp.ones(Nmodes)
            rad_orders = xp.sqrt(xp.arange(Nmodes)+1)
            amps /= xp.sqrt(rad_orders)
        for i in range(Nmodes):
            if phase_offset is None:
                print(f'\rReconstructing mode {i+1}/{Nmodes}', end='\r', flush=True)
            amp = amps[i]
            mode_phase = reshape_on_mask(MM[i,:]*amp, self.cmask)
            push_field = xp.exp(1j*mode_phase + 1j*offset) * electric_field_amp
            pull_field = xp.exp(-1j*mode_phase + 1j*offset) * electric_field_amp
            push_slope = slope_computer.compute_slopes(push_field, lambdaOverD, Nphotons)/amp
            pull_slope = slope_computer.compute_slopes(pull_field, lambdaOverD, Nphotons)/amp
            if slopes is None:
                slopes = (push_slope-pull_slope)/2
            else:
                slopes = xp.vstack((slopes,(push_slope-pull_slope)/2))
        return slopes

    
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
    

    def _add_telescope_spiders(self, spiderWidth, spiderAngles):
        """
        Adds radial spiders to self.cmask.

        Parameters
        ----------
        spiderWidth : array
            The spiders' width in pixels.
        spiderAngles : array
            Spiders' orientation angle, CCW from East.
        """
        cmask = self.cmask.copy()
        cx,cy = cmask.shape[0]/2,cmask.shape[1]/2
        top = xp.zeros_like(cmask)
        top[cx:,:] = 1
        right = xp.zeros_like(cmask)
        right[:,cy:] = 1
        for angle in spiderAngles:
            dist = lambda x,y: xp.asarray(y-cy)-xp.asarray(x-cx)*xp.tan(angle) if abs(abs(angle)-xp.pi/2) > 1e-10 else xp.asarray(x-cx)
            spider_mask = xp.fromfunction(lambda j,i: abs(dist(i,j))<spiderWidth, cmask.shape)
            dist_grid = xp.fromfunction(lambda j,i: dist(i,j), cmask.shape)
            if xp.sin(angle) >= 0:
                spider_mask *= top
            else:
                spider_mask *= (1-top).astype(bool)
            if xp.cos(angle) >= 0:
                spider_mask *= right
            else:
                spider_mask *= (1-right).astype(bool)
            cmask = xp.logical_or(cmask, (spider_mask).astype(bool))
        self.cmask = cmask.copy()
        

    @staticmethod
    def phase_rms(vec):
        return xp.sqrt(xp.sum(vec**2)/len(vec))

            








        