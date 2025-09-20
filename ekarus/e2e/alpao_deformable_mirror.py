import numpy as np
import configparser
import os
import xupy as xp

from ekarus.abstract_classes.deformable_mirror import DeformableMirror
import ekarus.e2e.utils.deformable_mirror_utilities as dmutils
from ekarus.e2e.utils.image_utils import get_circular_mask
from ekarus.e2e.utils import my_fits_package as myfits
from ekarus.e2e.utils.root import alpaopath


class ALPAODM(DeformableMirror):

    def __init__(self, input, **kwargs):
        """
        ALPAO DM constructor

        Parameters
        ----------
        input : int | str
            int: number of actuators of the ALPAO DM.
            str: tracking number of the DM measured iff
        """

        super().__init__(xp)

        self.config = configparser.ConfigParser()
        self.alpaopath = alpaopath
        config_path = os.path.join(self.alpaopath,'configuration.ini')
        self.config.read(config_path)

        if isinstance(input, int):
            Npix = kwargs['Npix']
            self._init_ALPAO_from_Nacts(input,Npix=Npix)
        elif isinstance(input, str):
            self._init_ALPAO_from_tn_data(input)
        elif isinstance(input, xp.array):
            self._init_ALPAO_from_act_coords(input)
        else:
            raise NotImplementedError(f'Initialization method for {input} not implemented, please pass a data tracking number or the number of actuators')

        self.act_pos = xp.zeros(self.Nacts, dtype=self.dtype)
        self.surface = self.IFF @ self.act_pos
        self.R, self.U = dmutils.compute_reconstructor(self.IFF)

        self.max_stroke = kwargs['max_stroke']
        

    def enslave_cmd(self, cmd, max_cmd: float = 0.9):
        """ DM utils slaving function wrapper """

        slaved_cmd = dmutils.slaving(self.act_coords, cmd, slaving_method = 'interp', cmd_thr = max_cmd)
        return slaved_cmd
    
    
    @staticmethod
    def _getALPAOcoordinates(nacts_row_sequence):
        """
        Generates the coordinates of the DM actuators for a given DM size and actuator sequence.
        
        Parameters
        ----------
        Nacts : int
            Total number of actuators in the DM.

        Returns
        -------
        np.array
            Array of coordinates of the actuators.
        """
        n_dim = nacts_row_sequence[-1]
        upper_rows = nacts_row_sequence[:-1]
        lower_rows = [l for l in reversed(upper_rows)]
        center_rows = [n_dim] * upper_rows[0]
        rows_number_of_acts = upper_rows + center_rows + lower_rows
        n_rows = len(rows_number_of_acts)
        cx = xp.array([], dtype=int)
        cy = xp.array([], dtype=int)
        for i in range(n_rows):
            cx = xp.concatenate((cx, xp.arange(rows_number_of_acts[i]) + (n_dim - rows_number_of_acts[i]) // 2))
            cy = xp.concatenate((cy, xp.full(rows_number_of_acts[i], i)))
        dtype = xp.float
        coords = xp.array([cx, cy], dtype=dtype)

        return coords


    def _init_ALPAO_from_Nacts(self, Nacts:int, Npix:int): #, multiprocessing:bool=False):
        """ 
        Initializes the ALPAO DM mask and actuator coordinates

        Parameters
        ----------
        Nacts : int
            The number of actuators in the DM.
        Npix : int
            The number of pixels across the pupil
        """
        self.Nacts = Nacts

        # read configuration file
        try:
            dms = self.config[f'DM{Nacts}']
        except:
            raise ValueError(f'ALPAO DM{Nacts} not found in configuration file')
        nacts_row_sequence = eval(dms['coords'])
        self.pupil_size = eval(dms['opt_diameter'])*1e-3  # in meters

        # Define mask & pixel scale
        self.mask = get_circular_mask((Npix,Npix),Npix//2)
        self.pixel_scale = Npix/self.pupil_size

        dir_path = os.path.join(self.alpaopath,'DM'+str(self.Nacts)+'/')
        hdr_dict = {'N_ACTS': self.Nacts, 'PUP_SIZE': self.pupil_size, 'PIX_SIZE': Npix}
        try:
            os.mkdir(dir_path)
        except FileExistsError:
            pass

        # Define coordinates in meters, centering in (0,0)
        coords_path = os.path.join(dir_path,'ActuatorCoordinates.fits')
        try:
            self.act_coords = myfits.read_fits(coords_path)
        except FileNotFoundError:
            coords = self._getALPAOcoordinates(nacts_row_sequence)
            coords[0] -= (max(coords[0])-min(coords[0]))/2
            coords[1] -= (max(coords[1])-min(coords[1]))/2   
            radii = xp.sqrt(coords[0]**2+coords[1]**2)/2
            self.act_coords = coords*self.pupil_size/2/(2*max(radii))
            myfits.save_fits(coords_path, self.act_coords, hdr_dict)

        iff_path = os.path.join(dir_path,str(Npix)+'pixels_InfluenceFunctions.fits')
        try:
            self.IFF = myfits.read_fits(iff_path)
        except FileNotFoundError:
            self.IFF = dmutils.simulate_influence_functions(self.act_coords, self.mask, self.pixel_scale)
            myfits.save_fits(iff_path, self.IFF, hdr_dict)



    def _init_ALPAO_from_tn_data(self, tn):
        """
        Get the ALPAO DM mask and actuator coordinates from the interaction matrix.

        Parameters
        ----------
        tn : string
            Tracking number of the saved data
        """

        im_path = os.path.join(self.alpaopath, str(tn) + '/IMCube.fits')
        IM = myfits.read_fits(im_path)

        self.Nacts = IM.shape[2]

        try:
            cmdmat_path = os.path.join(self.alpaopath, str(tn) + '/cmdMatrix.fits')
            self.CMat = myfits.read_fits(cmdmat_path)
        except FileNotFoundError:
            self.CMat = xp.eye(self.Nacts, dtype=self.dtype)

        dms = self.config[f'DM{self.Nacts}']
        pupil_size = eval(dms['opt_diameter'])*1e-3  # in meters

        pupil_mask = xp.sum(abs(IM),axis=2)
        pupil_mask = (pupil_mask).astype(bool)
        pupil_mask = 1-pupil_mask
        pix_coords = dmutils.getMaskPixelCoords(pupil_mask)
        self.mask = (pupil_mask).astype(bool)
        xx = pix_coords[0,~self.mask.flatten()] - max(pix_coords[0,:])/2
        yy = pix_coords[1,~self.mask.flatten()] - max(pix_coords[1,:])/2
        mask_diameter = xp.sqrt(xx**2+yy**2)*2
        self.pixel_scale = mask_diameter/pupil_size

        # Derive IFFs
        cube_mask = xp.tile(pupil_mask,self.Nacts)
        cube_mask = xp.reshape(cube_mask, IM.shape, order = 'F')
        masked_cube = np.ma.masked_array(IM,cube_mask)
        self.IM = dmutils.cube2mat(masked_cube,xp=xp)
        self.IFF = self.IM @ xp.linalg.inv(self.CMat)

        self.act_coords, self._iff_act_pix_coords = dmutils.get_coords_from_IFF(self.IFF, self.mask, use_peak = True)
        # self.K = dmutils.estimate_stiffness_from_IFF(self.IM, self.CMat, self._iff_act_pix_coords, xp=xp)


    def _init_ALPAO_from_act_coords(self, act_coords):
        """
        Get the ALPAO DM mask and influence functions from the actuator coordinated

        Parameters
        ----------
        act_coords : ndarray(float) [2,N]
            Actuator coordinates in meters
        """

        raise NotImplementedError()

