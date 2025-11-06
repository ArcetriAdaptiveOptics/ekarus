import numpy as np
import configparser
import os
import xupy as xp

from ekarus.e2e.devices.deformable_mirror import DeformableMirror
import ekarus.e2e.utils.deformable_mirror_utilities as dmutils
from ekarus.e2e.utils.image_utils import get_circular_mask, reshape_on_mask
from ekarus.e2e.utils import my_fits_package as myfits
from ekarus.e2e.utils.root import alpaopath


class ALPAODM(DeformableMirror):

    def __init__(self, input, pupil_mask=None, **kwargs):
        """
        ALPAO DM constructor

        Parameters
        ----------
        input : int | str
            int: number of actuators of the ALPAO DM.
            str: tracking number of the DM measured iff
        pupil_mask: int | ndarray(bool)
            int: number of pixels across the mask diameter
            ndarray(bool): the actual mask as array of booleans
        """

        super().__init__()

        self.config = configparser.ConfigParser()
        self.alpaopath = alpaopath
        config_path = os.path.join(self.alpaopath,'configuration.ini')
        self.config.read(config_path)

        if isinstance(input, int):
            # pupilDiamInPixels = kwargs['Npix']
            self._init_ALPAO_from_Nacts(input,pupil_mask)#,pupilDiamInPixels=pupilDiamInPixels)
        elif isinstance(input, str):
            self._init_ALPAO_from_tn_data(input,pupil_mask)
        elif isinstance(input, xp.array):
            self._init_ALPAO_from_act_coords(input,pupil_mask)
        else:
            raise NotImplementedError(f'Initialization method for {input} not implemented, please pass a data tracking number or the number of actuators')

        self.pupil_mask = xp.logical_or(self.mask, pupil_mask)
        self.act_pos = xp.zeros(self.Nacts, dtype=xp.float)
        self.surface = self.IFF @ self.act_pos
        self.R, self.U = dmutils.compute_reconstructor(self.IFF)

        self.max_stroke = kwargs['max_stroke'] if 'max_stroke' in kwargs else None

        valid_ids = xp.arange(xp.sum(1-self.mask))
        master_ids = dmutils.find_master_acts(self.pupil_mask, self.act_coords, self.pixel_scale)
        if len(master_ids) < self.Nacts: # slaving
            self.slaving = dmutils.get_slaving_m2c(self.act_coords, master_ids, slaving_method='wmean', p=2, d_thr=2*self.pupil_size/self.Nacts)
            self.master_ids = master_ids
        valid_ids = xp.arange(xp.sum(1-self.mask))
        valid_ids_2d = reshape_on_mask(valid_ids, self.mask)
        self.visible_pix_ids = (valid_ids_2d[~self.pupil_mask]).astype(int)



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
        coords = xp.array([cx, cy], dtype=xp.float)

        return coords


    def _init_ALPAO_from_Nacts(self, Nacts:int, pupil_mask): #, multiprocessing:bool=False):
        """
        Initializes the ALPAO DM mask and actuator coordinates

        Parameters
        ----------
        Nacts : int
            The number of actuators in the DM.
        pupilDiamInPixels : int
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
        if isinstance(pupil_mask, int):
            pupilDiamInPixels = pupil_mask
            self.mask = xp.array(get_circular_mask((pupil_mask,pupil_mask),pupil_mask//2),dtype=bool)
        else:
            pupil_mask = (pupil_mask).astype(bool)
            pix_coords = dmutils.getMaskPixelCoords(pupil_mask)
            xx = pix_coords[0,~pupil_mask.flatten()] - max(pix_coords[0,:])/2
            yy = pix_coords[1,~pupil_mask.flatten()] - max(pix_coords[1,:])/2
            diagonals = xp.sqrt(xx**2+yy**2)*2
            pupilDiamInPixels = xp.max(diagonals)
            self.mask = xp.array(get_circular_mask(pupil_mask.shape,pupilDiamInPixels//2),dtype=bool)
        self.pixel_scale = pupilDiamInPixels/self.pupil_size

        dir_path = os.path.join(self.alpaopath,'DM'+str(self.Nacts)+'/')
        hdr_dict = {'N_ACTS': self.Nacts, 'PUP_SIZE': self.pupil_size, 'PIX_SIZE': pupilDiamInPixels}
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
            radii *= 1.02 # increase pupil by 2% to make sure all actuators fit
            self.act_coords = xp.asarray(coords*self.pupil_size/2/(2*max(radii)))
            myfits.save_fits(coords_path, self.act_coords, hdr_dict)

        Npix = xp.sum(1-self.mask)

        iff_path = os.path.join(dir_path,str(Npix)+'pixels_InfluenceFunctions.fits')
        try:
            self.IFF = myfits.read_fits(iff_path)
        except FileNotFoundError:
            self.IFF = dmutils.simulate_influence_functions(self.act_coords, self.mask, self.pixel_scale)
            myfits.save_fits(iff_path, self.IFF, hdr_dict)



    def _init_ALPAO_from_tn_data(self, tn, mask=None):
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
            self.CMat = xp.eye(self.Nacts, dtype=xp.float)

        dms = self.config[f'DM{self.Nacts}']
        pupil_size = eval(dms['opt_diameter'])*1e-3  # in meters

        dm_mask = xp.sum(abs(IM),axis=2)
        dm_mask = (dm_mask).astype(bool)
        dm_mask = 1-dm_mask
        pix_coords = dmutils.getMaskPixelCoords(dm_mask)
        self.mask = xp.array(dm_mask).astype(bool)
        xx = pix_coords[0,~self.mask.flatten()] - max(pix_coords[0,:])/2
        yy = pix_coords[1,~self.mask.flatten()] - max(pix_coords[1,:])/2
        mask_diameter = xp.sqrt(xx**2+yy**2)*2
        if mask is not None and self.mask != mask:
            raise NotImplementedError('Mask should be re-centered, cropped and rescaled!')
        self.pixel_scale = mask_diameter/pupil_size

        # Derive IFFs
        cube_mask = xp.tile(dm_mask,self.Nacts)
        cube_mask = xp.reshape(cube_mask, IM.shape, order = 'F')
        masked_cube = np.ma.masked_array(xp.asnumpy(IM),xp.asnumpy(cube_mask))
        self.IM = dmutils.cube2mat(masked_cube)
        self.IFF = self.IM @ xp.linalg.inv(self.CMat)

        self.act_coords, self._iff_act_pix_coords = dmutils.get_coords_from_IFF(self.IFF, self.mask, use_peak = True)
        # self.K = dmutils.estimate_stiffness_from_IFF(self.IM, self.CMat, self._iff_act_pix_coords, xp=xp)


    def _init_ALPAO_from_act_coords(self, act_coords, mask):
        """
        Get the ALPAO DM mask and influence functions from the actuator coordinated

        Parameters
        ----------
        act_coords : ndarray(float) [2,N]
            Actuator coordinates in meters
        """

        raise NotImplementedError()

