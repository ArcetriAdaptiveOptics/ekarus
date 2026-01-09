import numpy as np
# import configparser
import os
import xupy as xp

from ekarus.e2e.devices.deformable_mirror import DeformableMirror
import ekarus.e2e.utils.deformable_mirror_utilities as dmutils
from ekarus.e2e.utils.image_utils import get_circular_mask, reshape_on_mask
from ekarus.e2e.utils import my_fits_package as myfits
from ekarus.e2e.utils.root import dmpath as _dmpath


class DSM(DeformableMirror):

    def __init__(self, input, pupil_mask=None, **kwargs):
        """
        DSM constructor

        Parameters
        ----------
        input : int | str
            int: number of actuators on the mirror diameter.
            str: tracking number of the DM measured iff
        pupil_mask: int | ndarray(bool)
            int: number of pixels across the mask diameter
            ndarray(bool): the actual mask as array of booleans
        """

        super().__init__()

        self._dmpath = _dmpath

        if isinstance(input, int):
            geom = kwargs['geom']
            self._init_from_Nacts(input,pupil_mask,geom)
        elif isinstance(input, str):
            self._init_from_tn_data(input,pupil_mask)
        elif isinstance(input, xp.array):
            self._init_from_act_coords(input,pupil_mask)
        else:
            raise NotImplementedError(f'Initialization method for {input} not implemented, please pass a data tracking number or the number of actuators')

        self.pupil_mask = xp.logical_or(self.mask, pupil_mask)
        self.act_pos = xp.zeros(self.Nacts, dtype=xp.float)
        self.surface = self.IFF @ self.act_pos
        # self.R, self.U = dmutils.compute_reconstructor(self.IFF)

        self.max_stroke = kwargs['max_stroke'] if 'max_stroke' in kwargs else None

        valid_ids = xp.arange(xp.sum(1-self.mask))
        master_ids = dmutils.find_master_acts(self.pupil_mask, self.act_coords, d_thr = xp.sqrt(4*self.pitch))
        if len(master_ids) < self.Nacts: # slaving
            # self.slaving = dmutils.get_slaving_m2c(self.act_coords, master_ids, slaving_method='wmean', p=2, d_thr=1.0*self.pitch)
            self.slaving = dmutils.get_slaving_m2c(self.act_coords, master_ids, slaving_method='zero')
            self.master_ids = master_ids
        valid_ids = xp.arange(xp.sum(1-self.mask))
        valid_ids_2d = reshape_on_mask(valid_ids, self.mask)
        self.visible_pix_ids = (valid_ids_2d[~self.pupil_mask]).astype(int)


    @staticmethod
    def _getDMcoordinates(n_act:int, dim:int, geom:str='circular', angle_offset:float=0.0):
        """
        Generates the coordinates of the DM actuators for a given DM size and actuator sequence.

        Parameters
        ----------
        n_act : int
            Number of actuators on the DM diameter
        dim: int
            Number of pixels on the pupil diameter

        geom: str (optional)
            Type of geometry:
                'circular': Microgate-like (Default)
                'alpao':    square grid cut to circular shape
                'square':   square grid
        angle_offset: float (optional)
            Only used ofr circular input: offset of the grid from horizontal

        Returns
        -------
        np.array
            Array of coordinates of the actuators.
        """
        step = float(dim)/float(n_act)    
        match geom:
            case 'circular':
                if n_act % 2 == 0:
                    n_act_radius = int(xp.ceil((n_act + 1) / 2))
                else:
                    n_act_radius = int(xp.ceil(n_act / 2))
                na = xp.arange(n_act_radius) * 6
                na[0] = 1  # The first value is always 1
                n_act_tot = int(xp.sum(na))
                # Calculate step based on number of actuators on diameter
                n_act_diameter = 2 * n_act_radius - 1
                step = float(dim - 1) / float(n_act_diameter - 1)

                pol_coords = xp.zeros((2, n_act_tot))
                ka = 0
                # Refactor this!
                for ia, _ in enumerate(na):
                    n_angles = int(na[ia])
                    for ja in range(n_angles):
                        pol_coords[0, ka] = 360. / na[ia] * ja + angle_offset  # Angle in degrees
                        pol_coords[1, ka] = ia * step  # Radial distance
                        ka += 1
                # System center - use (dim-1)/2 to properly center on the grid
                x_c, y_c = (dim - 1) / 2.0, (dim - 1) / 2.0
                # Convert from polar to Cartesian coordinates
                x = pol_coords[1] * xp.cos(xp.radians(pol_coords[0])) + x_c
                y = pol_coords[1] * xp.sin(xp.radians(pol_coords[0])) + y_c
            case 'alpao':
                x, y = xp.meshgrid(xp.linspace(0, dim - 1, n_act), xp.linspace(0, dim - 1, n_act))
                x, y = x.ravel(), y.ravel()
                x_c, y_c = (dim - 1) / 2.0, (dim - 1) / 2.0 # center
                rho = xp.sqrt((x-x_c)**2+(y-y_c)**2)
                rho_max = ((dim - 1)*(9/8-n_act/(24*16)))/2 # slightly larger than (dim-1)/2, depends on n_act
                x = x[rho<=rho_max]
                y = y[rho<=rho_max]
            case _:
                x, y = xp.meshgrid(xp.linspace(0, dim - 1, n_act), xp.linspace(0, dim - 1, n_act))
                x, y = x.ravel(), y.ravel()

        coords = xp.vstack((x,y)) #xp.array([x,y])
        return coords


    def _init_from_Nacts(self, n_act:int, pupil_mask, geom:str='alpao'): 
        """
        Initializes the DM mask and actuator coordinates

        Parameters
        ----------
        n_act : int
            The number of actuators on the DM diameter.
        pupil_mask : int | ndarray(bool)
            The number of pixels across the pupil or the pupil mask
        
        geom: str (optional)
            Type of geometry:
                'circular': Microgate-like
                'alpao':    square grid cut to circular shape (Default)
                'square':   square grid
        """

        # Define mask & pixel scale
        if isinstance(pupil_mask, int):
            pupilDiamInPixels = pupil_mask
            maskRadius = pupil_mask/2
            self.mask = xp.array(get_circular_mask((pupil_mask,pupil_mask),maskRadius),dtype=bool)
        else:
            pupil_mask = (pupil_mask).astype(bool)
            pix_coords = dmutils.getMaskPixelCoords(pupil_mask)
            xx = pix_coords[0,~pupil_mask.flatten()] - max(pix_coords[0,:])/2
            yy = pix_coords[1,~pupil_mask.flatten()] - max(pix_coords[1,:])/2
            diagonals = xp.sqrt(xx**2+yy**2)*2
            pupilDiamInPixels = xp.max(diagonals)
            maskRadius = max(pupil_mask.shape)/2
            self.mask = xp.array(get_circular_mask(pupil_mask.shape,maskRadius),dtype=bool)

        dir_path = os.path.join(self._dmpath,str(geom)+'_'+str(n_act)+'acts_on_diam/')
        hdr_dict = {'N_ACTS': n_act, 'PIX_SIZE': pupilDiamInPixels}
        try:
            os.mkdir(dir_path)
        except FileExistsError:
            pass

        self.pupil_size = pupilDiamInPixels
        Npix = xp.sum(1-self.mask)
        self.pitch = self.pupil_size/n_act

        coords_path = os.path.join(dir_path,str(Npix)+'pixels_ActuatorCoordinates.fits')
        try:
            self.act_coords = myfits.read_fits(coords_path)
        except FileNotFoundError:
            D = max(pupil_mask.shape)
            coords = self._getDMcoordinates(n_act, D, geom=geom)
            centered_coords = xp.asarray(coords,dtype=float)
            centered_coords[0] -= D/2
            centered_coords[1] -= D/2
            centered_coords *= (1.0 - 40**2/100/n_act**2)
            coords[0] = centered_coords[0] + D/2
            coords[1] = centered_coords[1] + D/2
            self.act_coords = coords.copy()
            myfits.save_fits(coords_path, self.act_coords, hdr_dict)

        self.Nacts = max(self.act_coords.shape)

        iff_path = os.path.join(dir_path,str(Npix)+'pixels_InfluenceFunctions.fits')
        try:
            self.IFF = myfits.read_fits(iff_path)
        except FileNotFoundError:
            self.IFF = dmutils.simulate_influence_functions(self.act_coords, self.mask)
            myfits.save_fits(iff_path, self.IFF, hdr_dict)



    def _init_from_tn_data(self, tn, mask=None):
        """
        Get the DM mask and actuator coordinates from the interaction matrix.

        Parameters
        ----------
        tn : string
            Tracking number of the saved data
        """

        im_path = os.path.join(self._dmpath, str(tn) + '/IMCube.fits')
        IM = myfits.read_fits(im_path)

        self.Nacts = IM.shape[2]

        try:
            cmdmat_path = os.path.join(self._dmpath, str(tn) + '/cmdMatrix.fits')
            self.CMat = myfits.read_fits(cmdmat_path)
        except FileNotFoundError:
            self.CMat = xp.eye(self.Nacts, dtype=xp.float)


        dm_mask = xp.sum(abs(IM),axis=2)
        dm_mask = (dm_mask).astype(bool)
        dm_mask = 1-dm_mask
        pix_coords = dmutils.getMaskPixelCoords(dm_mask)
        self.mask = xp.array(dm_mask).astype(bool)
        xx = pix_coords[0,~self.mask.flatten()] - max(pix_coords[0,:])/2
        yy = pix_coords[1,~self.mask.flatten()] - max(pix_coords[1,:])/2
        # mask_diameter = xp.sqrt(xx**2+yy**2)*2
        if mask is not None and self.mask != mask:
            raise NotImplementedError('Mask should be re-centered, cropped and rescaled!')

        # Derive IFFs
        cube_mask = xp.tile(dm_mask,self.Nacts)
        cube_mask = xp.reshape(cube_mask, IM.shape, order = 'F')
        masked_cube = np.ma.masked_array(xp.asnumpy(IM),xp.asnumpy(cube_mask))
        self.IM = dmutils.cube2mat(masked_cube)
        self.IFF = self.IM @ xp.linalg.inv(self.CMat)

        self.act_coords, self._iff_act_pix_coords = dmutils.get_coords_from_IFF(self.IFF, self.mask, use_peak = True)
        # self.K = dmutils.estimate_stiffness_from_IFF(self.IM, self.CMat, self._iff_act_pix_coords, xp=xp)


    def _init_from_act_coords(self, act_coords, mask):
        """
        Get the DM mask and influence functions from the actuator coordinated

        Parameters
        ----------
        act_coords : ndarray(float) [2,N]
            Actuator coordinates in meters
        """

        raise NotImplementedError()

