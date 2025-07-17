import numpy as np
import configparser
from astropy.io import fits as pyfits

from ekarus.e2e.utils.deformable_mirror_utilities import getMaskPixelCoords, get_coords_from_IFF, simulate_influence_functions, cube2mat


def init_ALPAO(input):
    """ Wrapper for ALPAO DM initialization functions """
    if isinstance(input, int):
        mask, act_coords, pixel_scale, IFF = _init_ALPAO_from_Nacts(input)
    elif isinstance(input, str):
        mask, act_coords, pixel_scale, IFF = _init_ALPAO_from_tn_data(input)
    else:
        raise NotImplementedError(f'Initialization method for {input} not implemented, please pass a data tracking number or the number of actuators')
    
    return mask, act_coords, pixel_scale, IFF


def _init_ALPAO_from_Nacts(Nacts:int, Npix:int = 128):
    """ 
    Initializes the ALPAO DM mask and actuator coordinates

    Parameters
    ----------
    Nacts : int
        The number of actuators in the DM.
    """

    # read configuration file
    config = configparser.ConfigParser()
    config.read('../alpao_dms/configuration.ini')
    dms = config[f'DM{Nacts}']
    nacts_row_sequence = eval(dms['coords'])
    pupil_size = eval(dms['opt_diameter'])*1e-3  # in meters

    # Define mask & pixel scale
    mask = np.fromfunction(lambda y,x: np.sqrt((x-Npix//2)**2+(y-Npix//2)**2)>Npix//2, (Npix, Npix))
    pix_scale = Npix/pupil_size

    # Define coordinates in meters, centering in (0,0)
    coords = (_getALPAOcoordinates(nacts_row_sequence)).astype(float)
    coords[0] -= (np.max(coords[0])-np.min(coords[0]))/2
    coords[1] -= (np.max(coords[1])-np.min(coords[1]))/2    
    radii = np.sqrt(coords[0]**2+coords[1]**2)/2
    coords *= pupil_size/np.max(radii)

    IFF = simulate_influence_functions(coords, mask, pix_scale)
    # IFF = cube2mat(IMCube)

    return mask, coords, pix_scale, IFF



def _init_ALPAO_from_tn_data(tn):
    """
    Get the ALPAO DM mask and actuator coordinates from the interaction matrix.

    Parameters
    ----------
    tn : string
        Tracking number of the saved data
    """

    IM = _read_fits('../alpao_dms/' + str(tn) + '/IMCube.fits')
    CMat = _read_fits('../alpao_dms/' + str(tn) + '/cmdMatrix.fits')

    Nacts = np.shape(IM)[2]

    config = configparser.ConfigParser()
    config.read('../alpao_dms/configuration.ini')
    dms = config[f'DM{Nacts}']
    pupil_size = eval(dms['opt_diameter'])*1e-3  # in meters

    # Command matrix
    if CMat is None:
        CMat = np.eye(Nacts)

    pupil_mask = np.sum(np.abs(IM),axis=2)
    pupil_mask = (pupil_mask).astype(bool)
    pupil_mask = 1-pupil_mask
    pix_coords = getMaskPixelCoords(pupil_mask)
    pupil_mask = (pupil_mask).astype(bool)
    xx = pix_coords[0,~pupil_mask.flatten()] - np.max(pix_coords[0,:])/2
    yy = pix_coords[1,~pupil_mask.flatten()] - np.max(pix_coords[1,:])/2
    mask_diameter = np.sqrt(xx**2+yy**2)*2
    pix_scale = mask_diameter/pupil_size

    # Derive IFFs
    cube_mask = np.tile(pupil_mask,Nacts)
    cube_mask = np.reshape(cube_mask, np.shape(IM), order = 'F')
    masked_cube = np.ma.masked_array(IM,cube_mask)
    IM = cube2mat(masked_cube)
    IFF = IM @ np.linalg.inv(CMat)

    act_coords = get_coords_from_IFF(IFF, pupil_mask, use_peak = True)
    
    return pupil_mask, act_coords, pix_scale, IFF


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
    cx = np.array([], dtype=int)
    cy = np.array([], dtype=int)
    for i in range(n_rows):
        cx = np.concatenate((cx, np.arange(rows_number_of_acts[i]) + (n_dim - rows_number_of_acts[i]) // 2))
        cy = np.concatenate((cy, np.full(rows_number_of_acts[i], i)))
    coords = np.array([cx, cy])

    return coords


def _read_fits(file_path):
    """ Basic function to read fits files """
    with pyfits.open(file_path) as hdu:
        data_out = np.array(hdu[0].data)
    return data_out