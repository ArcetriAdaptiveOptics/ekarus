import numpy as np

from tps import ThinPlateSpline # for the simulated IFF
from scipy.interpolate import griddata


def slaving(coords, cmd, slaving_method:str = 'interp', cmd_thr:float = None, xp=np):
    """ 
    Clip the command to avoid saturation

    Parameters
    ----------
    coords : ndarray(float) [2,Nacts]
        The actuator coordinates

    cmd : ndarray(float) [Nacts]
        The mirror command to be slaved

    cmd_thr : float, optional
        Threshold to define slave actuators.
        Slave acts are defined as the actuators for which: cmd > cmd_thr.
        Default: slave acts the ones outside 3 sigma of the mean cmd
        
    slaving_method : str, optional
        The way to treat slave actuators.
            'tps'     : thin plate spline interpolation. DEFAULT
            'zero'    : set slave actuator command to zero 
            'clip'    : clips the actuator commands to the given cmd_thr input
            'nearest' : nearest actuators interpolation
            'wmean'   : mean of nearby actuators weighted on 1/r^2

            'exclude' : exclude slaves from the reconstructor computation (TBI)

    """
    # Define master ids
    n_acts = len(cmd)
    act_ids = xp.arange(n_acts)

    if cmd_thr is None:
        master_ids = act_ids[abs(cmd-xp.mean(cmd)) <= 3*xp.std(cmd)]
    else:
        master_ids = act_ids[abs(cmd) <= cmd_thr]
    
    match slaving_method:

        case 'tps':
            tps = ThinPlateSpline(alpha=0.0)
            tps.fit(coords[master_ids], cmd[master_ids])
            rescaled_cmd = tps.transform(coords)
            slaved_cmd = rescaled_cmd[:,0]

        case 'zero':
            pad_cmd = xp.zeros_like(cmd)
            pad_cmd[master_ids] = cmd[master_ids]
            slaved_cmd = pad_cmd

        case 'clip':
            slaved_cmd = xp.minimum(abs(cmd), cmd_thr)
            slaved_cmd *= xp.sign(cmd)

        case 'nearest':
            master_coords = coords[:,master_ids]
            slaved_cmd = griddata(master_coords, cmd[master_ids], (coords[0], coords[1]), method='nearest')

        case 'wmean':
            master_coords = coords[:,master_ids]
            master_cmd = cmd[master_ids]
            dist2 = lambda xy: (xy[0]-master_coords[0])**2 + (xy[1]-master_coords[1])**2 
            is_slave = xp.ones_like(cmd, dtype=bool)
            is_slave[master_ids] = False
            slave_ids = act_ids[is_slave]
            slaved_cmd = cmd.copy()
            for slave in slave_ids:
                d2_slave = dist2(coords[:,slave])
                weighted_cmd = master_cmd / d2_slave
                slaved_cmd[slave] = xp.sum(weighted_cmd)*xp.sum(d2_slave)/n_acts

        # case 'exclude':
        #     masked_IFF = self.IFF[valid_ids,:]
        #     masked_IFF = masked_IFF[:,visible_acts]
        #     masked_R = np.linalg.pinv(masked_IFF)
            
        #     pad_cmd = np.zeros_like(act_cmd)
        #     pad_cmd[visible_acts] = matmul(masked_R, masked_shape)
        #     act_cmd = pad_cmd
            
        case _:
            raise NotImplementedError(f"{slaving_method} is not an available slaving method. Available methods are: 'tps', 'zer', 'clip', 'nearest', 'wmean'")

    return slaved_cmd
    

def compute_reconstructor(M, thr:float= 1e-12, xp=np):
    """
    Computes the reconstructor (pseudo-inverse) 
    for the interaction matrix M.

    Parameters
    ----------
    M : ndarray(float) [Npix,N]
        Interaction matrix to be inverted.
        
    thr : float, optional
        Threshold for the inverse eigenvalues. 
        The eigenvalues v  s. t. 1/v < thr 
        are discarded when computing the inverse.
        By default, no eigenvalues are discarded.

    Returns
    -------
    Rec : ndarray(float) [N,Npix]
        The pseudo-inverse of M.
    U : ndarray(float) [Npix,N]
        The lefteigenmodes of M.

    """

    U,S,V = xp.linalg.svd(M, full_matrices=False)
    Sinv = 1/S
    Sinv[Sinv < thr] = 0
    Rec = (V.T * Sinv) @ U.T

    return Rec, U
    
    
def simulate_influence_functions(act_coords, local_mask, pix_scale:float = 1.0, xp=np):
    """ Simulate the influence functions by 
    imposing 'perfect' zonal commands """
    
    # n_acts = np.max(np.shape(act_coords))
    # H,W = np.shape(local_mask)
    n_acts = max(act_coords.shape)

    mask_ids = xp.arange(xp.size(local_mask))
    pix_ids = mask_ids[~(local_mask).flatten()]
    
    pix_coords = getMaskPixelCoords(local_mask).T
    act_pix_coords = get_pixel_coords(local_mask, act_coords, pix_scale).T
    
    dtype = xp.float32 if xp.__name__ == 'cupy' else xp.float64
    IFF = xp.zeros([len(pix_ids),n_acts], dtyep=dtype)

    for k in range(n_acts):
        act_data = xp.zeros(n_acts, dtype=dtype)
        act_data[k] = 1e-6
        tps = ThinPlateSpline(alpha=0.0)
        tps.fit(act_pix_coords, act_data)
        img = tps.transform(pix_coords[pix_ids,:])
        IFF[:,k] = img[:,0]

    return IFF


def get_pixel_coords(mask, coords, pix_scale:float = 1.0, xp=np):
    """ 
    Convert x,y coordinates in coords to pixel coordinates
    or get the pixel coordinates of a mask

    Parameters
    ----------
    mask : ndarray(bool)
        The image mask where the pixels are.
        
    coords : ndarray(float) [2,N]
        The N coordinates to convert in pixel coordinates.
        Defaults to all pixels on the mask.
        
    pix_scale : float (Optional)
        The number of pixels per meter.
        Defaults to 1.0

    Returns
    -------
    pix_coords : ndarray(int) [2,N]
        The obtained pixel coordinates.
    """
    
    H,W = mask.shape
    dtype = xp.float32 if xp.__name__ == 'cupy' else xp.float64
    pix_coords = xp.zeros([2,xp.shape(coords)[-1]], dtype=dtype)
    pix_coords[0,:] = (coords[1,:]*pix_scale/2 + H)/2
    pix_coords[1,:] = (coords[0,:]*pix_scale/2 + W)/2
    
    return pix_coords


def get_coords_from_IFF(IFF, mask, use_peak=True, xp=np):
    """
    Get the coordinates of the actuators from the influence functions matrix.

    Parameters
    ----------
    IFF : ndarray(float) [Npix,Nacts]
        The influence functions matrix.
        
    mask : ndarray(bool) [Npix,Npix]
        The DM mask.
    
    use_peak : bool, optional
        If True, actuator coordinates are computed from the IFF peak.
        If False, actuator coordinates are computed from the photocenter of the IFF.
        Defaults to True, the photocenter approach seems to be giving issues

    Returns
    -------
    coords : ndarray(float) [2,Nacts]
        The coordinates of the actuators in the mask.
    """
    
    # Get pixel coordinates
    pix_coords = getMaskPixelCoords(mask)

    x_coords = pix_coords[0,:]
    y_coords = pix_coords[1,:]

    mask = (mask).astype(bool) # ensure mask is boolean
    x_coords = x_coords[~mask.flatten()]
    y_coords = y_coords[~mask.flatten()]

    # Get the coordinates of the actuators
    n_acts = IFF.shape[1]
    dtype = xp.float32 if xp.__name__ == 'cupy' else xp.float64
    act_coords = xp.zeros([2, n_acts], dtype=dtype)

    for k in range(n_acts):
        act_data = IFF[:, k]
        if use_peak:
            max_id = xp.argmax(act_data)
            act_coords[0,k] = x_coords[max_id]
            act_coords[1,k] = y_coords[max_id] 
        else:
            act_coords[0,k] = xp.sum(x_coords * act_data) / xp.sum(act_data)
            act_coords[1,k] = xp.sum(y_coords * act_data) / xp.sum(act_data)

    return act_coords


def cube2mat(cube, xp=np):
    """ Get influence functions matrix 
    from the image cube """
    
    n_acts = cube.shape[2]
    valid_len = int(xp.sum(1-cube.mask)/n_acts)
    
    flat_cube = cube.data[~cube.mask]
    local_IFF = xp.reshape(flat_cube, [valid_len, n_acts])

    dtype = xp.float32 if xp.__name__ == 'cupy' else xp.float64
    IFF = xp.array(local_IFF, dtype=dtype)
    
    return IFF


def getMaskPixelCoords(mask, xp=np):
    """ 
    Get the pixel coordinates of a mask

    Parameters
    ----------
    mask : ndarray(bool)
        The image mask where the pixels are.

    Returns
    -------
    pix_coords : ndarray(int) [2,N]
        The obtained pixel coordinates.
    """
    
    H,W = mask.shape
    dtype = xp.float32 if xp.__name__ == 'cupy' else xp.float64
    pix_coords = xp.zeros([2,H*W], dtype=dtype)
    pix_coords[0,:] = xp.repeat(xp.arange(H),W)
    pix_coords[1,:] = xp.tile(xp.arange(W),H)
    
    return pix_coords