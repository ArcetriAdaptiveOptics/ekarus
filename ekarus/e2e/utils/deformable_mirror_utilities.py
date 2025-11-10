import xupy as xp
import numpy as np

from tps import ThinPlateSpline # for the simulated IFF
# from scipy.interpolate import griddata

import matplotlib.pyplot as plt
    

def compute_reconstructor(M, thr:float= 1e-12):
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
    
    
def simulate_influence_functions(act_coords, local_mask):#, pix_scale:float=1.0):
    """ Simulate the influence functions by 
    imposing 'perfect' zonal commands """
    
    # local_mask = xp.asnumpy(local_mask)
    # act_coords = xp.asnumpy(act_coords)

    n_acts = max(act_coords.shape)

    mask_ids = xp.arange(xp.size(local_mask))
    pix_ids = mask_ids[xp.invert(local_mask).flatten()]
    
    pix_coords = xp.asnumpy(getMaskPixelCoords(local_mask)).T
    act_pix_coords = xp.asnumpy(act_coords.T) #xp.asnumpy(get_pixel_coords(local_mask, act_coords, pix_scale)).T
    pix_ids = xp.asnumpy(pix_ids)
        
    IFF = np.zeros([len(pix_ids),n_acts])
    for k in range(n_acts):
        print(f'\rSimulating influence functions: {k+1}/{n_acts}', end ='\r', flush=True)
        act_data = np.zeros(n_acts)
        act_data[k] = 1.0
        tps = ThinPlateSpline(alpha=0.0)
        tps.fit(act_pix_coords, act_data)
        img = tps.transform(pix_coords[pix_ids,:])
        IFF[:,k] = img[:,0]

    IFF = xp.asarray(IFF)#,dtype=xp.float)

    return IFF


# def get_pixel_coords(mask, coords, pix_scale:float = 1.0):
#     """ 
#     Convert x,y coordinates in coords to pixel coordinates
#     or get the pixel coordinates of a mask

#     Parameters
#     ----------
#     mask : ndarray(bool)
#         The image mask where the pixels are.
        
#     coords : ndarray(float) [2,N]
#         The N coordinates to convert in pixel coordinates.
#         Defaults to all pixels on the mask.
        
#     pix_scale : float (Optional)
#         The number of pixels per meter.
#         Defaults to 1.0

#     Returns
#     -------
#     pix_coords : ndarray(int) [2,N]
#         The obtained pixel coordinates.
#     """
    
#     H,W = mask.shape
#     pix_coords = xp.zeros([2,xp.shape(coords)[-1]], dtype=xp.float)
#     pix_coords[0,:] = coords[1,:]*pix_scale + H/2 #(coords[1,:]*pix_scale/2 + H)/2
#     pix_coords[1,:] = coords[0,:]*pix_scale + W/2 #(coords[0,:]*pix_scale/2 + W)/2
    
#     return pix_coords


def get_coords_from_IFF(IFF, mask, use_peak=True):
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
    xy_pix_coords = getMaskPixelCoords(mask)

    x_coords = xy_pix_coords[0,:]
    y_coords = xy_pix_coords[1,:]

    mask = (mask).astype(bool) # ensure mask is boolean
    x_coords = x_coords[xp.invert(mask).flatten()]
    y_coords = y_coords[xp.invert(mask).flatten()]

    dtype = xp.float

    # Get the coordinates of the actuators
    n_acts = IFF.shape[1]
    act_coords = xp.zeros([2, n_acts], dtype=dtype)
    iff_pix_coords = xp.zeros(n_acts, dtype=int)

    for k in range(n_acts):
        act_data = IFF[:, k]
        iff_pix_coords[k] = xp.argmax(act_data)
        if use_peak:
            max_id = xp.argmax(act_data)
            act_coords[0,k] = x_coords[max_id]
            act_coords[1,k] = y_coords[max_id] 
        else:
            act_coords[0,k] = xp.sum(x_coords * act_data) / xp.sum(act_data)
            act_coords[1,k] = xp.sum(y_coords * act_data) / xp.sum(act_data)

    return act_coords, iff_pix_coords


def estimate_stiffness_from_IFF(IFF, CMat, act_pix_coords, cmdAmps=None):
    """
    Get the coordinates of the actuators from the influence functions matrix.

    Parameters
    ----------
    IFF : ndarray(float) [Npix,Nacts]
        The influence functions matrix.

    Returns
    -------
    K : ndarray(float) [Nacts,Nacts]
        The stiffness matrix of the DM.
    """
    n_acts = IFF.shape[1]
    PMat = xp.zeros([n_acts,n_acts], dtype=xp.float)

    for k in range(n_acts):
        act_data = IFF[:, k]
        pos = act_data[act_pix_coords].copy()
        if cmdAmps is not None:
            pos *= 1/cmdAmps[k]
        PMat[:,k] = pos.copy()

    K = PMat @ xp.linalg.inv(CMat)

    return K


def cube2mat(cube):
    """ Get influence functions matrix 
    from the image cube """
    
    n_acts = cube.shape[2]
    valid_len = int(np.sum(1-cube.mask)/n_acts)

    flat_cube = cube.data[np.invert(cube.mask)]
    local_IFF = np.reshape(flat_cube, [valid_len, n_acts])

    IFF = xp.array(local_IFF, dtype=xp.float)
    
    return IFF


def getMaskPixelCoords(mask):
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
    dtype = xp.float
    pix_coords = xp.zeros([2,H*W], dtype=dtype)
    pix_coords[0,:] = xp.repeat(xp.arange(H),W)
    pix_coords[1,:] = xp.tile(xp.arange(W),H)
    
    return pix_coords


def find_master_acts(mask, coords, d_thr:float=xp.sqrt(2.0)):#, pix_scale:float = 1.0):
    """ Find the master actuator ids """

    nActs = len(coords[0,:])
    act_pix_coords = coords.copy() # get_pixel_coords(mask, coords, pix_scale)
    mask_coords = getMaskPixelCoords(mask)

    valid_mask_coords = mask_coords[:,~mask.flatten()]
    dist = lambda xy: xp.sqrt((xy[0]-valid_mask_coords[0])**2 
                              + (xy[1]-valid_mask_coords[1])**2)
    
    master_ids = []
    for i in range(nActs):
        min_pix_dist = xp.min(dist(act_pix_coords[:,i]))
        if min_pix_dist <= d_thr:# 1.0:
            master_ids.append(i)
    
    master_ids = xp.array(master_ids)
    if len(master_ids) < nActs:
        print(f'Unobscured actuators: {len(master_ids)}/{nActs}')
        plt.figure()
        plt.imshow(xp.asnumpy(mask),origin='lower',cmap='grey')
        plt.scatter(xp.asnumpy(act_pix_coords[0]),xp.asnumpy(act_pix_coords[1]),c='red',label='slaves')
        plt.scatter(xp.asnumpy(act_pix_coords[0,master_ids]),xp.asnumpy(act_pix_coords[1,master_ids]),c='green',label='masters')
        plt.legend()
        plt.grid()
    
    return master_ids


def get_slaving_m2c(coords, master_ids, slaving_method:str='wmean', p:int=1, d_thr:float=xp.inf):
    """ Compute the slaving matrix """

    nActs = len(coords[0,:])

    slaved_m2c = xp.zeros([nActs,nActs],dtype=xp.float)
    for master in master_ids:
        slaved_m2c[master,master] = 1

    act_ids = xp.arange(nActs)
    is_slave = xp.ones(nActs, dtype=bool)
    is_slave[master_ids] = 0
    slave_ids = act_ids[is_slave]

    master_coords = coords[:,master_ids]
    dist = lambda xy: xp.sqrt((xy[0]-master_coords[0])**2 + (xy[1]-master_coords[1])**2)

    match slaving_method:

        case 'zero':
            pass

        case 'wmean':
            for slave in slave_ids:
                d_slave = dist(coords[:,slave])**p
                masters = master_ids[d_slave <= d_thr]
                d_slave = d_slave[d_slave <= d_thr]
                slaved_m2c[slave,masters] = 1/d_slave/xp.sum(1/d_slave)

        case 'nearest':
            for slave in slave_ids:
                d_slave = dist(coords[:,slave])
                nearest_master = master_ids[xp.argmin(d_slave)]
                slaved_m2c[slave,nearest_master] = 1
        case _:
            raise NotImplementedError(f"{slaving_method} is not an available slaving method.\
                                       Available methods are: 'zero', 'nearest', 'wmean', 'w2mean'")
        
    slaved_m2c = slaved_m2c[:,master_ids]

    return slaved_m2c
    

# def slaving(coords, cmd, slaving_method:str = 'wmean', cmd_thr:float = None, xp=np):
#     """ 
#     Clip the command to avoid saturation

#     Parameters
#     ----------
#     coords : ndarray(float) [2,Nacts]
#         The actuator coordinates

#     cmd : ndarray(float) [Nacts]
#         The mirror command to be slaved

#     cmd_thr : float, optional
#         Threshold to define slave actuators.
#         Slave acts are defined as the actuators for which: cmd > cmd_thr.
#         Default: slave acts the ones outside 3 sigma of the mean cmd
        
#     slaving_method : str, optional
#         The way to treat slave actuators.
#             'tps'     : thin plate spline interpolation. DEFAULT
#             'zero'    : set slave actuator command to zero 
#             'clip'    : clips the actuator commands to the given cmd_thr input
#             'nearest' : nearest actuators interpolation
#             'wmean'   : mean of nearby actuators weighted on 1/r^2

#             'exclude' : exclude slaves from the reconstructor computation (TBI)

#     """
#     # Define master ids
#     n_acts = len(cmd)
#     act_ids = xp.arange(n_acts)

#     if cmd_thr is None:
#         master_ids = act_ids[abs(cmd-xp.mean(cmd)) <= 3*xp.std(cmd)]
#     else:
#         master_ids = act_ids[abs(cmd) <= cmd_thr]
    
#     match slaving_method:

#         case 'tps':
#             tps = ThinPlateSpline(alpha=0.0)
#             tps.fit(coords[master_ids], cmd[master_ids])
#             rescaled_cmd = tps.transform(coords)
#             slaved_cmd = rescaled_cmd[:,0]

#         case 'zero':
#             pad_cmd = xp.zeros_like(cmd)
#             pad_cmd[master_ids] = cmd[master_ids]
#             slaved_cmd = pad_cmd

#         case 'clip':
#             slaved_cmd = xp.minimum(abs(cmd), cmd_thr)
#             slaved_cmd *= xp.sign(cmd)

#         case 'nearest':
#             master_coords = coords[:,master_ids]
#             slaved_cmd = griddata(master_coords, cmd[master_ids], (coords[0], coords[1]), method='nearest')

#         case 'wmean':
#             master_coords = coords[:,master_ids]
#             master_cmd = cmd[master_ids]
#             dist2 = lambda xy: (xy[0]-master_coords[0])**2 + (xy[1]-master_coords[1])**2 
#             is_slave = xp.ones_like(cmd, dtype=bool)
#             is_slave[master_ids] = False
#             slave_ids = act_ids[is_slave]
#             slaved_cmd = cmd.copy()
#             for slave in slave_ids:
#                 d2_slave = dist2(coords[:,slave])
#                 weighted_cmd = master_cmd / d2_slave
#                 slaved_cmd[slave] = xp.sum(weighted_cmd)*xp.sum(d2_slave)/n_acts

#         # case 'exclude':
#         #     masked_IFF = self.IFF[valid_ids,:]
#         #     masked_IFF = masked_IFF[:,visible_acts]
#         #     masked_R = np.linalg.pinv(masked_IFF)
            
#         #     pad_cmd = np.zeros_like(act_cmd)
#         #     pad_cmd[visible_acts] = matmul(masked_R, masked_shape)
#         #     act_cmd = pad_cmd
            
#         case _:
#             raise NotImplementedError(f"{slaving_method} is not an available slaving method. Available methods are: 'tps', 'zer', 'clip', 'nearest', 'wmean'")

#     return slaved_cmd



# def simulate_influence_functions_with_multiprocessing(act_coords, local_mask, pix_scale:float = 1.0, xp=np):
#     """ Simulate the influence functions by 
#     imposing 'perfect' zonal commands """
    
#     if xp.__name__ == 'cupy':
#         local_mask = local_mask.get()
#         act_coords = act_coords.get()

#     n_acts = max(act_coords.shape)

#     mask_ids = np.arange(np.size(local_mask))
#     pix_ids = mask_ids[~(local_mask).flatten()]
    
#     pix_coords = getMaskPixelCoords(local_mask).T
#     act_pix_coords = get_pixel_coords(local_mask, act_coords, pix_scale).T

#     n_cores = multiprocessing.cpu_count() -2
#     print(f'Simulating IFFs on {n_cores:1.0f} cores...')

#     def get_act_iff(act_id):
#         print(f'\rSimulating influence functions: {act_id}/{n_acts}', end='')
#         act_data = np.zeros(n_acts)
#         act_data[act_id] = 1e-6
#         tps = ThinPlateSpline(alpha=0.0)
#         tps.fit(act_pix_coords, act_data)
#         img = tps.transform(pix_coords[pix_ids,:])
#         return img[:,0]
    
#     with multiprocessing.Pool(processes=n_cores) as pool:
#         iffs = [pool.apply_async(get_act_iff, j) for j in range(n_acts)]
#         IFF = np.stack([iff.get(timeout=10) for iff in iffs])

#     print(IFF)
#     print(IFF.shape)

#     if xp.__name__ == 'cupy': # tps seems to only work with numpy
#         IFF = xp.asarray(IFF,dtype=xp.float32)

#     return IFF
