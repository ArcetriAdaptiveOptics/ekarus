import xupy as xp
import numpy as np
# from arte.types.mask import CircularMask

import matplotlib.pyplot as plt



def image_grid(shape, recenter:bool = False):
    """
    Define a grid of X and Y coordinates on an image shape

    :param shape: tuple grid dimensions
    :param recenter: (optional) boolean to recenter 
    the coordinates wrt to the image center.
    Defaults to False
    :param xp: (optional) numpy or cupy for GPU acceleration
    :return: X,Y grid of coordinates
    """

    ny, nx = shape
    dtype = xp.float

    cy, cx = (0,0)
    if recenter:
        cy, cx = ny//2, nx//2

    x = xp.arange(nx, dtype=dtype) - cx
    y = xp.arange(ny, dtype=dtype) - cy
    X,Y = xp.meshgrid(x, y)

    return X,Y


def get_photocenter(image):
    """ 
    Compute the image photocenter
    
    :param image: 2D array intensity on which to compute the photocenter
    :param xp: (optional) numpy or cupy for GPU acceleration
    :return: y,x coordinates of the photocenter
    """
    X,Y = image_grid(image.shape)
    qy = xp.sum(Y * image) / xp.sum(image)
    qx = xp.sum(X * image) / xp.sum(image)

    qy += 0.5
    qx += 0.5

    return qx,qy 


def get_circular_mask(mask_shape, mask_radius, mask_center=None):
    """
    Create a circular mask for the given shape.
    :param shape: tuple (ny, nx) dimensions of the mask
    :param radius: radius of the circular mask
    :param center: tuple (cy, cx) center of the circular mask
    :return: boolean numpy array with the mask
    """
    H,W = mask_shape
    if mask_center is None:
        mask_center = (W/2,H/2)

    dist = lambda x,y: xp.sqrt((xp.asarray(x)-mask_center[0])**2+(xp.asarray(y)-mask_center[1])**2)
    mask = xp.fromfunction(lambda i,j: dist(j,i) >= mask_radius, [H,W])
    mask = xp.asarray(mask,dtype=bool)
    return mask


def reshape_on_mask(vec, mask):
    """
    Reshape a given array on a 2D mask.
    :param flat_array: array of shape sum(1-mask)
    :param mask: boolean 2D mask
    :return: 2D array with flat_array in ~mask
    """
    image = xp.zeros(mask.shape, dtype=xp.float)
    image[~mask] = vec
    image = xp.reshape(image, mask.shape)
    return image


def bilinear_interp(data_2D, full_mask, xy_shift:tuple, min_shift:float=1e-8):


    dx, dy = abs(xy_shift[0]), abs(xy_shift[1])
    sdx, sdy = int(xp.sign(xy_shift[0])), int(xp.sign(xy_shift[1]))

    dx_int = int(xp.floor(dx) * (sdx>0) + xp.ceil(dx) * (sdx<0))
    dy_int = int(xp.floor(dy) * (sdy>0) + xp.ceil(dy) * (sdy<0))

    shifted_mask = xp.roll(full_mask,(dy_int*sdy,dx_int*sdx),axis=(0,1))
    data = data_2D[~shifted_mask]
    shifted_data = reshape_on_mask(data, shifted_mask)

    dx -= dx_int
    dy -= dy_int

    if dx > min_shift and dy > min_shift:
        dx_data = reshape_on_mask(shifted_data[~xp.roll(shifted_mask,sdx,axis=1)], shifted_mask)
        dy_data = reshape_on_mask(shifted_data[~xp.roll(shifted_mask,sdy,axis=0)], shifted_mask)
        dxdy_data = reshape_on_mask(shifted_data[~xp.roll(shifted_mask,(sdy,sdx),axis=(0,1))], shifted_mask)
        interp_data = (shifted_data * (1-dx) + dx * dx_data) * (1-dy) + (dy_data * (1-dx) + dx * dxdy_data) * dy

    elif dx > min_shift:
        dx_data = reshape_on_mask(shifted_data[~xp.roll(shifted_mask,sdx,axis=1)], shifted_mask)
        interp_data = shifted_data * (1-dx) + dx_data * dx

    elif dy > min_shift:
        dy_data = reshape_on_mask(shifted_data[~xp.roll(shifted_mask,sdy,axis=0)], shifted_mask)
        interp_data = shifted_data * (1-dy) + dy_data * dy

    else:
        interp_data = shifted_data.copy()

    return interp_data


# def get_masked_array(vec, mask):
#     vec2D = reshape_on_mask(vec, mask)
#     if hasattr(vec2D, 'get'):
#         vec2D = vec2D.get()
#     if hasattr(mask, 'get'):
#         mask = mask.get() 
#     ma_vec = np.ma.masked_array(vec2D, mask)
#     return ma_vec


def imageShow(image2d, pixelSize=1, title='', xlabel='', ylabel='', zlabel='', shrink=1.0, **kwargs):
    sz=image2d.shape
    plt.imshow(image2d, extent=[-sz[0]/2*pixelSize, sz[0]/2*pixelSize,
                                -sz[1]/2*pixelSize, sz[1]/2*pixelSize],origin='lower', **kwargs)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    cbar= plt.colorbar(shrink=shrink)
    cbar.ax.set_ylabel(zlabel)

def showZoomCenter(image, pixelSize, **kwargs):
    '''show log(image) zoomed around center'''    
    if hasattr(image,'get'):
        image = image.get()
    imageHalfSizeInPoints= image.shape[0]/2
    roi= [int(imageHalfSizeInPoints*0.9), int(imageHalfSizeInPoints*1.1)]
    imageZoomedLog= np.log(image[roi[0]: roi[1], roi[0]:roi[1]])
    imageShow(imageZoomedLog, pixelSize=pixelSize, **kwargs)

def myimshow(image, title='', cbar_title='', shrink=1.0, **kwargs):
    if hasattr(image, 'asmarray'):
        image = image.asmarray()
    if hasattr(image,'get'):
        image = image.get()
    plt.imshow(image,origin='lower', **kwargs)
    cbar = plt.colorbar(shrink=shrink)
    cbar.set_label(cbar_title,loc='top')
    plt.title(title)


