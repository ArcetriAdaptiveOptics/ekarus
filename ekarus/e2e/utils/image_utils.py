import numpy as np
# from arte.types.mask import CircularMask
import matplotlib.pyplot as plt
from numpy.ma import masked_array



def image_grid(shape, recenter:bool = False, xp=np):
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
    dtype = xp.float32 if xp.__name__ == 'cupy' else xp.float64

    cy, cx = (0,0)
    if recenter:
        cy, cx = ny//2, nx//2

    x = xp.arange(nx, dtype=dtype) - cx
    y = xp.arange(ny, dtype=dtype) - cy
    X,Y = xp.meshgrid(x, y)

    return X,Y


def get_photocenter(image, xp=np):
    """ 
    Compute the image photocenter
    
    :param image: 2D array intensity on which to compute the photocenter
    :param xp: (optional) numpy or cupy for GPU acceleration
    :return: y,x coordinates of the photocenter
    """
    X,Y = image_grid(image.shape, xp=xp)
    qy = xp.sum(Y * image) / xp.sum(image)
    qx = xp.sum(X * image) / xp.sum(image)

    qy += 0.5
    qx += 0.5

    return qx,qy 


def get_circular_mask(mask_shape, mask_radius, mask_center=None, xp=np):
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
    mask = xp.asarray(mask,dtype=bool)  #mask.astype(bool)
    return mask

    # mask = CircularMask(shape, maskRadius=radius, maskCenter=center)
    # return (mask.mask()).astype(bool)


def reshape_on_mask(vec, mask, xp=np, dtype=None):
    """
    Reshape a given array on a 2D mask.
    :param flat_array: array of shape sum(1-mask)
    :param mask: boolean 2D mask
    :return: 2D array with flat_array in ~mask
    """
    if dtype is None:
        dtype = xp.float32
    image = xp.zeros(mask.shape,dtype=dtype)
    image[~mask] = vec
    image = xp.reshape(image, mask.shape)
    return image


def get_masked_array(vec, mask):
    if hasattr(vec, 'get'):
        vec = vec.get()
    if hasattr(mask, 'get'):
        mask = mask.get() 
    aux = reshape_on_mask(vec, mask)
    ma_vec = masked_array(aux, mask)
    return ma_vec


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
    if hasattr(image,'get'):
        image = image.get()
    plt.imshow(image,origin='lower', **kwargs)
    cbar = plt.colorbar(shrink=shrink)
    cbar.set_label(cbar_title,loc='top')
    plt.title(title)
