import numpy as np
# from arte.types.mask import CircularMask
import matplotlib.pyplot as plt

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

    cy, cx = (0,0)
    if recenter:
        cy, cx = ny//2, nx//2

    x = xp.arange(nx) - cx
    y = xp.arange(ny) - cy
    X,Y = xp.meshgrid(x, y)

    return X,Y


def get_photocenter(image, xp=np):
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

    return qy,qx


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

    dist = lambda x,y: xp.sqrt((x-mask_center[0])**2+(y-mask_center[1])**2)
    mask = xp.fromfunction(lambda i,j: dist(j,i) > mask_radius, [H,W])
    mask = mask.astype(bool)
    return mask
    # mask = CircularMask(shape, maskRadius=radius, maskCenter=center)
    # return (mask.mask()).astype(bool)


def reshape_on_mask(flat_array, mask, xp=np):
    """
    Reshape a given array on a 2D mask.
    :param flat_array: array of shape sum(1-mask)
    :param mask: boolean 2D mask
    :return: 2D array with flat_array in ~mask
    """
    dtype = xp.float32 if xp.__name__ == 'cupy' else xp.float64
    image = xp.zeros(mask.shape,dtype=dtype)
    image[~mask] = flat_array
    image = xp.reshape(image, mask.shape)
    return image


# def compute_pixel_size(wavelength, pupil_diameter_in_m, padding:int=1):
#     """ Get the number of pixels per radian """
#     return wavelength/pupil_diameter_in_m/padding


def imageShow(image2d, pixelSize=1, title='', xlabel='', ylabel='', zlabel='', shrink=1.0):
    sz=image2d.shape
    plt.imshow(image2d, extent=[-sz[0]/2*pixelSize, sz[0]/2*pixelSize,
                                -sz[1]/2*pixelSize, sz[1]/2*pixelSize],origin='lower')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    cbar= plt.colorbar(shrink=shrink)
    cbar.ax.set_ylabel(zlabel)

def showZoomCenter(image, pixelSize, **kwargs):
    '''show log(image) zoomed around center'''
    imageHalfSizeInPoints= image.shape[0]/2
    roi= [int(imageHalfSizeInPoints*0.9), int(imageHalfSizeInPoints*1.1)]
    imageZoomedLog= np.log(image[roi[0]: roi[1], roi[0]:roi[1]])
    imageShow(imageZoomedLog, pixelSize=pixelSize, **kwargs)