import numpy as np
from arte.types.mask import CircularMask


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


def get_circular_mask(shape, radius, center=None):
    """
    Create a circular mask for the given shape.
    
    :param shape: tuple (ny, nx) dimensions of the mask
    :param radius: radius of the circular mask
    :param center: tuple (cy, cx) center of the circular mask
    :return: boolean numpy array with the mask
    """
    mask = CircularMask(shape, maskRadius=radius, maskCenter=center)
    return (mask.mask()).astype(bool)


def compute_pixel_size(wavelength, pupil_diameter_in_m, padding:int=1):
    """ Get the number of pixels per radian """
    return wavelength/pupil_diameter_in_m/padding