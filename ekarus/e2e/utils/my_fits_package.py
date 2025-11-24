from astropy.io import fits as pyfits
# from numpy import uint8
from numpy.ma import masked_array
import xupy as xp
import numpy as np


def read_fits(filename, isBool:bool=False, return_header:bool=False):
    """
    Loads a FITS file.

    Parameters
    ----------
    filename : str
        Path to the FITS file.

    Returns
    -------
    data_out : np.ndarray or np.ma.MaskedArray
        FITS file data.
    """

    hdu = pyfits.open(filename)
    data_out = hdu[0].data

    if isBool is True:
        data_out = (data_out).astype(bool) 

    if len(hdu) > 1 and hasattr(hdu[1], "data"):
        mask = hdu[1].data.astype(bool)
        data_out = masked_array(data_out, mask=mask)
    #     print(mask.shape)
    #     data_out = xp.asmarray(data_out)
    else:
        data_out = xp.asarray(data_out)
    
    if return_header:
        return data_out, hdu[0].header
    else:
        return data_out


def save_fits(filename, datain, overwrite:bool=True, header_dict=None):
    """
    Saves a FITS file.

    Parameters
    ----------
    filepath : str
        Path to the FITS file.
    datain : xp.array
        Data to be saved.
    overwrite : bool, optional
        Whether to overwrite an existing file. Default is True.
    header_dict : dict[str, any] 
        Header dictionary to include in the FITS file.
    """

    hdr = pyfits.Header()
    if header_dict is not None:
        for key in header_dict:
            hdr[str(key)] = float(header_dict[key])

    if hasattr(datain, 'get'):
        datain = datain.get()
    
    if hasattr(datain, "mask"):
        pyfits.writeto(filename, datain.data, hdr, overwrite=overwrite)
        pyfits.append(filename, np.array(datain.mask).astype(np.uint8))
    else:
        pyfits.writeto(filename, datain, hdr, overwrite=overwrite)

# def read_fits(
#     filepath: str, return_header: bool = False
# ):
#     """
#     Loads a FITS file.

#     Parameters
#     ----------
#     filepath : str
#         Path to the FITS file.
#     return_header: bool
#         Wether to return the header of the loaded fits file. Default is False.

#     Returns
#     -------
#     fit : np.ndarray or np.ma.MaskedArray
#         FITS file data.
#     header : dict | fits.Header, optional
#         The header of the loaded fits file.
#     """
#     with _fits.open(filepath) as hdul:
#         fit = hdul[0].data
#         if len(hdul) > 1 and hasattr(hdul[1], "data"):
#             mask = hdul[1].data.astype(bool)
#             fit = xp.masked_array(fit, mask=mask)
#         else:
#             fit = xp.asarray(fit)
#         if return_header:
#             header = hdul[0].header
#             return fit, header
#     return fit


# def save_fits(
#     filepath: str,
#     data,
#     overwrite: bool = True,
#     header: _fits.Header = None,
# ) -> None:
#     """
#     Saves a FITS file.

#     Parameters
#     ----------
#     filepath : str
#         Path to the FITS file.
#     data : np.array
#         Data to be saved.
#     overwrite : bool, optional
#         Whether to overwrite an existing file. Default is True.
#     header : dict[str, any] | fits.Header, optional
#         Header information to include in the FITS file. Can be a dictionary or
#         a fits.Header object.
#     """
#     # Prepare the header
#     if header is not None:
#         header = _header_from_dict(header)
#     # Save the FITS file
#     if xp.on_gpu:
#         if hasattr(data, 'asmarray'):
#             data = data.asmarray()
#         else:
#             data = xp.asnumpy(data)
#     if isinstance(data, xp.MaskedArray):
#         _fits.writeto(filepath, data.data, header=header, overwrite=overwrite)
#         if hasattr(data, "mask"):
#             _fits.append(filepath, data.mask.astype(uint8))
#     else:
#         _fits.writeto(filepath, data, header=header, overwrite=overwrite)


# def _header_from_dict(
#     dictheader,
# ) -> _fits.Header:
#     """
#     Converts a dictionary to an astropy.io.fits.Header object.

#     Parameters
#     ----------
#     dictheader : dict
#         Dictionary containing header information. Each key should be a string,
#         and the value can be a tuple of length 2, where the first element is the
#         value and the second is a comment.

#     Returns
#     -------
#     header : astropy.io.fits.Header
#         The converted FITS header object.
#     """
#     if isinstance(dictheader, _fits.Header):
#         return dictheader
#     header = _fits.Header()
#     for key, value in dictheader.items():
#         if isinstance(value, tuple) and len(value) > 2:
#             raise ValueError(
#                 "Header values must be a tuple of length 2 or less, "
#                 "where the first element is the value and the second is the comment."
#                 f"{value}"
#             )
#         else:
#             header[key] = value
#     return header



    
