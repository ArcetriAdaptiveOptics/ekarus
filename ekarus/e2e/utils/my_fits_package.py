from astropy.io import fits as pyfits
import numpy as np
from numpy.ma import masked_array

def read_fits(filename, isBool:bool=False):

    hdu = pyfits.open(filename)
    data_out = hdu[0].data

    if isBool is True:
        data_out = (data_out).astype(bool) 

    if len(hdu) > 1 and hasattr(hdu[1], "data"):
        mask = hdu[1].data.astype(bool)
        data_out = masked_array(data_out, mask=mask)
        print(mask.shape)
    
    return data_out


def save_fits(filename, datain, overwrite:bool=True, header_dictionary=None):

    hdr = pyfits.Header()
    if header_dictionary is not None:
        for key in header_dictionary:
            hdr[str(key)] = float(header_dictionary[key])

    if hasattr(datain, 'get'):
        datain = datain.get()
    
    if hasattr(datain, "mask"):
        pyfits.writeto(filename, datain.data, hdr, overwrite=overwrite)
        pyfits.append(filename, np.array(datain.mask).astype(np.uint8))
    else:
        pyfits.writeto(filename, datain, hdr, overwrite=overwrite)
    
