from astropy.io import fits as pyfits
from numpy import uint8
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


def save_fits(filename, data, overwrite:bool=True, header_dictionary=None):

    hdr = pyfits.Header()
    if header_dictionary is not None:
        for key in header_dictionary:
            hdr[str(key)] = float(header_dictionary[key])

    if hasattr(data, 'get'):
        data = data.get()
    
    pyfits.writeto(filename, data, hdr, overwrite=overwrite)
    
    if hasattr(data, "mask"):
        pyfits.append(filename, data.mask.astype(uint8))
