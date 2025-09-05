from astropy.io import fits as pyfits

def read_fits(filename, isBool:bool = False):

    hdu = pyfits.open(filename)
    data_out = hdu[0].data

    # if xp is not None:
    #     dtype = xp.float32 if xp.__name__ == 'cupy' else xp.float64
    #     data_out = xp.asarray(data_out, dtype=dtype)

    if isBool is True:
        data_out = (data_out).astype(bool) 
    
    return data_out


def save_fits(filename, data, header_dictionary = None):

    hdr = pyfits.Header()
    if header_dictionary is not None:
        for key in header_dictionary:
            hdr[str(key)] = float(header_dictionary[key])

    if hasattr(data, 'get'):
        data = data.get()
    
    pyfits.writeto(filename, data, hdr, overwrite=True)