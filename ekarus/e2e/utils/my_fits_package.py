from astropy.io import fits as pyfits

def read_fits(filename, isBool:bool = False):

    hdu = pyfits.open(filename)
    data_out = hdu[0].data

    if isBool:
        data_out = (data_out).astype(bool) 
    
    return data_out


def save_fits(filename, data, header_dictionary = None):

    hdr = pyfits.Header()
    if header_dictionary is not None:
        for key in header_dictionary:
            hdr[str(key)] = header_dictionary[key]
    
    pyfits.writeto(filename, data, hdr, overwrite=True)



# import os

# class FolderTree():

#     def __init__(self):
#         self.cwd = os.getcwd()


#     def make_directory(self, dir_path):
#         full_dir_path = os.path.join(self.cwd, dir_path)
#         try:
#             os.mkdir(full_dir_path)
#         except FileExistsError:
#             pass
    

#     def get_full_path(self, file_path):
#         return os.path.join(self.cwd, file_path)