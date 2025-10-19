import numpy as np
from arte.dataelab.data_loader import NumpyDataLoader as ArteNumpyDataLoader

class NumpyDataLoader(ArteNumpyDataLoader):
    """Wrapper around arte.dataelab.NumpyDataLoader that enables allow_pickle=True
    and is resilient to different on-disk layouts (npz, dict-in-npy, plain ndarray).
    """
    def load(self):
        # load with allow_pickle to support dict/object saved in .npy
        arr = np.load(self._filename, allow_pickle=True)

        if self._key:
            # npz-like file (NpzFile) supports key access
            if isinstance(arr, np.lib.npyio.NpzFile) or hasattr(arr, 'files'):
                data = arr[self._key]
            # plain dict saved (np.save with a dict + allow_pickle) -> dict access
            elif isinstance(arr, dict):
                data = arr[self._key]
            # ndarray: could be a 0-d object array containing a dict, or a structured array
            elif isinstance(arr, np.ndarray):
                # 0-d object array that wraps a dict (common when saving a dict with np.save)
                if arr.dtype == object and arr.shape == ():
                    obj = arr.item()
                    if isinstance(obj, dict):
                        data = obj[self._key]
                    else:
                        raise IndexError(f"File '{self._filename}' contains a 0-d object array but not a dict.")
                else:
                    # try to index (for structured arrays with field names)
                    try:
                        data = arr[self._key]
                    except Exception as e:
                        raise IndexError(f"Unable to extract key '{self._key}' from ndarray in '{self._filename}'") from e
            else:
                raise IndexError(f"Unsupported numpy.load result type: {type(arr)} for file '{self._filename}'")
        else:
            data = arr

        # mirror possible post-processing/transposition behaviour of the base loader
        transpose_attr = getattr(self, '_transpose_axes', None)
        if transpose_attr is not None:
            data = data.transpose(*transpose_attr)
        postprocess = getattr(self, '_postprocess', None)
        if postprocess:
            data = postprocess(data)

        return data