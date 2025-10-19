"""Signal-to-modes reconstructor for Papyrus"""

import astropy.units as u
from arte.dataelab.base_data import BaseData
from ekarus.dataelab.numpy_loader import NumpyDataLoader


class SignalToModesReconstructor(BaseData):
    """Signal-to-modes reconstructor (s2m) for Papyrus
    
    Loads 's2m' matrix from the telemetry file.
    Expected shape: (nmodes, nsignals)
    """

    S2M_KEY = 's2m'

    def __init__(self, filename):
        """Initialize s2m reconstructor loader
        
        Parameters
        ----------
        filename : str
            Path to telemetry .npy or .npz file
        """
        super().__init__(data=NumpyDataLoader(filename, key=SignalToModesReconstructor.S2M_KEY),
                         astropy_unit=u.dimensionless_unscaled,
                         axes=('modes', 'signals'))
