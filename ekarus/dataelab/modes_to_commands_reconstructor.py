"""Modes-to-commands reconstructor for Ekarus"""

import astropy.units as u
from arte.dataelab.base_data import BaseData
from ekarus.dataelab.ekarus_telemetry_keys import EkarusTelemetryKeys
from ekarus.dataelab.numpy_loader import NumpyDataLoader


class ModesToCommandsReconstructor(BaseData):
    """Modes-to-commands reconstructor (m2c) for Ekarus
    
    Loads 'm2c' matrix from the telemetry file.
    Expected shape: (nacts, nmodes)
    """

    M2C_KEY = 'm2c'

    def __init__(self, filename):
        """Initialize m2c reconstructor loader
        
        Parameters
        ----------
        filename : str
            Path to telemetry .npy or .npz file
        """
        super().__init__(data=NumpyDataLoader(filename, key=EkarusTelemetryKeys.M2C_KEY),
                         astropy_unit=u.volt / u.dimensionless_unscaled,
                         axes=('actuators', 'modes'))
