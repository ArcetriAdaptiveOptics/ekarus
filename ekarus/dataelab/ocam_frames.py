"""Ocam camera frames for Papyrus"""

from astropy import units as u
from arte.dataelab.base_timeseries import BaseTimeSeries
from ekarus.dataelab.numpy_loader import NumpyDataLoader


class OcamFrames(BaseTimeSeries):
    """Ocam camera frames from Papyrus telemetry
    
    Loads 'ocamCube' from the telemetry file.
    Expected shape: (time, y, x)
    """

    OCAM_CUBE_KEY = 'ocamCube'
    TIMESTAMP_KEY = 'timeStampOcamCube'
    COUNTER_KEY = 'ocamCounter'

    def __init__(self, filename):
        """Initialize Ocam frames loader
        
        Parameters
        ----------
        filename : str
            Path to telemetry .npy or .npz file
        """
        data_loader = NumpyDataLoader(
            filename, key=OcamFrames.OCAM_CUBE_KEY)

        try:
            time_vector = NumpyDataLoader(
                filename, key=OcamFrames.TIMESTAMP_KEY)
        except (KeyError, ValueError):
            # If timestamps not available, use counter
            try:
                time_vector = NumpyDataLoader(
                    filename, key=OcamFrames.COUNTER_KEY)
            except (KeyError, ValueError):
                # If no time info, use frame indices
                time_vector = None

        super().__init__(data_loader, time_vector=time_vector,
                         astropy_unit=u.dimensionless_unscaled,
                         axes=('time', 'y', 'x'),
                         data_label='Ocam frames')

    def get_display_axes(self):
        return ('time', 'pixels', 'pixels')
