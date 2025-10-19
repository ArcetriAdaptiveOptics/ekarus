"""Deformable mirror commands for Papyrus"""

from astropy import units as u
from arte.dataelab.base_timeseries import BaseTimeSeries
from ekarus.dataelab.numpy_loader import NumpyDataLoader
import numpy as np
import datetime


class DMCommands(BaseTimeSeries):
    """Deformable mirror commands from Papyrus telemetry
    
    Loads 'dmCmdCube' from the telemetry file.
    Expected shape: (time, nacts) (squeezes trailing singleton axes)
    """

    DM_CMD_CUBE_KEY = 'dmCmdCube'
    TIMESTAMP_KEY = 'timeStampDmCube'
    COUNTER_KEY = 'dmCmdCounter'

    def __init__(self, filename):
        """Initialize DM commands loader
        
        Parameters
        ----------
        filename : str
            Path to telemetry .npy or .npz file
        """
        data_loader = NumpyDataLoader(
            filename, key=DMCommands.DM_CMD_CUBE_KEY)

        # remove singleton trailing axes (e.g. (T, N, 1) -> (T, N))
        data_loader._postprocess = lambda d: np.squeeze(d)

        try:
            time_loader = NumpyDataLoader(
                filename, key=DMCommands.TIMESTAMP_KEY)

            def _time_postproc(tv):
                tv = np.asarray(tv)
                # numpy datetime64
                if tv.dtype.kind == 'M':
                    secs = tv.astype('datetime64[ns]').astype('int64') / 1e9
                    return secs - secs[0]
                # object array with datetime.datetime
                if tv.dtype == object and tv.size > 0 and isinstance(tv.flat[0], datetime.datetime):
                    base = tv.flat[0]
                    return np.array([(t - base).total_seconds() for t in tv])
                # object array with timedelta
                if tv.dtype == object and tv.size > 0 and isinstance(tv.flat[0], datetime.timedelta):
                    return np.array([float(t.total_seconds()) for t in tv])
                # numeric already (seconds or indices) -> return as float array
                return tv.astype(float)

            time_loader._postprocess = _time_postproc

        except (KeyError, ValueError):
            try:
                time_loader = NumpyDataLoader(
                    filename, key=DMCommands.COUNTER_KEY)
            except (KeyError, ValueError):
                time_loader = None

        super().__init__(data_loader, time_vector=time_loader,
                         astropy_unit=u.volt,
                         axes=('time', 'actuators'),
                         data_label='DM commands')

    def get_display_axes(self):
        return ('time', 'actuators')
