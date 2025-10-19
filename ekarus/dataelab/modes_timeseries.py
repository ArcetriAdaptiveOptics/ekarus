"""Modal coefficients timeseries for Papyrus"""

from astropy import units as u
from arte.dataelab.base_timeseries import BaseTimeSeries
from ekarus.dataelab.numpy_loader import NumpyDataLoader
import numpy as np
import datetime


class ResidualModes(BaseTimeSeries):
    """Residual modal coefficients timeseries for Papyrus

    Loads 'modeCube' from the telemetry file.
    Expected shape on disk: (time, n_modes, 1) or (time, n_modes)
    This class squeezes singleton axes so returned data shape is (time, n_modes).
    """

    MODES_KEY = 'modeCube'
    TIMESTAMP_KEY = 'timeStampDmCube'

    def __init__(self, filename):
        data_loader = NumpyDataLoader(filename, key=ResidualModes.MODES_KEY)

        # ensure any singleton trailing axis is removed (e.g. shape (T, N, 1) -> (T, N))
        data_loader._postprocess = lambda d: np.squeeze(d)

        # time_vector optional: try to load and convert to seconds (float)
        try:
            time_loader = NumpyDataLoader(filename, key=ResidualModes.TIMESTAMP_KEY)

            def _time_postproc(tv):
                tv = np.asarray(tv)
                # numpy datetime64
                if tv.dtype.kind == 'M':
                    # convert to seconds since first sample
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
            time_loader = None

        super().__init__(data_loader,
                         time_vector=time_loader,
                         astropy_unit=u.dimensionless_unscaled,
                         axes=('time', 'modes'),
                         data_label='Residual modal coefficients')

    def get_display_axes(self):
        return ('time', 'modes')
