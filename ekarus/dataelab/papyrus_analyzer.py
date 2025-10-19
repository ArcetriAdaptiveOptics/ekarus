"""Papyrus telemetry analyzer for arte.dataelab

This analyzer follows the KADAT pattern, decomposing telemetry data
into separate classes for each data stream (Ocam frames, modal coefficients,
DM commands, reconstructors).

Expected telemetry file keys:
- ocamCube: camera frames (time, y, x)
- modeCube: reconstructed modal coefficients (time, nmodes)
- dmCmdCube: DM commands (time, nacts)
- s2m: signal-to-modes reconstructor matrix (nmodes, nsignals)
- m2c: modes-to-commands matrix (nacts, nmodes)
- validPixels: boolean mask for valid pixels
- ocamDark: dark frame or scalar
- ref: reference frame or vector
- lpGain: loop gain (scalar or per-mode)
- lpLeak: loop leak (scalar or per-mode)
- timeStampOcamCube, timeStampDmCube: timestamps
- ocamCounter, dmCmdCounter: frame counters
"""

from arte.dataelab.base_analyzer import BaseAnalyzer
from arte.dataelab.analyzer_plots import modalplot

from ekarus.dataelab.papyrus_file_walker import PapyrusFileWalker
from ekarus.dataelab.ocam_frames import OcamFrames
from ekarus.dataelab.modes_timeseries import ResidualModes
from ekarus.dataelab.dm_commands import DMCommands
from ekarus.dataelab.signal_to_modes_reconstructor import SignalToModesReconstructor
from ekarus.dataelab.modes_to_commands_reconstructor import ModesToCommandsReconstructor


class PapyrusAnalyzer(BaseAnalyzer):
    """Main analyzer for Papyrus telemetry
    
    This analyzer provides access to:
    - Raw telemetry data (Ocam frames, modal coefficients, DM commands)
    - Reconstructor matrices (s2m, m2c)
    - Derived quantities and analysis methods
    
    Usage example:
    --------------
    >>> analyzer = PapyrusAnalyzer.get('/path/to/telemetry.npy')
    >>> 
    >>> # Access raw data
    >>> ocam_data = analyzer.ocam_frames.get_data()
    >>> modes_data = analyzer.modes.get_data()
    >>> dm_data = analyzer.dm_commands.get_data()
    >>> 
    >>> # Access specific time frames
    >>> frame_10 = analyzer.ocam_frames.get_data()[10]
    >>> modes_10 = analyzer.modes.get_data()[10]
    >>> 
    >>> # Get statistics
    >>> modes_std = analyzer.modes.time_std()
    >>> modes_mean = analyzer.modes.time_mean()
    """

    def __init__(self, tag, recalc=False):
        """Initialize Papyrus analyzer
        
        Parameters
        ----------
        tag : str
            Path to telemetry .npy or .npz file
        recalc : bool, optional
            If True, clear cached data and recompute
        """
        super().__init__(tag, recalc)
        
        # File walker
        fw = PapyrusFileWalker(tag)
        telemetry_file = str(fw.telemetry_file())
        
        # Raw telemetry data
        self.ocam_frames = OcamFrames(telemetry_file)
        self.residual_modes = ResidualModes(telemetry_file)
        self.dm_commands = DMCommands(telemetry_file)
        
        # Reconstructor matrices
        self.s2m = SignalToModesReconstructor(telemetry_file)
        self.m2c = ModesToCommandsReconstructor(telemetry_file)

    def _info(self):
        """Return analyzer information dictionary"""
        info = super()._info()
        try:
            info['nframes_ocam'] = len(self.ocam_frames.get_data())
        except Exception:
            info['nframes_ocam'] = 'N/A'
        try:
            info['nframes_residual_modes'] = len(self.residual_modes.get_data())
        except Exception:
            info['nframes_residual_modes'] = 'N/A'
        try:
            info['nframes_dm'] = len(self.dm_commands.get_data())
        except Exception:
            info['nframes_dm'] = 'N/A'
        return info



