"""File walker for Papyrus telemetry data"""

from pathlib import Path
from arte.dataelab.base_file_walker import AbstractFileNameWalker


class PapyrusFileWalker(AbstractFileNameWalker):
    """File walker for Papyrus telemetry and calibration files
    
    Papyrus stores telemetry in a single .npy or .npz file containing
    all data streams (ocam, modes, DM commands, calibrations, etc.)
    """

    def __init__(self, telemetry_file):
        """Initialize file walker
        
        Parameters
        ----------
        telemetry_file : str or Path
            Path to the telemetry .npy or .npz file
        """
        self._telemetry_file = Path(telemetry_file)

    def walker_root_dir(self):
        """Return parent directory of telemetry file"""
        return self._telemetry_file.parent

    def telemetry_file(self):
        """Return path to telemetry file"""
        return self._telemetry_file

    def snapshot_dir(self, tag):
        """Return snapshot directory (same as telemetry file parent)"""
        return self.walker_root_dir()
