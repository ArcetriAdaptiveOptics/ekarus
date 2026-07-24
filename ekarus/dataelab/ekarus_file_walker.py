"""File walker for Ekarus telemetry data"""

from pathlib import Path
from typing import Optional, Union
from arte.dataelab.base_file_walker import AbstractFileNameWalker


class EkarusFileWalker(AbstractFileNameWalker):
    """File walker for Ekarus telemetry and calibration files.

    The walker may be constructed with either a concrete telemetry file
    path (used as-is), or with a `root_dir` that points to the directory
    containing the `Telemetry snapshots` tree. When a `tag` is provided to
    `telemetry_file(tag)`, the walker will resolve the expected filename
    under ``root_dir / 'Telemetry snapshots' / YYYY-MM-DD / <file>``.
    """

    def __init__(self, root_dir: Optional[Union[str, Path]] = None):
        """Initialize file walker.

        Parameters
        ----------
        root_dir : str | Path
            Root directory under which `Telemetry snapshots` are stored.
            This walker assumes `root_dir` is always provided.
        """
        if root_dir is None:
            raise ValueError('EkarusFileWalker requires root_dir')
        self._root_dir = Path(root_dir)

    def walker_root_dir(self) -> Path:
        """Return the configured root directory for snapshots.

        If the walker was constructed with a concrete telemetry file, the
        parent directory of that file is returned. Otherwise the configured
        `root_dir` is returned if present.
        """
        return self._root_dir

    def telemetry_file(self, tag: Optional[Union[str, Path]] = None) -> Path:
        """Return the telemetry file path.

        If `tag` is provided (a snapshot tag like 'YYYYMMDD_HHMMSS' or a
        path-like string), the walker will resolve it relative to the
        configured `root_dir`. If no `tag` is given and the walker was
        constructed with a concrete telemetry file, that path is returned.
        """
        if tag is None:
            raise ValueError('tag is required')

        p = Path(str(tag))
        if p.exists() and p.is_file():
            return p

        base = self._root_dir

        tag_str = str(tag)
        # Parse tag-like strings: 'YYYYMMDD_HHMMSS' or 'YYYYMMDD/HHMMSS'
        if ('_' in tag_str and tag_str.count('_') >= 1) or ('/' in tag_str and tag_str.count('/') >= 1):
            if '/' in tag_str:
                day_str, time_str = tag_str.split('/', 1)
            else:
                day_str, time_str = tag_str.split('_', 1)

            if len(day_str) >= 8:
                y, m, d = day_str[:4], day_str[4:6], day_str[6:8]
                date_dir = f"{y}-{m}-{d}"
            else:
                raise ValueError(f"Cannot parse day from tag '{tag_str}'")

            time_clean = time_str.replace('_', '')
            if len(time_clean) >= 6:
                hh, mm, ss = time_clean[:2], time_clean[2:4], time_clean[4:6]
                time_name = f"{hh}_{mm}_{ss}"
            else:
                raise ValueError(f"Cannot parse time from tag '{tag_str}'")

            filename = f"{date_dir}T{time_name}_telemetry_data.npy"
            return base / 'Telemetry snapshots' / date_dir / filename

        # Fallback: return as Path
        return p

    def snapshot_dir(self, tag: Union[str, Path]) -> Path:
        """Return snapshot directory for a given tag.

        If `tag` is a date-like tag, returns the directory under
        `root_dir / 'Telemetry snapshots' / YYYY-MM-DD`.
        """
        tag_str = str(tag)
        if '/' in tag_str:
            day_str = tag_str.split('/', 1)[0]
        else:
            day_str = tag_str.split('_', 1)[0]

        if len(day_str) >= 8:
            y, m, d = day_str[:4], day_str[4:6], day_str[6:8]
            date_dir = f"{y}-{m}-{d}"
            return self._root_dir / 'Telemetry snapshots' / date_dir

        return self._root_dir

