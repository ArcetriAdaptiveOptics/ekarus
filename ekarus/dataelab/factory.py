"""Factory helpers to configure ekarus.dataelab runtime environment.

Primary goal: allow tests and users to set a `data_root_dir` where
telemetry files are searched. The factory sets an environment variable
`EKARUS_DATA_ROOT` and provides convenience helpers for creating
analyzers and file walkers tied to that root.

Keep this module small — it's focused on data_root_dir customization.
"""

from pathlib import Path
import logging
from typing import Optional


class Factory:
    """Base factory that sets `EKARUS_DATA_ROOT` and configures logging.

    Parameters
    ----------
    data_root_dir : str | Path | None
        If provided, used as the data root. Otherwise a sensible default
        (current working directory /data) is used.
    log_level : int | str | None
        Optional logging level override.
    """

    def __init__(self, data_root_dir: Optional[Path | str] = None, log_level: Optional[str | int] = None):
        if data_root_dir is None:
            self._data_root = Path.cwd()
        else:
            self._data_root = Path(data_root_dir)
        self._setup_logging(log_level)

    def data_root_dir(self) -> Path:
        """Return the configured data root directory (Path).

        This factory does not export the path to environment variables by
        default; it keeps the configured path internal. If you need a
        global setter for other packages, call that package's setter
        explicitly.
        """
        return Path(self._data_root)

    def _setup_logging(self, log_level: Optional[str | int] = None):
        if log_level is None:
            level = logging.INFO
        else:
            level = getattr(logging, str(log_level).upper(), logging.INFO) if isinstance(log_level, str) else int(log_level)

        root = logging.getLogger()
        if not root.handlers:
            logging.basicConfig(level=level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        root.setLevel(level)

        self._logger = logging.getLogger(self.__class__.__name__)
        self._logger.info('Factory initialized: %s', self.__class__.__name__)
        self._logger.info('Data root: %s', self.data_root_dir())

    def file_walker(self):
        """Return an EkarusFileWalker rooted at `data_root_dir()`."""
        from ekarus.dataelab.ekarus_file_walker import EkarusFileWalker
        return EkarusFileWalker(root_dir=self.data_root_dir())

    def analyzer(self, tag, *args, **kwargs):
        """Create an `EkarusAnalyzer` for a given telemetry tag or full path.

        If `tag` looks like a snapshot tag (e.g. '20260723_144418' or a
        `arte.dataelab.tag.Tag` instance), the factory will resolve it into
        the telemetry file located under `data_root_dir()/Telemetry snapshots/...`.
        Otherwise `tag` is passed through as-is to `EkarusAnalyzer`.
        """
        from ekarus.dataelab.ekarus_analyzer import EkarusAnalyzer

        # Build a file walker rooted at the factory data root and pass it
        # to the analyzer so it can resolve tag -> telemetry file.
        fw = self.file_walker()
        return EkarusAnalyzer(tag, *args, file_walker=fw, **kwargs)

        


class FactoryDefault(Factory):
    """Default factory using `cwd/data` as data root."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class FactoryLaptopLorenzo(Factory):
    """Factory for a user-provided local data root."""

    def __init__(self, **kwargs):
        data_root_dir=Path(
            '/Users/lbusoni/Library/CloudStorage/GoogleDrive-lorenzo.busoni@inaf.it/Drive condivisi/EKARUS/Ekarus public folder/Data/')
        super().__init__(data_root_dir=data_root_dir, **kwargs)


class FactoryEkar(Factory):
    """Factory for lab configuration. Override `data_root_dir` with lab path."""

    def __init__(self, lab_data_root: Optional[Path | str] = None, **kwargs):
        # default lab path can be overridden by caller
        default = lab_data_root or Path('/home/daouser/DAODATA/telemetry_snapshots_or_something_similar')
        super().__init__(data_root_dir=default, **kwargs)


def get_factory(config: str = 'default', **kwargs) -> Factory:
    """Convenience to obtain a factory instance.

    config options:
    - 'default' : FactoryDefault
    - 'local'   : FactoryLocal (requires `data_root_dir` kwarg)
    - 'ekar'    : FactoryEkar (optional `lab_data_root` kwarg)
    """
    mapping = {
        'default': FactoryDefault,
        'ekar': FactoryEkar,
        'lorenzo': FactoryLaptopLorenzo,
    }

    if config not in mapping:
        raise ValueError(f'Unknown factory config: {config}. Available: {list(mapping.keys())}')

    FactoryClass = mapping[config]
    return FactoryClass(**kwargs)
