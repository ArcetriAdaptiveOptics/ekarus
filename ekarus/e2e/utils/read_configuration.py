import os.path as op
from .root import configpath
# from ruamel.yaml import YAML
import yaml

import xupy as xp
# import numpy as np


class ConfigReader():
    
    def __init__(self, tn: str):
        """ The constructor"""
        config_file = op.join(configpath, str(tn)+'.yaml')
        self._cfile = self._read_yaml(config_file)

    def read_telescope_pars(self):
        """ Read telescope parameters from the configuration file."""
        return self._cfile['TELESCOPE']

    def read_sensor_pars(self, sensor_name: str = 'WFS'):
        """ Read sensor parameters from the configuration file."""
        self._cfile[sensor_name]['lambdaInM'] = eval(self._cfile[sensor_name]['lambdaInM'])
        self._cfile[sensor_name]['bandWidthInM'] = eval(self._cfile[sensor_name]['bandWidthInM'])
        try:
            type = self._cfile[sensor_name]['type']
        except KeyError:
            self._cfile[sensor_name]['type'] = '4PWFS'
        return self._cfile[sensor_name]

    def read_detector_pars(self, detector_name: str = 'DETECTOR'):
        """ Read detector parameters from the configuration file."""
        return self._cfile[detector_name]
    
    def read_dm_pars(self, dm_name: str = 'DM'):
        """ Read deformable mirror parameters from the configuration file."""
        try:
            self._cfile[dm_name]['max_stroke_in_m'] = eval(self._cfile[dm_name]['max_stroke_in_m'])
        except KeyError:
            self._cfile[dm_name]['max_stroke_in_m'] = None
        return self._cfile[dm_name]
    
    def read_atmo_pars(self):
        """ Read atmosphere parameters from the configuration file."""
        try:
            cn2 = xp.array(self._cfile['ATMO']['cn2'])/100
            r0 = eval(self._cfile['ATMO']['r0'])
            self._cfile['ATMO']['r0'] = r0*cn2**(-3/5)
        except KeyError:
            self._cfile['ATMO']['r0'] = xp.array([eval(x) for x in self._cfile['ATMO']['r0']])
        self._cfile['ATMO']['windSpeed'] = xp.asarray(self._cfile['ATMO']['windSpeed'])
        self._cfile['ATMO']['windAngle'] = xp.asarray(self._cfile['ATMO']['windAngle'])*xp.pi/180
        return self._cfile['ATMO']
    
    def read_slope_computer_pars(self, slope_computer_name: str = 'SLOPE.COMPUTER'):
        """ Read slope computer parameters from the configuration file."""
        self._cfile[slope_computer_name]['integratorGain'] = xp.array(self._cfile[slope_computer_name]['integratorGain'])
        self._cfile[slope_computer_name]['nModes2Correct'] = xp.array(self._cfile[slope_computer_name]['nModes2Correct'])
        return self._cfile[slope_computer_name]

    def read_loop_pars(self):
        """ Read loop parameters from the configuration file."""
        return self._cfile['LOOP']

    def _read_yaml(self, file_path: str):
        """
        Read a YAML file and return the contents.
        """
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
        return data



   

    

