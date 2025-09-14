import configparser
import numpy as np

class ConfigReader():

    def __init__(self, config_path, xp=np):
        self._config = configparser.ConfigParser(inline_comment_prefixes=('#', ';'))
        self._config.read(config_path)
        self.xp = xp
    
    def read_telescope_pars(self):
        telescope_conf = self._config['TELESCOPE']
        aperture_size = float(telescope_conf['pupilSizeInM'])
        aperture_pixel_size = int(telescope_conf['pupilSizeInPixels'])
        # oversampling = int(telescope_conf['oversampling'])
        throughput = float(telescope_conf['throughput'])
        try:
            wfs_split = float(telescope_conf['beam_splitter'])
            throughput *= wfs_split
        except KeyError:
            pass
        return aperture_size, aperture_pixel_size, throughput
    
    def read_sensor_pars(self, sensor_name:str=None):
        if sensor_name is None:
            sensor_name = 'WFS'
        sensor_conf = self._config[sensor_name]
        apex_angle = float(sensor_conf['apex_angle'])
        oversampling = int(sensor_conf['oversampling'])
        return apex_angle, oversampling
    
    def read_dm_pars(self, dm_name:str=None):
        if dm_name is None:
            dm_name = 'DM'
        dm_conf = self._config[dm_name]
        Nacts = int(dm_conf['Nacts'])
        return Nacts
    
    def read_detector_pars(self, detector_name:str=None):
        if detector_name is None:
            detector_name = 'DETECTOR'
        detector_conf = self._config[detector_name]
        detector_shape = (int(detector_conf['detector_shape'].split(',')[0]), int(detector_conf['detector_shape'].split(',')[1]))
        RON = float(detector_conf['RON'])
        quantum_efficiency = float(detector_conf['quantum_efficiency'])
        return detector_shape, RON, quantum_efficiency
    
    def read_atmo_pars(self):
        atmo_conf = self._config['ATMO']
        r0 = self._read_array(atmo_conf['r0'])
        L0 = float(atmo_conf['L0'])
        windSpeed = self._read_array(atmo_conf['windSpeed'])
        windAngle = self._read_array(atmo_conf['windAngle'])*self.xp.pi/180 # radians 2 degrees
        return r0, L0, windSpeed, windAngle
    
    def _read_array(self, a):
        lenA = len(a.split(','))
        if lenA > 1:
            b = self.xp.zeros(lenA,dtype=float)
            for k in range(lenA):
                b[k] = float(a.split(',')[k])
        else:
            b = float(a)
        return b



        


    

