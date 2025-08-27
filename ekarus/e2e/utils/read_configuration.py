import configparser

class ConfigReader():

    def __init__(self, config_path):
        self._config = configparser.ConfigParser()
        self._config = self._config.read(config_path)

    def read_pupil_pars(self):
        pup_conf = self._config['PUPIL']
        pupil_size = pup_conf['pupilSizeInM']
        pupil_pixel_size = pup_conf['Npix']
        oversampling = pup_conf['oversampling']
        return pupil_size, pupil_pixel_size, oversampling
    
    def read_telescope_pars(self):
        telescope_conf = self._config['TELESCOPE']
        telescope_size = telescope_conf['telescopeSizeInM']
        throughput = telescope_conf['throughput']
        return telescope_size, throughput
    
    def read_sensor_pars(self):
        sensor_conf = self._config['WFS']
        apex_angle = sensor_conf['apex_angle']
        return apex_angle
    
    def read_dm_pars(self):
        dm_conf = self._config['DM']
        Nacts = dm_conf['Nacts']
        return Nacts
    
    def read_detector_pars(self):
        detector_conf = self._config['DETECTOR']
        detector_shape = detector_conf['detector_shape']
        RON = detector_conf['RON']
        return detector_shape, RON


        


    

