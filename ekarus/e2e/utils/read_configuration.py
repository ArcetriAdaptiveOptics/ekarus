import configparser

class ConfigReader():

    def __init__(self, config_path):
        self._config = configparser.ConfigParser()
        self._config.read(config_path)

    def read_pupil_pars(self):
        pup_conf = self._config['PUPIL']
        pupil_size = float(pup_conf['pupilSizeInM'])
        pupil_pixel_size = int(pup_conf['Npix'])
        oversampling = int(pup_conf['oversampling'])
        return pupil_size, pupil_pixel_size, oversampling
    
    def read_telescope_pars(self):
        telescope_conf = self._config['TELESCOPE']
        telescope_size = float(telescope_conf['telescopeSizeInM'])
        throughput = float(telescope_conf['throughput'])
        return telescope_size, throughput
    
    def read_sensor_pars(self):
        sensor_conf = self._config['WFS']
        apex_angle = float(sensor_conf['apex_angle'])
        subaperture_size = float(sensor_conf['subaperture_size'])
        return apex_angle, subaperture_size
    
    def read_dm_pars(self):
        dm_conf = self._config['DM']
        Nacts = int(dm_conf['Nacts'])
        return Nacts
    
    def read_detector_pars(self):
        detector_conf = self._config['DETECTOR']
        detector_shape = (int(detector_conf['detector_shape'].split(',')[0]), int(detector_conf['detector_shape'].split(',')[1]))
        RON = float(detector_conf['RON'])
        return detector_shape, RON
    
    def read_atmo_pars(self):
        atmo_conf = self._config['ATMO']
        r0 = float(atmo_conf['r0'])
        L0 = float(atmo_conf['L0'])
        windSpeed = float(atmo_conf['windSpeed'])
        windAngle = float(atmo_conf['windAngle'])
        return r0, L0, windSpeed, windAngle


        


    

