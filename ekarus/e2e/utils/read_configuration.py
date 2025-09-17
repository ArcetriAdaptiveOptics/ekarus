import configparser
import numpy as np

class ConfigReader():

    def __init__(self, config_path, xp=np):
        self._config = configparser.ConfigParser(inline_comment_prefixes=('#', ';'))
        self._config.read(config_path)
        self.xp = xp
    
    def read_telescope_pars(self):
        telescope_conf = self._config['TELESCOPE']
        pupilSizeInM = float(telescope_conf['pupilSizeInM'])
        pupilSizeInPixels = int(telescope_conf['pupilSizeInPixels'])
        throughput = float(telescope_conf['throughput'])
        return pupilSizeInM, pupilSizeInPixels, throughput
    
    def read_sensor_pars(self, sensor_name:str=None):
        if sensor_name is None:
            sensor_name = 'WFS'
        sensor_conf = self._config[sensor_name]
        apex_angle = float(sensor_conf['apex_angle'])
        oversampling = int(sensor_conf['oversampling'])
        modulationAngleInLambdaOverD = float(sensor_conf['modulationAngleInLambdaOverD'])
        subaperture_size = float(sensor_conf['subaperture_size'])
        return apex_angle, oversampling, modulationAngleInLambdaOverD, subaperture_size
    
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
        outerScaleInM = float(atmo_conf['outerScaleInM'])
        windSpeed = self._read_array(atmo_conf['windSpeed'])
        windAngle = self._read_array(atmo_conf['windAngle'])*self.xp.pi/180 # radians 2 degrees
        return r0, outerScaleInM, windSpeed, windAngle
    
    def read_target_pars(self):
        target_conf = self._config['TARGET']
        lambdaInM = float(target_conf['lambdaInNm'])*1e-9
        starMagnitude = float(target_conf['starMagnitude'])
        return lambdaInM, starMagnitude
    
    def read_loop_pars(self):
        loop_conf = self._config['LOOP']
        nIterations = int(loop_conf['nIterations'])
        loopFrequencyInHz = float(loop_conf['loopFrequencyInHz'])
        integratorGain = float(loop_conf['integratorGain'])
        delay = int(loop_conf['delay'])
        nModes2Correct = int(loop_conf['nModes2Correct'])
        # ttOffloadFrequencyInHz = float(loop_conf['ttOffloadFrequencyInHz'])
        return nIterations, loopFrequencyInHz, integratorGain, delay, nModes2Correct#, ttOffloadFrequencyInHz
    
    def _read_array(self, a):
        lenA = len(a.split(','))
        if lenA > 1:
            b = self.xp.zeros(lenA,dtype=float)
            for k in range(lenA):
                b[k] = float(a.split(',')[k])
        else:
            b = float(a)
        return b



        


    

