import numpy as np
from arte.math.toccd import toccd


class Detector:

    def __init__(self, detector_shape = None, RON:float = 0.0, quantum_efficiency:float = 1.0, beam_split_ratio:float = 1.0, max_bits:int = 12, xp=np):
        """
        Detector constructor.
        
        :param detector_shape: number of pixels in the detector
        :param RON: readout noise in electrons
        :param quantum_efficiency: the quantum efficiency of the detector
        :param beam_split_ratio: the percentage of light directed to the detetctor 
        :param max_bits: the maximum number of counts per pixel
        """

        self.detector_shape = detector_shape
        self.RON = RON
        self.quantum_efficiency = quantum_efficiency
        self.max_bits = max_bits
        self.beam_split_ratio = beam_split_ratio

        self.subapertures = None

        self._xp = xp
        self.dtype = xp.float32 if xp.__name__ == 'cupy' else xp.float64


    def image_on_detector(self, image, rebin_fact:int = 0, photon_flux = None):

        ccd_size =self.detector_shape
        if rebin_fact > 0:
            rebin = 4*rebin_fact
            ccd_size = (self.detector_shape[0]//rebin, self.detector_shape[1]//rebin)
        ccd_intensity = toccd(image, ccd_size)

        if photon_flux is not None:
            ccd_intensity = self.add_electron_noise(ccd_intensity, photon_flux)

        self.last_frame = ccd_intensity

        return ccd_intensity


    def add_electron_noise(self, intensity, flux):
        """
        Simulate detector noise based on the given intensity and real photon flux.
        
        Parameters:
        - intensity: 2D array of intensity values.
        - flux: Real photon flux in the image.
        
        Returns:
        - Noisy intensity image.
        """

        # Re-scale the intensity based on the flux and quantum efficiency
        norm_intensity = self.beam_split_ratio * self.quantum_efficiency * flux * intensity/self._xp.sum(intensity)

        # Noise
        poisson_noise = self._xp.random.poisson(norm_intensity, self._xp.shape(intensity)) # Possion noise
        readout_noise = self._xp.random.normal(0, self.RON, size=self._xp.shape(intensity)) # readout noise
        
        noisy_intensity = self._xp.round(norm_intensity + poisson_noise + readout_noise)

        # Saturation
        noisy_intensity = self._xp.minimum(2**self.max_bits,noisy_intensity)
        
        return self._xp.maximum(0,noisy_intensity)
    

    
