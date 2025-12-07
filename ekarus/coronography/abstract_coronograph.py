import xupy as xp
from abc import abstractmethod
import matplotlib.pyplot as plt


class Coronograph(object):
    
    @abstractmethod
    def _get_pupil_mask(self, field):
        """ Override this method with the 
        function to propagate an input focal
        plane electric field trhough the 
        coronograph pupil-plane mask """

    @abstractmethod
    def _get_focal_plane_mask(self, field):
        """ Override this method with the 
        function to propagate an input focal
        plane electric field trhough the 
        coronograph focal-plane mask """

    def _get_apodizer(self, lambdaInM):
        return 1.0

    def get_coronographic_psf(self, input_field, oversampling, lambdaInM=None):
        self.pupil = xp.abs(input_field) > 1e-6
        self.oversampling = oversampling
        self.lambdaInM = lambdaInM
        pad_width = int(max(input_field.shape)*(self.oversampling-1))//2
        self._apodizer = self._get_apodizer(lambdaInM)
        apodized_field = input_field * self._apodizer
        padded_field = xp.pad(apodized_field,pad_width=pad_width,mode='constant',constant_values=0.0)
        self._focal_field = xp.fft.fftshift(xp.fft.fft2(padded_field))
        self._focal_mask = self._get_focal_plane_mask(self._focal_field)
        prop_focal_field = self._focal_field * self._focal_mask
        pupil_field = xp.fft.ifft2(xp.fft.ifftshift(prop_focal_field))
        self._pupil_mask = self._get_pupil_mask(padded_field)
        coro_field = pupil_field * self._pupil_mask
        self._focal_coro_field = xp.fft.fftshift(xp.fft.fft2(coro_field))
        psf = abs(self._focal_coro_field)**2
        return psf
    
    def show_coronograph_prop(self, maxLogPsf=None, minVal:float=-24):
        phase = xp.angle(self._focal_mask)
        fcmap = 'RdBu'
        phase += 2*xp.pi * (phase < 0.0)
        if xp.max(phase) <= 1e-12:
            fcmap = 'grey'
            phase = xp.asarray(self._focal_mask)
        plt.figure(figsize=(22,4))
        plt.subplot(1,4,1)
        plt.imshow(xp.asnumpy(phase),cmap=fcmap,origin='lower')
        plt.title('Focal plane mask')
        plt.colorbar()
        plt.subplot(1,4,2)
        self.showZoomedPSF(xp.abs(self._focal_field)**2,
                           1/self.oversampling, minVal=minVal, title='PSF at focal mask',
                           maxLogVal=maxLogPsf)
        plt.subplot(1,4,3)
        plt.imshow(xp.asnumpy(xp.abs(self._pupil_mask)),cmap='grey',origin='lower')
        plt.title('Pupil stop')
        plt.subplot(1,4,4)
        self.showZoomedPSF(xp.abs(self._focal_coro_field)**2, 
                           1/self.oversampling, minVal=minVal, title='Coronographic PSF',
                           maxLogVal=maxLogPsf)
    

    @staticmethod
    def showZoomedPSF(image, pixelSize, minVal, maxLogVal = None, title='',
                   xlabel=r'$\lambda/D$', ylabel=r'$\lambda/D$', zlabel=''):
        imageHalfSizeInPoints= image.shape[0]/2
        roi= [int(imageHalfSizeInPoints*0.8), int(imageHalfSizeInPoints*1.2)]
        imageZoomedLog= xp.log(image[roi[0]: roi[1], roi[0]:roi[1]])
        if maxLogVal is None:
            maxLogVal = xp.max(imageZoomedLog)
        imageZoomedLog -= maxLogVal
        sz=imageZoomedLog.shape
        plt.imshow(xp.asnumpy(imageZoomedLog), 
                extent=[-sz[0]/2*pixelSize, sz[0]/2*pixelSize,
                    -sz[1]/2*pixelSize, sz[1]/2*pixelSize],
                    cmap='inferno',vmin=minVal)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        cbar= plt.colorbar()
        cbar.ax.set_ylabel(zlabel)
  