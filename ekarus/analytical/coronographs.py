import xupy as xp
from abc import abstractmethod

import matplotlib.pyplot as plt
from arte.types.mask import CircularMask


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

    def _get_apodizer(self):
        return 1.0

    def get_coronographic_psf(self, input_field, oversampling, lambdaInM=None):
        self.oversampling = oversampling
        self.lambdaInM = lambdaInM
        pad_width = int(max(input_field.shape)*(self.oversampling-1))//2
        self._apodizer = self._get_apodizer()
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
    
    def show_coronograph_prop(self, maxLogPsf=None):
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
                           1/self.oversampling,title='PSF at focal mask',
                           maxLogVal=maxLogPsf)
        plt.subplot(1,4,3)
        plt.imshow(xp.asnumpy(xp.abs(self._pupil_mask)),cmap='grey',origin='lower')
        plt.title('Pupil stop')
        plt.subplot(1,4,4)
        self.showZoomedPSF(xp.abs(self._focal_coro_field)**2,
                           1/self.oversampling,title='Coronographic PSF',
                           maxLogVal=maxLogPsf)

    @staticmethod
    def showZoomedPSF(image, pixelSize, maxLogVal = None, title='',
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
                    cmap='twilight',vmin=-24)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        cbar= plt.colorbar()
        cbar.ax.set_ylabel(zlabel)
    

class PerfectCoronograph(Coronograph):

    def __init__(self):
        pass

    def _get_pupil_mask(self, field):
        field_amp = xp.abs(field)
        phase = xp.angle(field)[field_amp>1e-12]
        phase_var = xp.sum((phase-xp.mean(phase))**2)/len(phase)
        pupil_mask = field_amp * (xp.sqrt(xp.exp(-phase_var))\
                                  -xp.exp(1j*xp.angle(field),dtype=xp.complex64)) \
                            * xp.exp(-1j*xp.angle(field),dtype=xp.complex64)
        return pupil_mask
    
    def _get_focal_plane_mask(self, field):
        return xp.ones(field.shape,dtype=bool)


class LyotCoronograph(Coronograph):

    def __init__(self,
                referenceLambdaInM:float,
                inFocalStopInLambdaOverD:float,
                outFocalStopInLambdaOverD:float=None,
                outPupilStopInFractionOfPupil:float=1.0,
                inPupilStopInFractionOfPupil:float=0.0):
        self._refLambdaInM = referenceLambdaInM
        self._iwaInLambdaOverD = inFocalStopInLambdaOverD
        self._owaInLambdaOverD = outFocalStopInLambdaOverD
        self._inPupilStopSize = inPupilStopInFractionOfPupil
        self._outPupilStopSize = outPupilStopInFractionOfPupil

    def _get_pupil_mask(self, field):
        inStop = CircularMask(field.shape, maskRadius=self._inPupilStopSize*max(field.shape)/self.oversampling)
        outStop = CircularMask(field.shape, maskRadius=self._outPupilStopSize*max(field.shape)/self.oversampling)
        pupil_mask = xp.logical_and(xp.asarray(outStop.asTransmissionValue()),xp.asarray(inStop.mask()))
        return pupil_mask
    
    def _get_focal_plane_mask(self, field):
        lambdaInM2Px = self.oversampling
        if self.lambdaInM is not None:
            lambdaInM2Px *= self._refLambdaInM/self.lambdaInM
        iwa = CircularMask(field.shape, maskRadius=self._iwaInLambdaOverD*lambdaInM2Px)
        if self._owaInLambdaOverD is not None:
            owa = CircularMask(field.shape, maskRadius=self._owaInLambdaOverD*lambdaInM2Px)
            focal_mask = xp.logical_and(xp.asarray(iwa.mask()),xp.asarray(owa.asTransmissionValue()))
        else:
            focal_mask = xp.asarray(iwa.mask())
        return focal_mask
    

class KnifeEdgeCoronograph(Coronograph):

    def __init__(self,
                referenceLambdaInM:float,
                iwaFocalStopInLambdaOverD:float,
                outPupilStopInFractionOfPupil:float=1.0,
                inPupilStopInFractionOfPupil:float=0.0):
        self._refLambdaInM = referenceLambdaInM
        self._edgeInLambdaOverD = iwaFocalStopInLambdaOverD
        self._inPupilStopSize = inPupilStopInFractionOfPupil
        self._outPupilStopSize = outPupilStopInFractionOfPupil

    def _get_pupil_mask(self, field):
        inStop = CircularMask(field.shape, maskRadius=self._inPupilStopSize*max(field.shape)/self.oversampling)
        outStop = CircularMask(field.shape, maskRadius=self._outPupilStopSize*max(field.shape)/self.oversampling)
        pupil_mask = xp.logical_and(xp.asarray(outStop.asTransmissionValue()),xp.asarray(inStop.mask()))
        return pupil_mask
    
    def _get_focal_plane_mask(self, field):
        lambdaInM2Px = self.oversampling
        if self.lambdaInM is not None:
            lambdaInM2Px *= self._refLambdaInM/self.lambdaInM
        _,X = xp.mgrid[0:field.shape[0],0:field.shape[1]]
        edge = (field.shape[1]//2+self._edgeInLambdaOverD*lambdaInM2Px)
        focal_mask = xp.ones(field.shape)
        focal_mask[X<=edge] = 0
        return focal_mask
    

class FourQuadrantCoronograph(Coronograph):

    def __init__(self,
                referenceLambdaInM:float,
                outPupilStopInFractionOfPupil:float=1.0,
                inPupilStopInFractionOfPupil:float=0.0):
        self._refLambdaInM = referenceLambdaInM
        self._inPupilStopSize = inPupilStopInFractionOfPupil
        self._outPupilStopSize = outPupilStopInFractionOfPupil

    def _get_pupil_mask(self, field):
        inStop = CircularMask(field.shape, maskRadius=self._inPupilStopSize*max(field.shape)/self.oversampling)
        outStop = CircularMask(field.shape, maskRadius=self._outPupilStopSize*max(field.shape)/self.oversampling)
        pupil_mask = xp.logical_and(xp.asarray(outStop.asTransmissionValue()),xp.asarray(inStop.mask()))
        return pupil_mask
    
    def _get_focal_plane_mask(self, field):
        nx,ny = field.shape
        cx,cy = nx//2,ny//2
        X,Y = xp.mgrid[0:nx,0:ny]
        fmask = xp.ones(field.shape)*xp.pi
        top_left = xp.logical_and(X>=cx, Y<cy)
        bottom_right = xp.logical_and(X<cx, Y>=cy)
        fmask[top_left] = 0
        fmask[bottom_right] = 0
        if self.lambdaInM is not None:
            fmask *= self._refLambdaInM/self.lambdaInM
        focal_mask = xp.exp(1j*fmask)
        return focal_mask
    

class VectorVortexCoronograph(Coronograph):

    def __init__(self,
                referenceLambdaInM:float,
                charge:int,
                outPupilStopInFractionOfPupil:float=1.0,
                inPupilStopInFractionOfPupil:float=0.0,
                addInVortex:bool=False,
                inVortexRadInLambdaOverD:float=None,
                inVortexCharge:int=None,
                inVortexShift:float=None):
        self._refLambdaInM = referenceLambdaInM
        self._charge = charge
        self._inPupilStopSize = inPupilStopInFractionOfPupil
        self._outPupilStopSize = outPupilStopInFractionOfPupil
        self._inVortex = addInVortex
        if addInVortex:
            self._innerRadInLambdaOverD = 0.62 if inVortexRadInLambdaOverD is None else inVortexRadInLambdaOverD
            self._innerCharge = charge if inVortexCharge is None else inVortexCharge
            self._innerShift = xp.pi if inVortexShift is None else inVortexShift

    def _get_pupil_mask(self, field):
        inStop = CircularMask(field.shape, maskRadius=self._inPupilStopSize*max(field.shape)/self.oversampling)
        outStop = CircularMask(field.shape, maskRadius=self._outPupilStopSize*max(field.shape)/self.oversampling)
        pupil_mask = xp.logical_and(xp.asarray(outStop.asTransmissionValue()),xp.asarray(inStop.mask()))
        return pupil_mask
    
    def _get_focal_plane_mask(self, field):
        nx, ny = field.shape
        cx, cy = nx // 2, ny // 2
        X, Y = xp.mgrid[0:nx, 0:ny]   
        theta = xp.arctan2((X - cx), (Y - cy))
        theta = (theta + 2 * xp.pi) % (2 * xp.pi)
        vortex = self._charge * theta
        if self._inVortex is True:
            rho = xp.sqrt((X-cx)**2+(Y-cy)**2)
            lambdaInM2Px = self.oversampling
            if self.lambdaInM is not None:
                lambdaInM2Px *= self._refLambdaInM/self.lambdaInM
            inTheta = xp.arctan2((X - cx), (Y - cy))
            inTheta = (inTheta + 2 * xp.pi) % (2 * xp.pi)
            inVortex = self._innerCharge * inTheta + self._innerShift
            inRho = self._innerRadInLambdaOverD * lambdaInM2Px
            vortex[rho<=inRho] = inVortex[rho<=inRho]
        if self.lambdaInM is not None:
            vortex *= self._refLambdaInM/self.lambdaInM
        focal_mask = xp.exp(1j*vortex)
        return focal_mask

