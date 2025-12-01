import xupy as xp
from abc import abstractmethod

from arte.types.mask import CircularMask

class Coronograph(object):
    
    @abstractmethod
    def _propagate_field_through_pupil_mask(self, field):
        """ Override this method with the 
        function to propagate an input focal
        plane electric field trhough the 
        coronograph pupil-plane mask """

    @abstractmethod
    def _propagate_field_through_focal_plane_mask(self, field):
        """ Override this method with the 
        function to propagate an input focal
        plane electric field trhough the 
        coronograph focal-plane mask """

    def get_coronographic_psf(self, input_field, oversampling):
        self.oversampling = oversampling
        pad_width = int(max(input_field.shape)*(self.oversampling-1))//2
        padded_field = xp.pad(input_field,pad_width=pad_width,mode='constant',constant_values=0.0)
        self._focal_field = xp.fft.fftshift(xp.fft.fft2(padded_field))
        prop_focal_field = self._propagate_field_through_focal_plane_mask(self._focal_field)
        pupil_field = xp.fft.ifft2(xp.fft.ifftshift(prop_focal_field))
        coro_field = self._propagate_field_through_pupil_mask(pupil_field)
        self._focal_coro_field = xp.fft.fftshift(xp.fft.fft2(coro_field))
        psf = abs(self._focal_coro_field)**2
        return psf
    

class PerfectCoronograph(Coronograph):

    def __init__(self):
        pass

    def _propagate_field_through_pupil_mask(self, field):
        field_amp = xp.abs(field)
        phase = xp.angle(field)[field_amp>1e-12]
        print(xp.size(field_amp),len(phase))
        phase_var = xp.sum((phase-xp.mean(phase))**2)/len(phase)
        perfect_coro_field = field_amp * (xp.sqrt(xp.exp(-phase_var))-xp.exp(1j*xp.angle(field),dtype=xp.complex64))
        return perfect_coro_field
    
    def _propagate_field_through_focal_plane_mask(self, field):
        return field


class LyotCoronograph(Coronograph):

    def __init__(self,
                inFocalStopInLambdaOverD:float,
                outFocalStopInLambdaOverD:float=None,
                outPupilStopInFractionOfPupil:float=1.0,
                inPupilStopInFractionOfPupil:float=0.0):
        self._iwaInLambdaOverD = inFocalStopInLambdaOverD
        self._owaInLambdaOverD = outFocalStopInLambdaOverD
        self._inPupilStopSize = inPupilStopInFractionOfPupil
        self._outPupilStopSize = outPupilStopInFractionOfPupil

    def _propagate_field_through_pupil_mask(self, field):
        inStop = CircularMask(field.shape, maskRadius=self._inPupilStopSize*max(field.shape)/self.oversampling)
        outStop = CircularMask(field.shape, maskRadius=self._outPupilStopSize*max(field.shape)/self.oversampling)
        self.pupil_mask = xp.logical_and(xp.asarray(outStop.asTransmissionValue()),xp.asarray(inStop.mask()))
        prop_field = field * self.pupil_mask
        return prop_field
    
    def _propagate_field_through_focal_plane_mask(self, field):
        iwa = CircularMask(field.shape, maskRadius=self._iwaInLambdaOverD*self.oversampling)
        if self._owaInLambdaOverD is not None:
            owa = CircularMask(field.shape, maskRadius=self._owaInLambdaOverD*self.oversampling)
            self.focal_mask = xp.logical_and(xp.asarray(iwa.mask()),xp.asarray(owa.asTransmissionValue()))
        else:
            self.focal_mask = xp.asarray(iwa.mask())
        prop_field = field * self.focal_mask
        return prop_field
    

class KnifeEdgeCoronograph(Coronograph):

    def __init__(self,
                iwaFocalStopInLambdaOverD:float,
                outPupilStopInFractionOfPupil:float=1.0,
                inPupilStopInFractionOfPupil:float=0.0):
        self._edgeInLambdaOverD = iwaFocalStopInLambdaOverD
        self._inPupilStopSize = inPupilStopInFractionOfPupil
        self._outPupilStopSize = outPupilStopInFractionOfPupil

    def _propagate_field_through_pupil_mask(self, field):
        inStop = CircularMask(field.shape, maskRadius=self._inPupilStopSize*max(field.shape)/self.oversampling)
        outStop = CircularMask(field.shape, maskRadius=self._outPupilStopSize*max(field.shape)/self.oversampling)
        self.pupil_mask = xp.logical_and(xp.asarray(outStop.asTransmissionValue()),xp.asarray(inStop.mask()))
        prop_field = field * self.pupil_mask
        return prop_field
    
    def _propagate_field_through_focal_plane_mask(self, field):
        _,X = xp.mgrid[0:field.shape[0],0:field.shape[1]]
        self.focal_mask = X > field.shape[1]//2+self._edgeInLambdaOverD*self.oversampling
        prop_field = field * self.focal_mask
        return prop_field
    

class FourQuadrantCoronograph(Coronograph):

    def __init__(self,
                outPupilStopInFractionOfPupil:float=1.0,
                inPupilStopInFractionOfPupil:float=0.0):
        self._inPupilStopSize = inPupilStopInFractionOfPupil
        self._outPupilStopSize = outPupilStopInFractionOfPupil

    def _propagate_field_through_pupil_mask(self, field):
        inStop = CircularMask(field.shape, maskRadius=self._inPupilStopSize*max(field.shape)/self.oversampling)
        outStop = CircularMask(field.shape, maskRadius=self._outPupilStopSize*max(field.shape)/self.oversampling)
        self.pupil_mask = xp.logical_and(xp.asarray(outStop.asTransmissionValue()),xp.asarray(inStop.mask()))
        prop_field = field * self.pupil_mask
        return prop_field
    
    def _propagate_field_through_focal_plane_mask(self, field):
        nx,ny = field.shape
        cx,cy = nx//2,ny//2
        X,Y = xp.mgrid[0:nx,0:ny]
        fmask = xp.ones(field.shape)*xp.pi
        top_left = xp.logical_and(X>=cx, Y<cy)
        bottom_right = xp.logical_and(X<cx, Y>=cy)
        fmask[top_left] = 0
        fmask[bottom_right] = 0
        self.focal_mask = fmask
        prop_field = field * xp.exp(1j*fmask)
        return prop_field
    

class VortexCoronograph(Coronograph):

    def __init__(self,
                charge:int,
                outPupilStopInFractionOfPupil:float=1.0,
                inPupilStopInFractionOfPupil:float=0.0):
        self._charge = charge
        self._inPupilStopSize = inPupilStopInFractionOfPupil
        self._outPupilStopSize = outPupilStopInFractionOfPupil

    def _propagate_field_through_pupil_mask(self, field):
        inStop = CircularMask(field.shape, maskRadius=self._inPupilStopSize*max(field.shape)/self.oversampling)
        outStop = CircularMask(field.shape, maskRadius=self._outPupilStopSize*max(field.shape)/self.oversampling)
        self.pupil_mask = xp.logical_and(xp.asarray(outStop.asTransmissionValue()),xp.asarray(inStop.mask()))
        prop_field = field * self.pupil_mask
        return prop_field
    
    def _propagate_field_through_focal_plane_mask(self, field):
        nx, ny = field.shape
        cx, cy = nx // 2, ny // 2
        X, Y = xp.mgrid[0:nx, 0:ny]   
        theta = xp.arctan2((X - cx), (Y - cy))
        theta = (theta + 2 * xp.pi) % (2 * xp.pi)
        vortex = self._charge * theta
        self.focal_mask = vortex
        prop_field = field * xp.exp(1j*vortex)
        return prop_field

