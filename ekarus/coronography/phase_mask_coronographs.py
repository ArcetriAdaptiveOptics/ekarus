import xupy as xp
from arte.types.mask import CircularMask

from ekarus.coronography.abstract_coronograph import Coronograph

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

class FourQuadrantCoronograph(Coronograph):

    def __init__(self,
                referenceLambdaInM:float,
                outPupilStopInFractionOfPupil:float=1.0,
                inPupilStopInFractionOfPupil:float=0.0):
        self._refLambdaInM = referenceLambdaInM
        self._inPupilStopSize = inPupilStopInFractionOfPupil
        self._outPupilStopSize = outPupilStopInFractionOfPupil

    def _get_pupil_mask(self, field):
        inStop = CircularMask(field.shape, maskRadius=self._inPupilStopSize*max(field.shape)/2)#/self.oversampling)
        outStop = CircularMask(field.shape, maskRadius=self._outPupilStopSize*max(field.shape)/2)#/self.oversampling)
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
        inStop = CircularMask(field.shape, maskRadius=self._inPupilStopSize*max(field.shape)/2)#/self.oversampling)
        outStop = CircularMask(field.shape, maskRadius=self._outPupilStopSize*max(field.shape)/2)#/self.oversampling)
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