import xupy as xp
from arte.types.mask import CircularMask

from ekarus.coronography.abstract_coronograph import Coronograph


class LyotCoronograph(Coronograph):

    def __init__(self,
                referenceLambdaInM:float,
                inFocalStopInLambdaOverD:float,
                outFocalStopInLambdaOverD:float=None,
                outPupilStopInFractionOfPupil:float=1.0,
                inPupilStopInFractionOfPupil:float=0.0,
                knife_egde:bool=False):
        if knife_egde is True and outFocalStopInLambdaOverD is not None:
            raise ValueError('Outer working angle for the focal plane cannot be defined for a knife edge coronograph')
        self._refLambdaInM = referenceLambdaInM
        self._iwaInLambdaOverD = inFocalStopInLambdaOverD
        self._owaInLambdaOverD = outFocalStopInLambdaOverD
        self._inPupilStopSize = inPupilStopInFractionOfPupil
        self._outPupilStopSize = outPupilStopInFractionOfPupil
        self._knifeEdge = None
        if knife_egde:
            self._knifeEdge = inFocalStopInLambdaOverD

    def _get_pupil_mask(self, field):
        inStop = CircularMask(field.shape, maskRadius=self._inPupilStopSize*max(field.shape)/self.oversampling)
        outStop = CircularMask(field.shape, maskRadius=self._outPupilStopSize*max(field.shape)/self.oversampling)
        pupil_mask = xp.logical_and(xp.asarray(outStop.asTransmissionValue()),xp.asarray(inStop.mask()))
        return pupil_mask
    
    def _get_focal_plane_mask(self, field):
        lambdaInM2Px = self.oversampling
        if self.lambdaInM is not None:
            lambdaInM2Px *= self._refLambdaInM/self.lambdaInM
        if self._knifeEdge is not None:
            _,X = xp.mgrid[0:field.shape[0],0:field.shape[1]]
            edge = (field.shape[1]//2+self._knifeEdge*lambdaInM2Px)
            focal_mask = xp.ones(field.shape)
            focal_mask[X<=edge] = 0
        else:
            iwa = CircularMask(field.shape, maskRadius=self._iwaInLambdaOverD*lambdaInM2Px)
            if self._owaInLambdaOverD is not None:
                owa = CircularMask(field.shape, maskRadius=self._owaInLambdaOverD*lambdaInM2Px)
                focal_mask = xp.logical_and(xp.asarray(iwa.mask()),xp.asarray(owa.asTransmissionValue()))
            else:
                focal_mask = xp.asarray(iwa.mask())
        return focal_mask
    

# class KnifeEdgeCoronograph(Coronograph):

#     def __init__(self,
#                 referenceLambdaInM:float,
#                 iwaFocalStopInLambdaOverD:float,
#                 outPupilStopInFractionOfPupil:float=1.0,
#                 inPupilStopInFractionOfPupil:float=0.0):
#         self._refLambdaInM = referenceLambdaInM
#         self._edgeInLambdaOverD = iwaFocalStopInLambdaOverD
#         self._inPupilStopSize = inPupilStopInFractionOfPupil
#         self._outPupilStopSize = outPupilStopInFractionOfPupil

#     def _get_pupil_mask(self, field):
#         inStop = CircularMask(field.shape, maskRadius=self._inPupilStopSize*max(field.shape)/self.oversampling)
#         outStop = CircularMask(field.shape, maskRadius=self._outPupilStopSize*max(field.shape)/self.oversampling)
#         pupil_mask = xp.logical_and(xp.asarray(outStop.asTransmissionValue()),xp.asarray(inStop.mask()))
#         return pupil_mask
    
#     def _get_focal_plane_mask(self, field):
#         lambdaInM2Px = self.oversampling
#         if self.lambdaInM is not None:
#             lambdaInM2Px *= self._refLambdaInM/self.lambdaInM
#         _,X = xp.mgrid[0:field.shape[0],0:field.shape[1]]
#         edge = (field.shape[1]//2+self._edgeInLambdaOverD*lambdaInM2Px)
#         focal_mask = xp.ones(field.shape)
#         focal_mask[X<=edge] = 0
#         return focal_mask