import xupy as xp
from ekarus.coronography.apodizer import define_apodizing_phase

from ekarus.coronography.abstract_coronograph import Coronograph
from ekarus.coronography.lyot_coronographs import LyotCoronograph

class ApodizerPhasePlateCoronograph(Coronograph):

    def __init__(self,
                 referenceLambdaInM:float,
                 pupil,
                 contrastInDarkHole:float,
                 iwaInLambdaOverD:float,
                 owaInLambdaOverD:float,
                 oversampling:int=8,
                 beta:float=0.9,
                 max_its:int=1000,
                 show:bool=True):
        self._refLambdaInM = referenceLambdaInM
        self._telescopePupil = pupil.copy()
        self._apodizer_phase = define_apodizing_phase(pupil, contrastInDarkHole, 
                              iwaInLambdaOverD, owaInLambdaOverD,
                              beta, oversampling, max_its=max_its, show=show)

    def _get_apodizer(self, lambdaInM):
        phase = self._apodizer_phase 
        if lambdaInM is not None:
            phase *= self._refLambdaInM/lambdaInM
        return xp.exp(1j*phase,dtype=xp.cfloat)

    def _get_pupil_mask(self, field):
        return 1.0
    
    def _get_focal_plane_mask(self, field):
        return 1.0
    

class PAPLC(LyotCoronograph):

    def __init__(self,
                 referenceLambdaInM:float,
                 pupil,
                 contrastInDarkHole:float,
                 iwaInLambdaOverD:float,
                 owaInLambdaOverD:float,
                 fpmIWAInLambdaOverD:float,
                 fpmOWAInLambdaOverD:float=None,
                 knife_edge:bool=True,
                 outPupilStopInFractionOfPupil:float=1.0,
                 inPupilStopInFractionOfPupil:float=0.0,
                 oversampling:int=8,
                 beta:float=0.9,
                 max_its:int=1000,
                 show:bool=True):
        super().__init__(referenceLambdaInM,
                        fpmIWAInLambdaOverD,
                        fpmOWAInLambdaOverD,
                        outPupilStopInFractionOfPupil,
                        inPupilStopInFractionOfPupil,
                        knife_edge)
        self._telescopePupil = pupil.copy()
        self._apodizer_phase = define_apodizing_phase(pupil, contrastInDarkHole, 
                              iwaInLambdaOverD, owaInLambdaOverD,
                              beta, oversampling, max_its=max_its, show=show)

    def _get_apodizer(self, lambdaInM):
        phase = self._apodizer_phase 
        if lambdaInM is not None:
            phase *= self._refLambdaInM/lambdaInM
        return xp.exp(1j*phase,dtype=xp.cfloat)
