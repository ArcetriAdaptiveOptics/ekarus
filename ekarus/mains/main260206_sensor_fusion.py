import matplotlib.pyplot as plt

from ekarus.e2e.woofer_tweeter_ao_class import WooferTweeterAO
import xupy as xp


def main(tn:str, 
         atmo_tn:str='paranal',
         show:bool=False,
         show_contrast:bool=True,
         lambdaRef:float=700e-9,
         saveprefix:str=None,
         bootStrapIts:int=0):

    print('Initializing devices ...')
    wt = WooferTweeterAO(tn)

    amp = 25e-9

    wt.initialize_turbulence(tn=atmo_tn)
    wt.get_photons_per_subap(wt.starMagnitude)
    KL, m2c = wt.define_KL_modes(wt.dm, zern_modes=2)

    _, IM1 = wt.compute_reconstructor(wt.sc1, KL, lambdaInM=wt.wfs1.lambdaInM, ampsInM=amp)
    wt.sc1.load_reconstructor(IM1,m2c)

    # Redefine slope computer
    wt.sc2 = CombinedSlopeComputer(wt.sc1,wt.sc2)

    _, IM2 = wt.compute_reconstructor(wt.sc2, KL[:wt.sc2.nModes,:], lambdaInM=wt.wfs2.lambdaInM, ampsInM=5e-9, save_prefix='fusion_')
    wt.sc2.load_reconstructor(IM2,m2c[:,:wt.sc2.nModes])


    wt.bootStrapIts = bootStrapIts

    wt.KL = KL.copy()
    
    print('Running the loop ...')
    if saveprefix is None:
        saveprefix = f'wt_mag{wt.starMagnitude:1.0f}_'
    sig2, input_sig2 = wt.run_loop(lambdaRef, wt.starMagnitude, save_prefix=saveprefix)
    wt.sig2 = sig2

    if show_contrast:
        wt.psd, wt.pix_scale = wt.plot_contrast(lambdaRef=lambdaRef, 
                                                                    frame_ids=xp.arange(100+wt.bootStrapIts,wt.Nits).tolist(),
                                                                    save_prefix=saveprefix, oversampling=10)
        # wt.smf = wt.plot_ristretto_contrast(lambdaRef=lambdaRef,frame_ids=xp.arange(wt.Nits).tolist(), save_prefix=saveprefix, oversampling=10)
        
    if show:
        wt.plot_iteration(lambdaRef, frame_id=-1, save_prefix=saveprefix)
        
        tvec = xp.asnumpy(xp.arange(wt.Nits)*wt.dt*1e+3)
        plt.figure()#figsize=(1.7*Nits/10,3))
        plt.plot(tvec,xp.asnumpy(input_sig2),'-.',label='open loop')
        plt.plot(tvec,xp.asnumpy(sig2),'-.',label='closed loop')
        plt.legend()
        plt.grid()
        plt.xlim([0.0,tvec[-1]])
        plt.xlabel('Time [ms]')
        plt.ylabel(r'$\sigma^2 [rad^2]$')
        plt.gca().set_yscale('log')

        plt.show()

    return wt

# class CombinedSlopeComputer():

#     def __init__(self,sc1,sc2):
#         self._sc1 = sc1
#         self._sc2 = sc2

#         self.iir_num = sc2.iir_num
#         self.iir_den = sc2.iir_den
#         self.modalGains = sc2.modalGains
#         self.nModes = sc2.nModes
#         self.slope_null = xp.hstack([sc1.slope_null,sc2.slope_null])
#         self.lambda1Over2 = sc1._wfs.lambdaInM/sc2._wfs.lambdaInM

#         self.dt = sc2.dt
#         self.delay = sc2.delay
#         self.lambdaInM = sc2._wfs.lambdaInM


#     def load_reconstructor(self, IM, m2c):
#         IMn = IM[:,:self.nModes]
#         U,D,V = xp.linalg.svd(IMn, full_matrices=False)
#         Rec = (V.T * 1/D) @ U.T
#         self.Rec = Rec
#         self.m2c = m2c[:,:self.nModes]

#     def iir_filter(self,x,y):
#         L = xp.shape(x)[0]
#         fb = xp.stack([self.iir_num[i]*x[-1-i] for i in range(min(len(self.iir_num),L))])
#         iir = xp.sum(fb,axis=0)
#         if self.iir_den is not None and L > 1:
#             ff = xp.stack([self.iir_den[i]*y[-1-i] for i in range(1,min(len(self.iir_den),L))])
#             iir -= xp.sum(ff,axis=0)
#         return iir

#     def compute_slopes(self, input_field2, lambdaOverD, nPhotons):
#         input_field1 = xp.abs(input_field2) * xp.exp(1j*xp.angle(input_field2)/self.lambda1Over2,dtype=xp.cfloat)
#         slopes1 = self._sc1.compute_slopes(input_field1,lambdaOverD,nPhotons)
#         slopes2 = self._sc2.compute_slopes(input_field2,lambdaOverD,nPhotons)
#         slopes = xp.hstack([slopes1,slopes2])
#         slopes -= self.slope_null
#         return slopes 


if __name__ == '__main__':
    wt = main(show=True)