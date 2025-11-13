import xupy as xp
from ekarus.e2e.utils.image_utils import reshape_on_mask
from ekarus.analytical.turbulence_layers import TurbulenceLayers


def tf_wfs(g_opt,T,T_readout,om):
    return g_opt*xp.exp(-1j*om*(T/2+T_readout))*xp.sinc(om*T/2)

def tf_rtc_and_dm(T,T_rtc,T_dm,om):
    return xp.exp(-1j*om*(T/2+T_rtc+T_dm))*xp.sinc(om*T/2)

def rad2nm(sigma, lambda_ref):
    return sigma*lambda_ref/(2*xp.pi)

def r0_at_lambda0(r0, lambda0):
    return r0*(lambda0/500e-9)**(6/5)

def temporal_psd(freq, n:int, sigma2:float, D:float, V:float):
    om = 2*xp.pi*freq
    f_cut = 0.3*(n+1)*V/D
    om_cut = f_cut*(2*xp.pi)
    PSD = xp.where(om<=om_cut, 1, om_cut**(17/3) * om**(-17/3))
    PSD *= sigma2/xp.trapz(PSD,om)
    return PSD

def von_karman_power(k,r0,L0):
    C = 0.02289558710855519
    B = k**2 + (1/L0)**2 #(2*xp.pi*k)**2 + (2*xp.pi/L0)**2
    return C * r0**(-5.0/3.0) * B**(-11.0/6.0)

def radial_order(i_mode):
    noll = i_mode + 2
    return xp.ceil(-3.0/2.0+xp.sqrt(1+8*noll)/2.0)

def Nphot(T,mag,D,thrp=1.0):
    B0 = 8.9e+9
    flux = B0 * 10**(-mag/2.5)
    area = xp.pi*D**2/4
    phots = flux*area*T
    return phots*thrp



class ErrorBudget():

    def __init__(self,r0s,L0,V,D,delays,camera,IM,wfs_frame,subaperture_masks,Npix:int=8):

        self.D = D

        self.r0 = (1/xp.sum(r0s**(-5/3)))**(3/5)
        self.L0 = L0
        self.V = V

        self.T_readout,self.T_rtc,self.T_dm = delays
        self.RON,self.dark,self.F,self.thrp = camera

        DtD = IM.T @ IM
        self.p = xp.diag(xp.linalg.inv(DtD)) #1/xp.diag(DtD)#
        self.IM = IM.copy()

        self.Npix = Npix # number of pixels fro slope computation
        # 2 slopes on the 4PWFS require 8 pixels

        self._define_pixel_intensities(wfs_frame,subaperture_masks)

    
    def update_freq_range(self,freq):
        self.freq = freq
        self.om = 2*xp.pi*freq

    def set_loop_gains(self,gains,T):
        self.gain = gains
        self.T = T

    def load_optical_gains(self,optical_gains):
        self.g_opt = optical_gains

    def control_tf(self,T,i):
        g = self.gain[i]*xp.pi/(4*(T+self.T_readout+self.T_rtc+self.T_dm))
        return g/(1j*self.om)

    def RTF(self,T,i):
        w = tf_wfs(self.g_opt[i],T,self.T_readout,self.om)
        rm = tf_rtc_and_dm(T,self.T_rtc,self.T_dm,self.om)
        c = self.control_tf(T,i)
        return 1.0/(1.0+w*rm*c)
    
    def NTF(self,T,i):
        w = tf_wfs(self.g_opt[i],self.T,self.T_readout,self.om)
        rm = tf_rtc_and_dm(T,self.T_rtc,self.T_dm,self.om)
        c = self.control_tf(T,i)
        return rm*c/(1.0+w*rm*c)

    def sigma_fit(self,N,lambdaInM):
        r0 = r0_at_lambda0(self.r0,lambdaInM)
        sigma_rad = xp.sqrt(0.2778*N**(-0.9)*(self.D/r0)**(5.0/3.0))
        return rad2nm(sigma_rad,lambdaInM)
    
    def sigma_temp(self,T,lambdaInM):
        sig2_rad = 0.0
        for i in range(len(self.gain)):
            sig2_rad += self._sigma_temp_i(T,i,lambdaInM)
        sig_rad = xp.sqrt(sig2_rad)
        return rad2nm(sig_rad,lambdaInM)
    
    def sigma_alias(self,T,N,lambdaInM):
        sig2_rad = 0.0
        for i in range(len(self.gain)):
            sig2_rad += self._sigma_alias_i(T,i,N)
        sig_rad = xp.sqrt(sig2_rad)
        return rad2nm(sig_rad,lambdaInM)
    
    def sigma_meas(self,T,mag,lambdaInM):
        sig2_rad = 0.0
        for i in range(len(self.gain)):
            sig2_rad += self._sigma_meas_i(T,i,mag)
        sig_rad = xp.sqrt(sig2_rad)
        return rad2nm(sig_rad,lambdaInM)

    def sigma_tot(self,lambdaInM,N,T,mag):
        sig_fit = self.sigma_fit(N,lambdaInM)
        sig_meas = self.sigma_meas(T,mag,lambdaInM)
        sig_alias = self.sigma_alias(T,N,lambdaInM)
        sig_temp = self.sigma_temp(T,lambdaInM)
        sig_tot = xp.sqrt(sig_fit**2 + sig_meas**2 + sig_temp**2 + sig_alias**2)
        return sig_tot, sig_alias, sig_fit, sig_meas, sig_temp
    

    def get_SR(self,lambdaInM,N,T,mag):
        sig,_,_,_,_ = self.sigma_tot(lambdaInM,N,T,mag)
        sig2_rad = (sig*(2*xp.pi)/lambdaInM)**2
        SR = xp.exp(-sig2_rad)
        return SR   
    
    def _sigma_temp_i(self,T,i,lambdaInM):
        n = radial_order(i+1) # start from tilt, not piston 
        k = n/self.D
        r0 = r0_at_lambda0(self.r0,lambdaInM)
        amp = von_karman_power(k,r0,self.L0)
        PSD = temporal_psd(self.freq, n, 1.0, self.D, self.V)
        RTF2 = xp.abs(self.RTF(T,i))**2
        sigma2_rad = amp*xp.trapz(PSD*RTF2,self.om) #xp.sum(xp.dot(PSD,RTF2))
        return sigma2_rad

    def _sigma_meas_i(self,T,i,mag):
        N = self.get_total_intensity(mag,T)
        sig2_I = self.F**2*(self.dark+self.get_pix_intensities(mag,T))+self.RON**2
        sig2_s = xp.sum(self.Npix*sig2_I)/N**2
        sig2_w = self.p[i] * sig2_s
        NTF2 = xp.abs(self.NTF(T,i))**2
        sigma2_rad = sig2_w*xp.trapz(NTF2,self.om) #xp.sum(xp.dot(sig2_w,NTF2))
        return sigma2_rad

    def _sigma_alias_i(self,T,i,N):
        sig2_alias = self.sig2_alias[i]
        NTF2 = xp.abs(self.NTF(T,i))**2
        n = radial_order(N)*2 # estimated cut-off at double the highest spatial frequency
        PSD = temporal_psd(self.freq, n, 1.0, self.D, self.V)
        sigma2_rad = sig2_alias*xp.trapz(PSD*NTF2,self.om) #xp.sum(xp.dot(PSD,NTF2))
        return sigma2_rad


    def define_aliasing_variance(self,r0s,cmask,slope_computer,lambdaInM,pixSize,KL,N,Nmodes):
        aliasing = None
        phase2modes = xp.linalg.pinv(KL.T) 
        Rec = xp.linalg.pinv(self.IM[:,:Nmodes])
        field_amp = 1-cmask
        lambdaOverD = lambdaInM/self.D
        m2rad = (2*xp.pi)/lambdaInM
        turbulence = TurbulenceLayers(r0s,self.L0)
        for i in range(N):
            turbulence.generate_phase_screens(screenSizeInMeters=self.D,screenSizeInPixels=pixSize)
            turbulence.rescale_phasescreens()
            atmo_screen = xp.sum(turbulence.phase_screens,axis=0)
            atmo_phase = atmo_screen[~cmask]
            atmo_phase -= xp.mean(atmo_phase)
            atmo_modes = phase2modes @ atmo_phase
            rec_phase = KL.T @ atmo_modes
            res_phase = atmo_phase - rec_phase

            phase_in_rad = reshape_on_mask(res_phase*m2rad, cmask)
            input_field = field_amp * xp.exp(1j*phase_in_rad)
            slope = slope_computer.compute_slopes(input_field, lambdaOverD, None)
            wfs_modes_in_rad = Rec @ slope
            if aliasing is None:
                aliasing = wfs_modes_in_rad**2
            else:
                aliasing += wfs_modes_in_rad**2
        aliasing = xp.sqrt(aliasing/N)
        self.sig2_alias = aliasing**2

    def get_total_intensity(self,mag,T):
        return Nphot(T,mag,self.D,self.thrp)#*self.diffraction_loss
    
    def get_pix_intensities(self,mag,T):
        I = self._ref_pix_intensities.copy()
        I *= self.get_total_intensity(mag,T)/xp.sum(abs(I))
        return I

    def _define_pixel_intensities(self, wfs_frame, subaperture_masks):
        A = wfs_frame[~subaperture_masks[0]]
        B = wfs_frame[~subaperture_masks[1]]
        C = wfs_frame[~subaperture_masks[2]]
        D = wfs_frame[~subaperture_masks[3]]
        mean_intensity = xp.mean(xp.hstack((A,B,C,D)))
        self.diffraction_loss = mean_intensity/xp.sum(wfs_frame) #
        self._ref_pix_intensities = xp.mean(xp.vstack((A,B,C,D)),axis=0)
        # up_down_slope = (A+B)-(C+D)
        # left_right_slope = (A+C)-(B+D)
        # self._ref_pix_intensities = xp.hstack((up_down_slope, left_right_slope))/mean_intensity
        

    # @staticmethod
    # def _Npix_from_Nsubaps(rebin):
    #     return xp.pi/4*rebin**2