import xupy as xp
from numpy.ma import masked_array
import matplotlib.pyplot as plt
import os
from astropy.io import fits

from ekarus.e2e.utils.image_utils import reshape_on_mask, image_grid
from ekarus.e2e.single_stage_ao_class import SingleStageAO
from ekarus.analytical.kl_modes import compute_ifs_covmat

def main(tn:str='offaxis_modalbase',
         show:bool=False,
         show_contrast:bool =False,
         atmo_tn='nine_layers_25mOS',
         offset_calibration:bool=True,
         use_ncpa_cmd:bool=True,
         use_slope_null:bool=False,
         lambdaRef:float=700e-9,):

    ssao = SingleStageAO(tn)
    ssao.initialize_turbulence(tn=atmo_tn)

    # Define apodizer phase
    oversampling = 4
    pupil = 1-ssao.cmask.copy()
    dark_zone = define_target_roi(pupil, iwa=2.5, owa=5.5, oversampling=oversampling)
    target_contrast = xp.ones([pupil.shape[0]*oversampling,pupil.shape[1]*oversampling])
    target_contrast[dark_zone] = 1e-7
    app_phase = get_apodizer_phase(pupil, target_contrast, oversampling=oversampling, max_its=5000, beta=0.975)
    app_psf = calc_psf(pupil*xp.exp(1j*app_phase,dtype=xp.complex64))

    plt.figure()
    show_psf(app_psf,title='Apodized PSF',vmin=-8)

    # Ortho-normalize modes
    coeff = lambdaRef/ssao.pyr.lambdaInM
    app_phase *= coeff
    app_cmd = xp.linalg.pinv(ssao.dm.IFF) @ app_phase[pupil.astype(bool)]
    p = app_cmd/xp.sqrt(xp.sum(app_cmd**2))
    P = xp.eye(ssao.dm.Nacts) - xp.outer(p,p)
    Ckol = compute_ifs_covmat(pupil,diameter=ssao.pupilSizeInM,
                              influence_functions=ssao.dm.IFF.T,r0=ssao.layers._r0,
                              L0=ssao.layers.L0,oversampling=4,verbose=True,
                              xp=xp,dtype=xp.float32)
    Cred = P @ Ckol @ P.T
    m2c,D,_ = xp.linalg.svd(Cred)
    KL = (ssao.dm.IFF @ m2c).T

    # Define KL and m2c
    last_mode = KL[-1,:].copy() # apodizer
    first_mode = KL[0,:].copy() # 'piston'
    KL[-1,:] = first_mode
    KL[0,:] = last_mode
    last_mode_cmd = m2c[:,-1]
    first_mode_cmd = m2c[:,0]
    m2c[:,0] = last_mode_cmd.copy()
    m2c[:,-1] = first_mode_cmd.copy()

    # Save KL and m2c
    if not os.path.exists(ssao.savecalibpath):
        os.mkdir(ssao.savecalibpath)
    kl_path = os.path.join(ssao.savecalibpath,'KL.fits')
    fits.writeto(kl_path, xp.asnumpy(KL), overwrite=True)
    m2c_path = os.path.join(ssao.savecalibpath,'m2c.fits')
    fits.writeto(m2c_path, xp.asnumpy(m2c), overwrite=True)

    if show:
        plt.figure(figsize=(12,4.5))
        plt.subplot(1,2,1)
        plt.imshow(xp.asnumpy(app_phase),origin='lower',cmap='RdBu')
        plt.colorbar()
        plt.title('Apodizer phase')
        plt.subplot(1,2,2)
        show_psf(app_psf,vmin=-9,title='Apodized PSF')

        plt.figure()
        plt.plot(xp.asnumpy(D),'-o')
        plt.grid()
        plt.xscale('log')
        plt.yscale('log')
        plt.title('Modes eigenvalues')
        
        display_modes(pupil, xp.asnumpy(KL.T), N=8)

    # Calibration and loop
    amp = 25e-9
    _, IM = ssao.compute_reconstructor(ssao.sc, KL, ssao.pyr.lambdaInM, ampsInM=amp, scaleAmps=False)
    ssao.sc.load_reconstructor(IM,m2c)
    ssao.KL = KL.copy()
    err2, _ = ssao.run_loop(lambdaRef, starMagnitude=ssao.starMagnitude, save_prefix='')

    # Repeat everything with apodizer offset
    apo_ssao = SingleStageAO(tn)
    apo_ssao.initialize_turbulence(tn=atmo_tn)
    if use_slope_null:
        lambdaOverD = ssao.pyr.lambdaInM/ssao.pupilSizeInM
        nPhotons = ssao.get_photons_per_second(ssao.starMagnitude)*ssao.sc.dt
        apo_field = (1-ssao.cmask) * xp.exp(1j*app_phase,dtype=xp.cfloat)
        apo_ssao.sc.compute_slope_null(apo_field,lambdaOverD,nPhotons)
        plt.figure()
        plt.imshow(xp.asnumpy(apo_ssao.ccd.last_frame),origin='lower',cmap='RdGy')
        plt.colorbar()
        plt.title('Slope null frame')
    apo_ssao.KL = KL.copy()
    if offset_calibration:
        _, apoIM = apo_ssao.compute_reconstructor(apo_ssao.sc, KL, apo_ssao.pyr.lambdaInM, ampsInM=amp, scaleAmps=False, phase_offset=app_phase[~ssao.cmask], save_prefix=f'apo{coeff:1.1f}_offset_')
    else:
        _, apoIM = apo_ssao.compute_reconstructor(apo_ssao.sc, KL, apo_ssao.pyr.lambdaInM, ampsInM=amp, scaleAmps=False, save_prefix='apo_')
    apo_ssao.sc.load_reconstructor(apoIM,m2c)
    if use_ncpa_cmd:
        ncpa_cmd = app_cmd*lambdaRef/(2*xp.pi)
        # plt.figure()
        # plt.imshow(xp.asnumpy(reshape_on_mask(ssao.dm.IFF @ ncpa_cmd,ssao.dm.mask)),origin='lower',cmap='RdBu')
        # plt.colorbar()
        apo_err2, in_err2 = apo_ssao.run_loop(lambdaRef, starMagnitude=ssao.starMagnitude, save_prefix='apo_', ncpa_cmd=ncpa_cmd)
    else:
        apo_err2, in_err2 = apo_ssao.run_loop(lambdaRef, starMagnitude=ssao.starMagnitude, save_prefix='apo_')

    if show_contrast:
        ssao.plot_contrast(lambdaRef, frame_ids=xp.arange(max(0,ssao.Nits-100),ssao.Nits).tolist(), save_prefix='apo_', one_sided_contrast=True)

    tvec = xp.asnumpy(xp.arange(ssao.Nits)*ssao.dt*1e+3)
    plt.figure()
    plt.plot(tvec,xp.asnumpy(in_err2),'-.',label='turbulence')
    plt.plot(tvec,xp.asnumpy(err2),'-.',label='standard loop')
    plt.plot(tvec,xp.asnumpy(apo_err2),'-.',label='apodizer loop')
    plt.legend()
    plt.grid()
    plt.xlim([0.0,tvec[-1]])
    plt.xlabel('Time [ms]')
    plt.ylabel(r'$\sigma^2 [rad^2]$')
    plt.gca().set_yscale('log')

    ssao.plot_iteration(lambdaRef=lambdaRef, save_prefix='')
    apo_ssao.plot_iteration(lambdaRef=lambdaRef, save_prefix='apo_')

    plt.show()
    return ssao




def calc_psf(ef, oversampling:int=8):
    pad_width = int((max(ef.shape)*(oversampling-1)//2))
    pad_ef = xp.pad(ef, pad_width=pad_width, mode='constant', constant_values=0.0)
    ff = xp.fft.fftshift(xp.fft.fft2(pad_ef))
    return xp.real(ff * xp.conj(ff))

def show_psf(psf, norm=None, oversampling:int=8, title:str='', ext=0.3, vmin=-10):
    pixelSize = 1/oversampling
    imageHalfSizeInPoints= psf.shape[0]/2
    roi= [int(imageHalfSizeInPoints*(1-ext)), int(imageHalfSizeInPoints*(1+ext))]
    psfZoom = psf[roi[0]: roi[1], roi[0]:roi[1]]
    sz = psfZoom.shape
    if norm is None:
        norm = xp.max(psf)
    plt.imshow(xp.asnumpy(xp.log10(psfZoom/norm)), extent=
               [-sz[0]/2*pixelSize, sz[0]/2*pixelSize,
               -sz[1]/2*pixelSize, sz[1]/2*pixelSize],
               origin='lower',cmap='inferno',vmin=vmin,vmax=0)
    plt.xlabel(r'$\lambda/D$')
    plt.ylabel(r'$\lambda/D$')
    cbar= plt.colorbar()
    cbar.ax.set_title('Contrast')
    plt.title(title)

def plot_mode_j(pupil,Mat,j:int,title:str=''):
    mode = reshape_on_mask(Mat[:,j],(1-pupil).astype(bool))
    plt.imshow(masked_array(xp.asnumpy(mode),xp.asnumpy(1-pupil)),origin='lower',cmap='RdBu')
    plt.colorbar()
    plt.axis('off')
    plt.title(title)

def display_modes(pupil, Mat, N:int=8):
    nModes = xp.shape(Mat)[1]
    plt.figure(figsize=(2.25*N,12))
    for i in range(N):
        plt.subplot(6,N,i+1)
        plot_mode_j(pupil,Mat,i,title=f'Mode {i}')
        plt.subplot(6,N,i+1+N)
        plot_mode_j(pupil,Mat,i+N,title=f'Mode {i+N}')

        plt.subplot(6,N,i+1+N*2)
        plot_mode_j(pupil,Mat,nModes//2-N+i,title=f'Mode {nModes//2-N+i}')
        plt.subplot(6,N,i+1+N*3)
        plot_mode_j(pupil,Mat,nModes//2+N+i,title=f'Mode {nModes//2+N+i}')

        plt.subplot(6,N,i+1+N*4)
        plot_mode_j(pupil,Mat,nModes-2*N+i,title=f'Mode {nModes-2*N+i}')
        plt.subplot(6,N,i+1+N*5)
        plot_mode_j(pupil,Mat,nModes-N+i,title=f'Mode {nModes-N+i}')

def define_target_roi(pupil, iwa, owa, oversampling:int, symmetric:bool=False, heightInLambdaOverD=None):
    mask_shape = max(pupil.shape)
    padded_pupil = xp.pad(1-pupil.copy(), pad_width=int((mask_shape*(oversampling-1)//2)), mode='constant', constant_values=0.0)
    X,Y = image_grid(padded_pupil.shape,recenter=True)
    rho = xp.sqrt(X**2+Y**2)
    if symmetric is True:
        where = (rho <= owa*oversampling) * (rho >= iwa*oversampling)
    else:
        where = (rho <= owa*oversampling) * (X >= iwa*oversampling) 
    if heightInLambdaOverD is not None:
        where *= (abs(Y) < heightInLambdaOverD/2*oversampling)
    return where

def get_apodizer_phase(pupil, target_contrast, oversampling:int=4,
                       max_its:int=20, beta:float=0, IF=None):
    N = max(pupil.shape)
    pad_width = N*(oversampling-1)//2
    pad_pupil = xp.pad(pupil, pad_width, mode='constant', constant_values=0.0)
    dark_zone = target_contrast < 0.1

    a = pad_pupil.copy().astype(xp.complex64)
    old_ff = None

    if IF is not None:
        Rec = xp.linalg.pinv(IF)
    for it in range(max_its):
        ff = xp.fft.fftshift(xp.fft.fft2(a))
        if not xp.any(xp.abs(ff)**2 / xp.max(xp.abs(ff)**2) > target_contrast):
            break
        new_ff = ff.copy()
        if beta != 0 and old_ff is not None:
            new_ff[dark_zone] = old_ff[dark_zone] * beta - new_ff[dark_zone] * (1 + beta)
        else:
            new_ff[dark_zone] = 0
        old_ff = new_ff.copy()
        a = xp.fft.ifft2(xp.fft.ifftshift(new_ff))
        a[~pad_pupil.astype(bool)] = 0
        phase = xp.angle(a)
        if IF is not None:
            cmd = Rec @ phase[pad_pupil.astype(bool)]
            phase = reshape_on_mask(IF @ cmd,(1-pad_pupil).astype(bool))
        a = xp.abs(pad_pupil) * xp.exp(1j*(phase),dtype=xp.complex64) 
    psf = xp.abs(ff)**2
    ref_psf = xp.abs(xp.fft.fftshift(xp.fft.fft2(pupil)))**2
    contrast =  psf / xp.max(ref_psf)
    if it == max_its-1:
        print(f'Maximum number of iterations ({max_its:1.0f}) reached, worst contrast in dark hole is: {xp.log10(xp.max(contrast[dark_zone])):1.1f}')
    else:
        print(f'Apodizer computed in {it:1.0f} iterations: average contrast in dark hole is {xp.mean(xp.log10(contrast[dark_zone])):1.1f}, Strehl is {xp.max(psf)/xp.max(ref_psf)*1e+2:1.2f}%')
    pad_phase = xp.angle(a)
    phase = reshape_on_mask(pad_phase[pad_pupil.astype(bool)],(1-pupil).astype(bool))
    return phase

if __name__ == '__main__':
    ssao = main()