from arte.atmo.phase_screen_generator import PhaseScreenGenerator


def turbulence_generator(lambdaInM, r0, L0, Nscreens, screenSizeInPixels, screenSizeInMeters, savepath:str= None):

    phs = PhaseScreenGenerator(screenSizeInPixels, screenSizeInMeters, \
                               outerScaleInMeters=L0, seed=42)
    
    phs.generate_normalized_phase_screens(Nscreens)
    phs.rescale_to(r0At500nm=r0)
    phs.get_in_radians_at(wavelengthInMeters=lambdaInM)

    if savepath is not None :
        phs.save_normalized_phase_screens(savepath)

    return phs._phaseScreens