from ekarus.dataelab import factory

tag_400KL='20260723_144418'
tag_400KL_turbo='20260723_144441'
tag_400KL_turbo_OL='20260723_144457'

def main(tag):
    ff = factory.get_factory('lorenzo')
    # Load analyzer (use .get() for automatich caching)
    an = ff.analyzer(tag)

    # raw data (return array numpy)
    ocam_data = an.ocam_frames.get_data()          # shape: (nframes, y, x)
    modes_data = an.residual_modes.get_data()      # shape: (nframes, nmodes)
    dm_data = an.dm_commands.get_data()             # shape: (nframes, nacts)

    # Control matrices
    s2m_matrix = an.s2m.get_data()                  # shape: (nmodes, nsignals)
    m2c_matrix = an.m2c.get_data()                  # shape: (nacts, nmodes)


    # temporal stats
    modes_mean = an.residual_modes.time_mean.get_data()      # shape: (nmodes,)
    modes_std = an.residual_modes.time_std.get_data()        # shape: (nmodes,)
    modes_rms = an.residual_modes.time_rms.get_data()        # shape: (nmodes,)

    # modal PSD
    modes_psd = an.residual_modes.power()

    # Select temporal subset of frames
    modes_subset = an.residual_modes.get_data(times=(10, 100))  # frames da 10 a 100

    # Extract single frame
    frame_10 = an.ocam_frames.get_data()[10]        # shape: (y, x)

    # Extract single mode in time
    mode_5 = an.residual_modes.get_data()[:, 5]    # shape: (nframes,)

    # Use get_index_of for more advanced access (if supported)

    # Print info
    an.summary()

    # Info as dict
    info = an.info()
    print(f"Frames Ocam: {info['nframes_ocam']}")
    print(f"Frames residual modes: {info.get('nframes_residual_modes', info.get('nframes_modes','N/A'))}")
    print(f"Frames DM: {info['nframes_dm']}")

    return an

if __name__ == '__main__':
    main()