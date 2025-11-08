from ekarus.dataelab import PapyrusAnalyzer


def main():
    fname='/Users/lbusoni/Downloads/2025-10-13T22_24_56_telemetry_data.npy'
    # Caricare l'analyzer (usa il metodo .get() per caching automatico)
    an = PapyrusAnalyzer.get(fname)

    # Accedere ai dati raw (restituisce array numpy)
    ocam_data = an.ocam_frames.get_data()          # shape: (nframes, y, x)
    modes_data = an.residual_modes.get_data()      # shape: (nframes, nmodes)
    dm_data = an.dm_commands.get_data()             # shape: (nframes, nacts)

    # Accedere alle matrici di ricostruzione
    s2m_matrix = an.s2m.get_data()                  # shape: (nmodes, nsignals)
    m2c_matrix = an.m2c.get_data()                  # shape: (nacts, nmodes)


    # Calcolare media e std nel tempo
    modes_mean = an.residual_modes.time_average()      # shape: (nmodes,)
    modes_std = an.residual_modes.time_std()        # shape: (nmodes,)
    modes_rms = an.residual_modes.time_rms()        # shape: (nmodes,)

    # Calcolare PSD
    modes_psd = an.residual_modes.power()

    # Selezionare intervalli temporali
    modes_subset = an.residual_modes.get_data(times=(10, 100))  # frames da 10 a 100

    # Prendere un singolo frame
    frame_10 = an.ocam_frames.get_data()[10]        # shape: (y, x)

    # Prendere un singolo modo nel tempo
    mode_5 = an.residual_modes.get_data()[:, 5]    # shape: (nframes,)

    # Usare get_index_of per accessi più avanzati (se supportato)

    # Stampare info
    an.summary()

    # Info come dict
    info = an.info()
    print(f"Frames Ocam: {info['nframes_ocam']}")
    print(f"Frames residual modes: {info.get('nframes_residual_modes', info.get('nframes_modes','N/A'))}")
    print(f"Frames DM: {info['nframes_dm']}")

    return an

if __name__ == '__main__':
    main()