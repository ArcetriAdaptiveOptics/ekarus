
import numpy as np
import matplotlib.pyplot as plt
from ekarus.e2e.pyramid_wfs import PyramidWFS
from arte.types.mask import CircularMask
from ekarus.e2e.utils.create_field_from_zernike_coefficients import create_field_from_zernike_coefficients

def main():
    nx = 128
    # Parametri del sensore a piramide
    apex_angle = TODO  # vertex angle in radians, can be tricky to find the right value
    
    # Create pupil mask
    oversampling = 4
    mask = CircularMask((oversampling * nx, oversampling * nx), maskRadius=nx // 2)

    # Create the input electric field for flat wavefront (a piston of 1 radians)
    input_field = create_field_from_zernike_coefficients(mask, 1, 1)

    # Inizializzazione del sensore
    wfs = PyramidWFS(apex_angle)
    # Propagazione del campo attraverso il sensore
    output_field = wfs.propagate(input_field)

    # Visualizzazione dei risultati

    plt.figure(1, figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.title("Input Electric Field (Phase)")
    plt.imshow(np.angle(input_field), cmap='twilight')
    plt.colorbar()
    plt.subplot(1, 3, 2)
    plt.title("PSF on Focal Plane (Intensity) - Log Scale")
    plt.imshow(np.log(np.abs(wfs.field_on_focal_plane**2)), cmap='inferno')
    plt.colorbar()
    plt.subplot(1, 3, 3)
    plt.title("Output Electric Field (Intensity)")
    plt.imshow(np.abs(output_field**2))
    plt.colorbar()
    plt.show()
    
    plt.figure(2)
    plt.imshow(wfs.pyramid_phase_delay((nx, nx)), cmap='twilight')
    plt.title("Pyramid phase mask")
    plt.colorbar()
    plt.show()

    return wfs, input_field, output_field