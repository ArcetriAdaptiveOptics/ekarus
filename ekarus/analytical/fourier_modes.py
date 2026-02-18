import xupy as xp
from ekarus.e2e.utils.image_utils import image_grid

def define_fourier_basis(mask, max_freq:int, thr:float=None):
    """
    Build a real Fourier modal basis sampled on the valid pixels inside `mask`.
    - mask: boolean array where True = masked/outside pupil (same convention as notebook).
    - max_freq: maximum integer spatial frequency in each axis (frequencies range -max_freq..max_freq).
    - orthonormalize: if True, attempt to orthonormalize the modes (QR on the sampling matrix).
    - use_cos_sin: if True produce separate cos and sin components (real basis).
    Returns:
    - fourier_mat: xp.array shape (n_modes, n_valid_pixels)
    """
    X, Y = image_grid(mask.shape, recenter=True)
    Xv = X[~mask]/(xp.max(X)-xp.min(X))
    Yv = Y[~mask]/(xp.max(Y)-xp.min(Y))
    freqs = xp.arange(1,max_freq, dtype=int)
    modes = []
    spatial_freqs = []  # Track spatial frequency magnitude for each mode
    for fx in freqs:
        for fy in freqs:
            arg = xp.pi * (fx * Xv + fy * Yv)
            modes.append(xp.cos(arg))
            modes.append(xp.sin(arg))
            spatial_freqs.append(xp.sqrt(fx**2 + fy**2))
            spatial_freqs.append(xp.sqrt(fx**2 + fy**2))
    fourier_mat = xp.vstack(modes)  # shape (n_modes, M)
    spatial_freqs = xp.array(spatial_freqs)

    if thr is not None:
        # remove near-duplicate / zero-energy rows (numerical safety)
        norms = xp.linalg.norm(fourier_mat, axis=1)
        keep = norms > (thr * xp.max(norms))
        fourier_mat = fourier_mat[keep, :]
        spatial_freqs = spatial_freqs[keep]

    # Normalization
    norm = xp.std(fourier_mat,axis=1)
    fourier_modes = (fourier_mat.T/norm).T
    
    # Sorting by ascending spatial frequency
    sort_idx = xp.argsort(spatial_freqs)
    fourier_modes = fourier_modes[sort_idx, :]

    return fourier_modes

# def make_fourier_basis(coordinates, frequencies, sort_by_energy=True):
#     '''Make a Fourier basis.
#     '''
#     modes_cos = []
#     modes_sin = []
#     energies = []
#     ignore_list = []

#     for i, freq in enumerate(frequencies):
#         if i in ignore_list:
#             continue

#         mode_cos = xp.cos(xp.dot(freq, coordinates))
#         mode_sin = xp.sin(xp.dot(freq, coordinates))

#         modes_cos.append(mode_cos)
#         modes_sin.append(mode_sin)

#         j = xp.argmin(frequencies+freq)

#         dist = frequencies[j] + freq
#         dist2 = xp.dot(dist, dist)

#         p_length2 = xp.dot(freq, freq)
#         energies.append(p_length2)

#         if dist2 < (_epsilon * p_length2):
#             ignore_list.append(j)

#     if sort_by_energy:
#         ind = xp.argsort(energies)
#         modes_sin = [modes_sin[i] for i in ind]
#         modes_cos = [modes_cos[i] for i in ind]
#         energies = xp.array(energies)[ind]

#     modes = []
#     for i, E in enumerate(energies):
#         # Filter out and correctly normalize zero energy vs non-zero energy modes.
#         if E > _epsilon:
#             modes.append(modes_cos[i] * xp.sqrt(2))
#             modes.append(modes_sin[i] * xp.sqrt(2))
#         else:
#             modes.append(modes_cos[i])

#     return xp.array(modes)

# def make_complex_fourier_basis(grid, fourier_grid, sort_by_energy=True):
#     '''Make a complex Fourier basis.

#     Fourier modes this function are defined to be complex. For each point in `fourier_grid` the complex Fourier mode is contained in the output.

#     Parameters
#     ----------
#     grid : Grid
#         The :class:`Grid` on which to calculate the modes.
#     fourier_grid : Grid
#         The grid defining all frequencies.
#     sort_by_energy : bool
#         Whether to sort by increasing energy or not.

#     Returns
#     -------
#     ModeBasis
#         The mode basis containing all Fourier modes.
#     '''
#     c = xp.array(grid.coords)

#     modes = [Field(xp.exp(1j * xp.dot(p, c)), grid) for p in fourier_grid.points]
#     energies = [xp.dot(p, p) for p in fourier_grid.points]

#     if sort_by_energy:
#         ind = xp.argsort(energies)
#         modes = [modes[i] for i in ind]

#     return ModeBasis(modes, grid)