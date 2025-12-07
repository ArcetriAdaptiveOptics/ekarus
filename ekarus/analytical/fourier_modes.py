import xupy as xp
from ekarus.e2e.utils.image_utils import image_grid

def define_fourier_basis(mask, max_freq:int, thr:float=5e-2):
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
    freqs = xp.arange(max_freq, dtype=int)
    modes = []
    for fx in freqs:
        for fy in freqs:
            if fx == 0 and fy == 0:
                continue
            arg = xp.pi * (fx * Xv + fy * Yv)
            modes.append(xp.cos(arg))
            modes.append(xp.sin(arg))
    fourier_mat = xp.vstack(modes)  # shape (n_modes, M)

    # remove near-duplicate / zero-energy rows (numerical safety)
    norms = xp.linalg.norm(fourier_mat, axis=1)
    keep = norms > (thr * xp.max(norms))
    fourier_mat = fourier_mat[keep, :]

    return fourier_mat

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