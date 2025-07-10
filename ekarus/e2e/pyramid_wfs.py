import numpy as np

class PyramidWFS:
    """
    Optical modeling of a pyramid wavefront sensor for adaptive optics.

    This class simulates the propagation of an electric field through a pyramid
    wavefront sensor, applying a phase mask that simulates a 4-faces pyramid,
    where the phase shift depends on the distance from the apex.
    """

    def __init__(self, apex_angle):
        """
        Inizializza il sensore a piramide.

        :param apex_angle: angolo al vertice della piramide (in radianti)
        """
        self.apex_angle = apex_angle

    def pyramid_phase_delay(self, shape):
        """
        Calcola il ritardo di fase introdotto dalla piramide nel piano focale.
        Il ritardo dipende dalla distanza dal vertice e dalla faccia della piramide.

        :param shape: tuple (ny, nx) dimensioni del campo
        :return: array numpy 2D float (ritardo di fase in radianti)
        """
        ny, nx = shape
        cx, cy = nx // 2, ny // 2

        # Coordinate rispetto al centro
        x = np.arange(nx) - cx
        y = np.arange(ny) - cy
        X, Y = np.meshgrid(x, y)

        # Calcola il ritardo di fase: ogni faccia ha una pendenza diversa
        phase_delay = np.zeros_like(X, dtype=np.float64)

        # TODO add the phase delay for each face of the pyramid

        return phase_delay

    def propagate(self, input_field):
        """
        Propagate the electric field through the pyramid:
        1. From the pupil plane to the focal plane (FFT)
        2. Apply the pyramid phase delay (shift each point by the phase delay)
        3. From the focal plane to the output pupil plane (IFFT)

        :param input_field: complex array numpy 2D representing the input electric field
        :return: complex array numpy 2D representing the output electric field
        """
        # 1. Propagation from the pupil plane to the focal plane (Fourier transform)
        self.field_on_focal_plane =  TODO 

        # 2. Calcola il ritardo di fase della piramide
        phase_delay = self.pyramid_phase_delay(self.field_on_focal_plane.shape)

        # 2b. Applica il ritardo di fase: ogni punto viene ritardato di phase_delay (fase)
        self._ef_focal_plane_delayed = TODO

        # 3. Inverse propagation from the focal plane to the output pupil plane (inverse transform)
        output_field = TODO(self._ef_focal_plane_delayed)

        return output_field