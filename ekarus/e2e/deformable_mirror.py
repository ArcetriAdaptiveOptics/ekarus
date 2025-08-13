import numpy as np
import matplotlib.pyplot as plt

from numpy.ma import masked_array


class DeformableMirror():
    """ 
    This class defines the methods for a generic deformable mirror.
    It implements the following functions:

        - get_position(): returns the position of the DM actuators

        - set_position(): apply a zonal or modal command to the DM
        
        - plot_surface(): plot the current shape or a shape in input
        
        - plot_position(): plot a command or the actuator position at the actuator coordinates

    """
    
    def __init__(self, xp=np):
        self._xp = xp
        self.dtype = xp.float32 if xp.__name__ == 'cupy' else xp.float64


    def get_position(self):
        """ Returns the current actuator position """
        return self.act_pos
    
    
    def set_position(self, cmd_amps, absolute:bool = False, modal:bool = False):
        """
        Computes and applies a zonal/modal amplitude
        command to the DM

        Parameters
        ----------
        cmd_amps : ndarray(float) [Nacts]
            The array of zonal (modal) amplitudes.
            
        absolute : bool, optional
            True if the command is absolute and not
            relative to the current actuators' position.
            The default is False.
            
        modal : bool, optional
            True for modal amplitudes. The default is False.

        """
        
        if modal: # convert modal to zonal
            mode_amps = cmd_amps.copy()
            n_modes = self._xp.shape(self.U)[1]
            if len(mode_amps) < n_modes:
                mode_amps = self._xp.zeros(n_modes, dtype = self.dtype)
                mode_amps[:len(cmd_amps)] = cmd_amps
        
            shape = self.U @ mode_amps
            cmd_amps = self.R @ shape
        
        if absolute:
            cmd_amps -= self.act_pos

        # Update positions and shape
        self.act_pos += cmd_amps
        self.surface += self.IFF @ cmd_amps
        


    def plot_surface(self, surf2plot = None, title:str = '', plt_mask = None):
        """
        Plots surf2plot or (default) the segment's
        current shape on the DM mask

        Parameters
        ----------
        surf2plot : ndarray(float) [Npix], optional
            The shape to plot. Defaults to the DM current shape.
            
        plt_title : str, optional
            The plot title. The default is no title.
            
        plt_mask : ndarray(bool), optional
            The mask to use on the plot. Default is None.

        """
        
        if surf2plot is None:
            surf2plot = self.surface
        
        mask_ids = self._xp.arange(self._xp.size(self.mask))
        pix_ids = mask_ids[~(self.mask).flatten()]
            
        image = self._xp.zeros(self._xp.size(self.mask), dtype = self.dtype)
        image[pix_ids] = surf2plot
        image = self._xp.reshape(image, self.mask.shape)
        
        plt_mask = self._xp.logical_or(self.mask, plt_mask) if plt_mask is not None else self.mask.copy()

        if hasattr(image, 'get'):
            image = image.get()

        if hasattr(plt_mask, 'get'):
            plt_mask = plt_mask.get()
        
        image = masked_array(image, plt_mask)
        plt.imshow(image, origin = 'lower', cmap = 'hot')
        
        img_rms = np.std(image.data[~image.mask])
        
        if title == '':
            title = f"RMS: {img_rms:.2e}"
            
        plt.axis('off')
        plt.title(title)
        
        return img_rms
    
        

    def plot_position(self, pos = None):
        """
        Plots pos on the actuator coordinates.
        Defaults to current actuator position

        Parameters
        -------
        pos : ndarray(float) [Nacts,]
            The array of position/commands to be plot.
            Default: current actuator positions on the DM

        """
        
        if pos is None:
            pos = self.act_pos.copy()
        x,y = self.act_coords[0,:], self.act_coords[1,:] 
        
        act_pix_size = 12 if self.Nacts < 100 else 3

        if hasattr(pos, 'get'):
            pos = pos.get()

        if hasattr(x, 'get'):
            x = x.get()
            y = y.get()
        
        plt.scatter(x,y, c=pos, s=act_pix_size**2, cmap='hot')
        plt.axis('equal')
        plt.colorbar()
    




    