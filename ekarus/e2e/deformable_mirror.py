import numpy as np
import matplotlib.pyplot as plt

from ekarus.e2e.utils.deformable_mirror_utilities import slaving, compute_reconstructor
from ekarus.e2e.utils.alpao_initializer import init_ALPAO


class DeformableMirror():
    """ 
    This class defines the methods for a generic deformable mirror.
    It implements the following functions:
        
        - plot_surface(): plot the current shape or a shape in input
        
        - plot_position(): plot a command or the actuator position at the actuator coordinates

        - set_position(): apply a zonal or modal command to the DM
        
    """
    
    def __init__(self, input):
        self.mask, self.act_coords, self.pixel_scale, self.IFF = init_ALPAO(input)

        # initialize mirror surface and actuator position
        self.Nacts = np.shape(self.act_coords)[1]
        self.act_pos = np.zeros(self.Nacts)
        self.surface = self.IFF @ self.act_pos

        # compute reconstructor
        self.R, self.U = compute_reconstructor(self.IFF)


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
        
        mask_ids = np.arange(np.size(self.mask))
        pix_ids = mask_ids[~(self.mask).flatten()]
            
        image = np.zeros(np.size(self.mask))
        image[pix_ids] = surf2plot
        image = np.reshape(image, np.shape(self.mask))
        
        if plt_mask is None:
            plt_mask = self.mask
        else:
            plt_mask = np.logical_or(self.mask, plt_mask)
        
        image = np.ma.masked_array(image, plt_mask)
        
        #plt.figure(figsize=(10,10))  
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
        
        act_pix_size = 3
        if np.sum(self.Nacts) < 100:
            act_pix_size = 12
        
        #plt.figure(figsize = (10,10))
        plt.scatter(x,y, c=pos, s=act_pix_size**2, cmap='hot')
        plt.axis('equal')
        plt.colorbar()
    

    
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
            
            # length check
            mode_amps = cmd_amps.copy()
            n_modes = np.shape(self.U)[1]
            if len(mode_amps) < n_modes:
                mode_amps = np.zeros(n_modes)
                mode_amps[:len(cmd_amps)] = cmd_amps
        
            shape = self.U @ mode_amps
            cmd_amps = self.R @ shape
        
        # Position (zonal) command
        if absolute:
            cmd_amps -= self.act_pos

        # Update positions and shape
        self.act_pos += cmd_amps
        self.surface += self.IFF @ cmd_amps

 

    def _slave_cmd(self, cmd, max_cmd: float = 0.9):
        """ DM utils slaving function wrapper """

        slaved_cmd = slaving(self.act_coords, cmd, slaving_method = 'interp', cmd_thr = max_cmd)
        return slaved_cmd


    