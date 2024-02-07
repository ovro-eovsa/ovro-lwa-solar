import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from casatools import table, msmetadata
# the functions to plot the data

def inspection_bl_flag(ms_file):
    """
    Function to inspect the baseline flagging of the data
    
    :param ms_file: str : path to the measurement set file
    """


    tb = table()
    msmd = msmetadata()
    tb.open(ms_file)

    # Extract DATA Column
    visibility_data = tb.getcol('DATA')

    # Extract UVW Column
    uvw_data = tb.getcol('UVW')

    # extract ant1 and2 columns
    ant1 = tb.getcol('ANTENNA1')
    ant2 = tb.getcol('ANTENNA2')

    # extract location of antennas
    antenna_positions = msmd.antennaposition()
    antenna_names = msmd.antennanames()

    u_col = uvw_data[0, :]
    v_col = uvw_data[1, :]
    w_col = uvw_data[2, :]

    stokes_I = 0.5 * (visibility_data[0,:,:] + visibility_data[3,:,:])

    # Extract FLAG Column
    flag_data = tb.getcol('FLAG')


    # Close the table
    tb.close()

    img_cross = np.zeros((352, 352))

    for idx in range(stokes_I.shape[1]):
        # img_cross[ant1[idx], ant2[idx]] = np.mean(np.abs(stokes_I[:,idx]), axis=0)
        # insert flag_data
        img_cross[ant1[idx], ant2[idx]] = np.mean(np.abs(flag_data[0,:,idx]), axis=0)
        img_cross[ant2[idx], ant1[idx]] = np.mean(np.abs(flag_data[0,:,idx]), axis=0)

    fig_plt  = plt.imshow((img_cross), cmap='viridis', origin='lower', norm=mcolors.PowerNorm(0.5))
    return fig_plt