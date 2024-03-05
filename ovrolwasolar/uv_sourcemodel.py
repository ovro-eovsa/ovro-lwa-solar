import casatools
import casatasks
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from ovrolwasolar import utils

def func_elip_gauss(uv, a, b, theta, amp):
    u, v = uv
    return amp/(2*np.pi) * np.exp(- 2*np.pi**2 * ((u)*np.cos(theta) + (v)*np.sin(theta))**2 * (a**2) 
                                  - 2*np.pi**2 * ((u)*np.sin(theta) - (v)*np.cos(theta))**2 * (b**2))
    #return amp/2/np.pi * np.exp(-0.5 * (a**2 * (u*np.cos(theta) - v*np.sin(theta))**2 + b**2*(u*np.sin(theta) + v*np.cos(theta))**2))

def func_phase_sin(uv, l0, m0):
    u, v = uv
    return (np.sin(2*np.pi*(u*l0 + v*m0)))

from scipy.optimize import curve_fit


def uv_tapper_weight(uvw, uv_tapper_factor=1):
    """ make a function, reads in uvw, returns the weight
    weight scheme is guassian distribution with center at u,v = 0,0
    use ``sigma_norm  = np.inf`` to turn off the taper
    The uv_taper_factor is applied after normalising all uvs
    with respect to the max uv_dist.

    :param uvw: uvw coordinates
    :param uv_tapper_factor: factor to control the tapering, normlised to max uv distance
    """
    uv_dist = np.sqrt(uvw[0]**2 + uvw[1]**2)
    uv_dist_norm = uv_dist / np.max(uv_dist)
    weight = np.exp(-uv_dist_norm**2 / (2 * uv_tapper_factor**2))
    return weight

from astropy.coordinates import SkyCoord
from astropy import units

def lm_to_radec(l, m, ref_ra, ref_dec):
    """
    Convert (l, m) coordinates to RA and DEC.

    :param l: Direction cosine, dimensionless (assumed to be small for this calculation).
    :param m: Direction cosine, dimensionless (assumed to be small for this calculation).
    :param ref_ra: Reference right ascension in degrees.
    :param ref_dec: Reference declination in degrees.

    :return: RA, DEC in rad.
    

    """

    # Calculate DEC using the approximation for small angles
    dec = np.arcsin(np.sin(ref_dec) * np.sqrt(1 - l**2 - m**2) + m * np.cos(ref_dec)) 

    # Calculate RA using the approximation for small angles, and adjust for RA wrapping
    ra = ( ref_ra + np.arctan2(l, np.sqrt(1 - l**2 - m**2)* np.cos(ref_dec) - m * np.sin(ref_dec)))

    if ra > np.pi:
        ra -= 2*np.pi
    if ra < -np.pi:
        ra += 2*np.pi 
    
    return ra, dec

def fast_vis_1gauss(fname_ms,
                    uv_tapper_factor =0.3,\
                    datacolumn=None):
    """
    This function fits the fast visibilities using a 2D gaussian. Since we are only
    using a single gaussian, it is advised that all strong sources are subtracted 
    before using this functionality. The function loops over all channels and all
    scans to produce fitted values. It will return the source locations in (l,m)
    coordinate sysytem, the fitted gaussian parameters. The returned list has the 
    following format:
    
    [[scan 1, channel 1],[scan 2, channel 1],...[scan M, channel 1],\
    [[scan 1, channel 2],[scan 2, channel 2],...[scan M, channel 2],\
    ...
    ...
    ...
    [[scan 1, channel N],[scan 2, channel N],...[scan M, channel N]]
    
    All the returned quantities are lists
    

    :param fname_ms: Name of the fast vis MS
    :param uv_tapper_factor: This is the factor which is passed to uv_tapper_weight. This
    is used to weight the visibilties of different baselines.
    :param datacolumn: Which datacolumn of MS to use. Default: None. If None, will check
    if corrected data is present. If present use corrected data. Else data.

    :return: Fitted gaussian, Parameters are ordered in this manner (sigma_l,sigma_m,theta, amp)
    

    """
# extract the uv and vis and the scan number
# get flag
    
    if datacolumn is None:
        if utils.check_corrected_data_present(fname_ms):
            datacolumn='CORRECTED_DATA'
        else:
            datacolumn='DATA'
    tb = casatools.table()
    tb.open(fname_ms)
    uvw = tb.getcol('UVW')
    data = tb.getcol(datacolumn)
    scan = tb.getcol('SCAN_NUMBER')
    flag = tb.getcol('FLAG')
    tb.close()

    spTB = casatools.table()
    spTB.open(fname_ms + '/SPECTRAL_WINDOW')
    chan_freqs = spTB.getcol('CHAN_FREQ')
    spTB.close()

    tb.open(fname_ms + '/FIELD')
    ref_ra_dec = tb.getcol('PHASE_DIR').squeeze()
    tb.close()
    # convert the uvw to wavelength lambda


    stokes_I_all_ch = (0.5 * (data[0,:,:] + data[1,:,:])).squeeze()

    flag_I = np.any(np.any(flag, axis=0),axis=0)

    popt_list = []
    popt_phase_list = []
    ref_proc_list = []

    scan_unique = np.unique(scan)
    print(stokes_I_all_ch.shape)
    for i in range(stokes_I_all_ch.shape[0]): # channel

        wavelen = 3e8 / (chan_freqs[i])
        uvw = uvw / wavelen

        popt_list_tmp = []
        popt_phase_list_tmp = []
        ref_proc_list_tmp = []
        for time_idx,j in enumerate(scan_unique): # time slot
            
            ind = np.where(scan == j)[0]
            uvw_this_scan_goodvis = uvw[:,ind][:,~flag_I[ind]]

            u = uvw_this_scan_goodvis[0]
            v = uvw_this_scan_goodvis[1]
#            print(u.shape, flag_I[i,ind].shape)
            stokes_I_this_scan_goodvis = stokes_I_all_ch[:,ind][i, ~flag_I[ind]]
            stokes_I_amp = np.abs(stokes_I_this_scan_goodvis)
            phase_angle = np.angle(stokes_I_this_scan_goodvis)

            weight_this_scan = uv_tapper_weight(uvw_this_scan_goodvis, uv_tapper_factor=uv_tapper_factor)
            popt, pcov = curve_fit(func_elip_gauss, (u, v), stokes_I_amp,
                p0=[1/np.max(u)/2,1/np.max(v)/2,0,np.max(stokes_I_amp)],sigma = 1/weight_this_scan**2,
                maxfev = 5000)

            popt_phase, pcov_phase = curve_fit(func_phase_sin, 
                (u, v), (np.sin(phase_angle)),sigma = 1/weight_this_scan**2, 
                  p0=[0,0], absolute_sigma=True)

            ref_proc = ref_ra_dec[:,time_idx]
            
            l0, m0 = popt_phase
            alpha_0, delta_0 = ref_proc   
            alpha_rad, delta_rad = lm_to_radec(l0, m0, alpha_0, delta_0)

            popt_list_tmp.append(popt)
            popt_phase_list_tmp.append([alpha_rad, delta_rad])
            ref_proc_list_tmp.append(ref_proc)
        
        popt_list.append(popt_list_tmp)
        popt_phase_list.append(popt_phase_list_tmp)
        ref_proc_list.append(ref_proc_list_tmp)

    return popt_list, popt_phase_list, ref_proc_list

def wrap_solution_save_hdf5(source_morphology, source_radec, phasecenters, fname_hdf5):
    """
    This function writes the fitted parameters into hdf5 file. The dataset names of the
    hdf5 file  are "source_morphology","source_radec","phasecenters"
    """
    import h5py
    with h5py.File(fname_hdf5, 'w') as f:
        f.create_dataset('source_morphology', data=np.array(source_morphology))
        f.create_dataset('source_radec', data=np.array(source_radec))
        f.create_dataset('phasecenters', data=np.array(phasecenters))

def plot_img_from_uvparm(source_morphology, source_radec, ref_proc): 
    """
    This function plots the source in RA-DEC frame. Mostly for
    debugging purposes.

    :param source_morphology: List/array with [sigma_l,sigma_m, theta, amp]
                        of the fitted gaussian
    :param source_radec: Source ra dec
    :param ref_proc: Reference phase center

    """                       
    sigma_x = source_morphology[0]
    sigma_y = source_morphology[1]
    theta = -source_morphology[2]
    amp = source_morphology[3]

    alpha_0, delta_0 = ref_proc
    alpha_rad, delta_rad = source_radec

    def func_gauss_xy(x, y, x0, y0, sigma_x, sigma_y, theta, amp):
        xx = x - x0
        yy = y - y0
        return amp/np.abs(4*np.pi**2 *sigma_x*sigma_y) * np.exp(
            -0.5 * (xx*np.cos(theta) - yy*np.sin(theta))**2 / sigma_x**2 
            -0.5 * (xx*np.sin(theta) + yy*np.cos(theta))**2 / sigma_y**2)

    x = np.linspace(-.015+alpha_0, .015+alpha_0, 300)
    y = np.linspace(-.015+delta_0, .015+delta_0, 300)


    X, Y = np.meshgrid(x, y)
    Z = func_gauss_xy(X, Y, alpha_rad, delta_rad, sigma_x, sigma_y, theta, amp)
    fig, ax = plt.subplots(1,1, figsize=(5,5))
    ax.contourf(X, Y, Z, cmap='gray_r', origin='lower')
    ax.set_xlim(alpha_0+.015, alpha_0-.015)
    ax.set_aspect('equal', 'box')

    r_sun_rad  = 959.63 / 3600 * np.pi/180
    ax.plot(alpha_0 + r_sun_rad * np.cos(np.linspace(0, 2*np.pi, 100)), delta_0 + r_sun_rad * np.sin(np.linspace(0, 2*np.pi, 100)), color='red')

    ax.plot(alpha_0, delta_0, '+', color='red')
    ax.set_xlabel('RA (rad)')
    ax.set_ylabel('DEC (rad)')

    return fig, ax
