from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import glob
# keep only the largest connected component
from skimage import measure
from skimage.morphology import remove_small_objects
from scipy.ndimage import center_of_mass
from suncasa.io import ndfits

from scipy.ndimage import binary_erosion, binary_dilation, binary_closing
import sunpy.map as smap


def thresh_func(freq): # freq in Hz
    """Return the threshold for the given frequency
    
    :param freq: frequency in Hz

    :return: threshold in Tb
    """

    return 1.1e6 * (1-1.8e4* freq**(-0.6))


def find_center_of_thresh(data_this, thresh, meta):
    """
    Find the center of the thresholded image
    
    :param data_this: the data to be thresholded
    :param thresh: the threshold
    :param meta: the meta data of the fits file
    
    """
    threshed_img = (data_this > thresh)
    threshed_img_1st = remove_small_objects(threshed_img, min_size=1000, connectivity=1)
    # perform erosion to remove the small features
    threshed_img_2nd = binary_erosion(threshed_img_1st, iterations=3)

    # keep only the largest connected component
    threshed_img_3rd = remove_small_objects(threshed_img_2nd, min_size=1000, connectivity=1)    

    # dialate the image back to the original size
    threshed_img_4th = binary_dilation(threshed_img_3rd, iterations=3)

    # find the centroid of threshed_img_1st, coords in x_arr, y_arr
    com = center_of_mass(threshed_img_4th)
    # convert to arcsec

    x_arr = meta['CRVAL1'] + meta['CDELT1'] * (np.arange(meta['NAXIS1']) - (meta['CRPIX1'] - 1))
    y_arr = meta['CRVAL2'] + meta['CDELT2'] * (np.arange(meta['NAXIS2']) - (meta['CRPIX2'] - 1))

    # convert com from pixel to arcsec (linear)
    com_x_arcsec = x_arr[0] + com[1] * (x_arr[-1] - x_arr[0]) / (len(x_arr) - 1)
    com_y_arcsec = y_arr[0] + com[0] * (y_arr[-1] - y_arr[0]) / (len(y_arr) - 1)


    # move x_arr, y_arr to the center of the image
    x_arr_new = x_arr - com_x_arcsec
    y_arr_new = y_arr - com_y_arcsec

    return [com_x_arcsec, com_y_arcsec, com, x_arr_new, y_arr_new, threshed_img,
            threshed_img_1st, threshed_img_2nd, threshed_img_3rd, threshed_img_4th]

def refraction_fit_param(fname):
    """
    Take in a multi-frequency fits file and return the refraction fit parameters for:
    ```
    x = px[0] * 1/freq**2 + px[1]
    y = py[0] * 1/freq**2 + py[1]
    ```
    
    :param fname: the fits file name
    """

    meta,data = ndfits.read(fname)
#    hdu = fits.open(fname)
#    data = hdu[0].data
#    meta = hdu[0].header

    freqs_arr = meta['ref_cfreqs'] # Hz
    header_meta = meta["header"]

    com_x_arr = []
    com_y_arr = []
    peak_values_tmp = []
    area_collect_tmp = []
    for idx_this,idx_img in enumerate(range(0, freqs_arr.shape[0])):
        thresh = thresh_func(freqs_arr[idx_img])/5
        data_this =  np.squeeze(data[0,idx_img,:,:])
        (com_x_arcsec, com_y_arcsec, com, x_arr_new, y_arr_new, threshed_img,
            threshed_img_1st, threshed_img_2nd, threshed_img_3rd, threshed_img_4th
                ) = find_center_of_thresh(data_this, thresh, header_meta)
        peak_values_tmp.append(np.nanmax(data_this))
        com_x_arr.append(com_x_arcsec)
        com_y_arr.append(com_y_arcsec)
        area_collect_tmp.append(np.sum(threshed_img_4th > thresh))

    com_x_tmp = np.array(com_x_arr)
    com_y_tmp = np.array(com_y_arr)
    peak_values_tmp = np.array(peak_values_tmp)
    
    idx_for_gt_40MHz = np.where(freqs_arr > 40e6)


    freq_for_fit = freqs_arr[idx_for_gt_40MHz]
    com_x_for_fit = com_x_tmp[idx_for_gt_40MHz]
    com_y_for_fit = com_y_tmp[idx_for_gt_40MHz]
    peak_values_for_fit = peak_values_tmp[idx_for_gt_40MHz]

    # idx for peak values > 2e6
    idx_for_gt_2e6 = np.where(peak_values_for_fit < 2e6)
    freq_for_fit_v1 = freq_for_fit[idx_for_gt_2e6]
    com_x_for_fit_v1 = com_x_for_fit[idx_for_gt_2e6]
    com_y_for_fit_v1 = com_y_for_fit[idx_for_gt_2e6]
    peak_values_for_fit_v1 = peak_values_for_fit[idx_for_gt_2e6]
    
    #linear fit
    if freq_for_fit_v1.size > 5:
        px = np.polyfit(1/freq_for_fit_v1**2, com_x_for_fit_v1, 1)
        py = np.polyfit(1/freq_for_fit_v1**2, com_y_for_fit_v1, 1)
    else:
        px = [np.nan, np.nan]
        py = [np.nan, np.nan]

    com_x_fitted = px[0] * 1/freqs_arr**2 + px[1]
    com_y_fitted = py[0] * 1/freqs_arr**2 + py[1]

    return [px, py, com_x_fitted, com_y_fitted]

def save_refraction_fit_param(fname_in, fname_out, px, py, com_x_fitted, com_y_fitted):
    """
    Save the refraction fit parameters to the fits

    :param fname_in: the input fits file name
    :param fname_out: the output fits file name
    :param px: the fit parameters for x
    :param py: the fit parameters for y
    :param com_x_fitted: the fitted com_x
    :param com_y_fitted: the fitted com_y
    """
    hdul = fits.open(fname_in)

    # correction distance per freq ch
    col_add1 = fits.Column(name='refra_shift_x', format='E', array=com_x_fitted)
    col_add2 = fits.Column(name='refra_shift_y', format='E', array=com_y_fitted)
    hdu = fits.BinTableHDU.from_columns(hdul[1].columns + col_add1 + col_add2)

    # also the parms for x = px[0] * 1/freq**2 + px[1]
    hdul[0].header["REFRA_PX0"] = str(px[0])
    hdul[0].header["REFRA_PX1"] = str(px[1])
    hdul[0].header["REFRA_PY0"] = str(py[0])
    hdul[0].header["REFRA_PY1"] = str(py[1])

    hdul[0].header["REFRA_COR"] = True
    hdul[0].header["REFRA_VER"] = '1.0'

    hdul[0].header["HISTORY"] = str(hdul[0].header["HISTORY"])+ " Refraction correction applied"

    hdul[1] = hdu
    hdul.writeto(fname_out, overwrite=True)
    return True