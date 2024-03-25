from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.time import Time
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
from shutil import copyfile

from scipy.ndimage import binary_erosion, binary_dilation, binary_closing
import sunpy.map as smap


def thresh_func(freq):  # freq in Hz
    """Return the threshold for the given frequency
    
    :param freq: frequency in Hz

    :return: threshold in Tb
    """

    return 1.1e6 * (1 - 1.8e4 * freq ** (-0.6))


def find_center_of_thresh(data_this, thresh, meta):
    """
    Find the center of the thresholded image
    
    :param data_this: the data to be thresholded
    :param thresh: the threshold
    :param meta: the meta data of the fits file
    
    """
    threshed_img = (data_this > thresh)
    threshed_img_1st = remove_small_objects(threshed_img, min_size=1000/(meta['CDELT1']/60.)**2., connectivity=1)
    # perform erosion to remove the small features
    threshed_img_2nd = binary_erosion(threshed_img_1st, iterations=3)

    # keep only the largest connected component
    threshed_img_3rd = remove_small_objects(threshed_img_2nd, min_size=1000/(meta['CDELT1']/60.)**2., connectivity=1)

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


def refraction_fit_param(fname, thresh_freq=45e6, overbright=2.0e6, min_freqfrac=0.3):
    """
    Take in a multi-frequency fits file and return the refraction fit parameters for:
    `
    x = px[0] * 1/freq**2 + px[1]
    y = py[0] * 1/freq**2 + py[1]
    `
    
    :param fname: the fits file name
    :param thresh_freq: the threshold frequency for the fit
    :param overbright: peak brightness temperature exceeding this value (in Kelvin) will be excluded for fitting 
    :param min_freqfrac: minimum fraction of usable frequency channels 
        (above the frequency threshhold) to do the fit. Absolute minimum is 5.
    """

    meta, data = ndfits.read(fname)
    #    hdu = fits.open(fname)
    #    data = hdu[0].data
    #    meta = hdu[0].header

    freqs_arr = meta['ref_cfreqs']  # Hz
    header_meta = meta["header"]

    com_x_arr = []
    com_y_arr = []
    peak_values_tmp = []
    area_collect_tmp = []
    for idx_this, idx_img in enumerate(range(0, freqs_arr.shape[0])):
        thresh = thresh_func(freqs_arr[idx_img]) / 5
        data_this = np.squeeze(data[0, idx_img, :, :])
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

    idx_for_gt_freqthresh = np.where(freqs_arr > thresh_freq)

    freq_for_fit = freqs_arr[idx_for_gt_freqthresh]
    com_x_for_fit = com_x_tmp[idx_for_gt_freqthresh]
    com_y_for_fit = com_y_tmp[idx_for_gt_freqthresh]
    peak_values_for_fit = peak_values_tmp[idx_for_gt_freqthresh]

    # Only frequencies with peak values < overbright are considered for fitting
    idx_not_too_bright = np.where(peak_values_for_fit < overbright)
    freq_for_fit_v1 = freq_for_fit[idx_not_too_bright]
    com_x_for_fit_v1 = com_x_for_fit[idx_not_too_bright]
    com_y_for_fit_v1 = com_y_for_fit[idx_not_too_bright]

    # peak_values_for_fit_v1 = peak_values_for_fit[idx_not_too_bright]

    # linear fit
    if freq_for_fit_v1.size > max(int(len(idx_for_gt_freqthresh[0]) * min_freqfrac), 5):
    #if freq_for_fit_v1.size > 5:
        px = np.polyfit(1 / freq_for_fit_v1 ** 2, com_x_for_fit_v1, 1)
        py = np.polyfit(1 / freq_for_fit_v1 ** 2, com_y_for_fit_v1, 1)
    else:
        px = [np.nan, np.nan]
        py = [np.nan, np.nan]

    com_x_fitted = px[0] * 1 / freqs_arr ** 2 + px[1]
    com_y_fitted = py[0] * 1 / freqs_arr ** 2 + py[1]

    return [px, py, com_x_fitted, com_y_fitted]


def save_refraction_fit_param(fname_in, fname_out, px, py, com_x_fitted, com_y_fitted):
    """
    Updates a FITS file with new refraction fit parameters and copies it to a new file.
    Do this in-place by using same file name for `fname_in` and `fname_out`.

    :param fname_in: Path to the input FITS file.
    :type fname_in: str
    :param fname_out: Path to the output (updated) FITS file.
    :type fname_out: str
    :param px: The fit parameters for the x-direction, expected to have at least 2 elements.
    :type px: list or np.ndarray
    :param py: The fit parameters for the y-direction, expected to have at least 2 elements.
    :type py: list or np.ndarray
    :param com_x_fitted: The fitted com_x coordinates to add as a new column to the FITS table.
    :type com_x_fitted: np.ndarray
    :param com_y_fitted: The fitted com_y coordinates to add as a new column to the FITS table.
    :type com_y_fitted: np.ndarray
    """
    # Copy the input file to the output file location
    copyfile(fname_in, fname_out)

    # correction distance per freq ch
    col_add1 = fits.Column(name='refra_shift_x', format='E', array=com_x_fitted)
    col_add2 = fits.Column(name='refra_shift_y', format='E', array=com_y_fitted)
    new_table_columns = [col_add1, col_add2]

    # also the parms for x = px[0] * 1/freq**2 + px[1]
    new_header_entries = {
        "RFRPX0": px[0],
        "RFRPX1": px[1],
        "RFRPY0": py[0],
        "RFRPY1": py[1],
        "RFRCOR": False,
        "RFRVER": "1.0",
        "LVLNUM": "1.0",
        "HISTORY": "Refraction corrections V1.0 calculated and saved to the header on {0:s}. No corrections applied to the data.".format(Time.now().isot[:19])
    }

    success = ndfits.update(fname_out, new_table_columns, new_header_entries)
    if success:
        print("FITS file successfully updated.")
    else:
        print("Failed to update FITS file.")
    return True


def save_resample_align(fname_in, fname_out, px, py, com_x_fitted, com_y_fitted):
    """
    Creat level 1.5 fits file
    Read in the fits file, apply the refraction correction, move the CRVAL1, CRVAL2 to the center of the image,
    resample the image to the same size, and save the result to a new fits file
    
    :param fname_in: Path to the input FITS file.
    :type fname_in: str
    :param fname_out: Path to the output (updated) FITS file.
    :type fname_out: str
    :param px: The fit parameters for the x-direction, expected to have at least 2 elements.
    :type px: list or np.ndarray
    :param py: The fit parameters for the y-direction, expected to have at least 2 elements.
    :type py: list or np.ndarray
    :param com_x_fitted: The fitted com_x coordinates to add as a new column to the FITS table.
    :type com_x_fitted: np.ndarray
    :param com_y_fitted: The fitted com_y coordinates to add as a new column to the FITS table.
    :type com_y_fitted: np.ndarray

    """
    copyfile(fname_in, fname_out)

    hdul = fits.open(fname_in)
    datasize = hdul[0].data.shape
    new_data = np.zeros(datasize)
    delta_x = hdul[0].header["CDELT1"]
    delta_y = hdul[0].header["CDELT2"]
    nx = hdul[0].header["NAXIS1"]
    ny = hdul[0].header["NAXIS2"]

    # modify the data array move the center of the image to the fitted center
    for pol in range(datasize[0]):
        for chn in range(datasize[1]):
            datatmp = hdul[0].data[pol, chn, :, :]

            shift_x_tmp, shift_y_tmp = com_x_fitted[chn], com_y_fitted[chn]

            datatmp = np.roll(datatmp, -int(np.round(shift_y_tmp / delta_y)), axis=0)
            datatmp = np.roll(datatmp, -int(np.round(shift_x_tmp / delta_x)), axis=1)
            new_data[pol, chn, :, :] = datatmp
    hdul.close()

    new_header_entry = {
        "CRVAL1": 0,
        "CRVAL2": 0,
        "CRPIX1": nx // 2,
        "CRPIX2": ny // 2,
        "RFRPX0": px[0],
        "RFRPX1": px[1],
        "RFRPY0": py[0],
        "RFRPY1": py[1],
        "RFRCOR": True,
        "RFRVER": "1.0",
        "LVLNUM": "1.5",
        "HISTORY": "Refraction corrections V1.0 applied to data array on {0:s}".format(Time.now().isot[:19])
    }

    success = ndfits.update(fname_out, new_data=new_data, new_header_entries=new_header_entry)
    if success:
        print("FITS file successfully updated.")
    else:
        print("Failed to update FITS file.")

    return True
