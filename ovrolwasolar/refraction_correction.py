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

from skimage.morphology import convex_hull_image
from scipy.ndimage import binary_erosion, binary_dilation, binary_closing
import sunpy.map as smap


def thresh_func(freq):  # freq in Hz
    """Return the threshold for the given frequency
    
    :param freq: frequency in Hz

    :return: threshold in Tb
    """

    return 1.1e6 * (1 - 1.8e4 * freq ** (-0.6))


def find_center_of_thresh(data_this, thresh, meta, 
                          index, min_size_50=1000,convex_hull=False):
    """
    Find the center of the thresholded image
    
    :param data_this: the data to be thresholded
    :param thresh: the threshold
    :param meta: the meta data of the fits file
    :param index: the index of the image contained in the fits file
    :param min_size_50: The smallest allowable object area, in pixels, at 50 MHz. min_size will scale with 1/(nu[MHz]/50MHz)**2.
    
    """
    meta_header=meta['header']
    threshed_img = (data_this > thresh)
    min_size = min_size_50/(meta_header['CDELT1']/60.)**2./(meta['ref_cfreqs'][index]/50e6)**2.
    threshed_img_1st = remove_small_objects(threshed_img, min_size=min_size, connectivity=1)
    # perform erosion to remove the small features
    threshed_img_2nd = binary_erosion(threshed_img_1st, iterations=3)

    # keep only the largest connected component
    threshed_img_3rd = remove_small_objects(threshed_img_2nd, min_size=min_size, connectivity=1)

    # dialate the image back to the original size
    threshed_img_4th = binary_dilation(threshed_img_3rd, iterations=3)

    if convex_hull:
        threshed_img_4th = convex_hull_image(threshed_img_4th)

    # find the centroid of threshed_img_1st, coords in x_arr, y_arr
    com = center_of_mass(threshed_img_4th)
    # convert to arcsec

    x_arr = meta_header['CRVAL1'] + meta_header['CDELT1'] * (np.arange(meta_header['NAXIS1']) - (meta_header['CRPIX1'] - 1))
    y_arr = meta_header['CRVAL2'] + meta_header['CDELT2'] * (np.arange(meta_header['NAXIS2']) - (meta_header['CRPIX2'] - 1))

    # convert com from pixel to arcsec (linear)
    com_x_arcsec = x_arr[0] + com[1] * (x_arr[-1] - x_arr[0]) / (len(x_arr) - 1)
    com_y_arcsec = y_arr[0] + com[0] * (y_arr[-1] - y_arr[0]) / (len(y_arr) - 1)

    # move x_arr, y_arr to the center of the image
    x_arr_new = x_arr - com_x_arcsec
    y_arr_new = y_arr - com_y_arcsec

    return [com_x_arcsec, com_y_arcsec, com, x_arr_new, y_arr_new, threshed_img,
            threshed_img_1st, threshed_img_2nd, threshed_img_3rd, threshed_img_4th]


def refraction_fit_param(fname, thresh_freq=45e6, overbright=2.0e6, min_freqfrac=0.3,
                          return_record=False, convex_hull=False, background_factor=1/8):
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
    :param return_record: if True, return a refraction coefficient record containing a time stamp, else just the coefficients
    :type py: boolean 
    """

    meta, data = ndfits.read(fname)
    freqs_arr = meta['ref_cfreqs']  # Hz

    com_x_arr = []
    com_y_arr = []
    peak_values_tmp = []
    area_collect_tmp = []
    for idx_this, idx_img in enumerate(range(0, freqs_arr.shape[0])):
        thresh = thresh_func(freqs_arr[idx_img]) * background_factor
        data_this = np.squeeze(data[0, idx_img, :, :])
        (com_x_arcsec, com_y_arcsec, com, x_arr_new, y_arr_new, threshed_img,
         threshed_img_1st, threshed_img_2nd, threshed_img_3rd, threshed_img_4th
         ) = find_center_of_thresh(data_this, thresh, meta, idx_img, convex_hull=convex_hull)
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

    # remove nan from com_x com_y
    idx_nan = np.where(np.isnan(com_x_for_fit_v1) | np.isnan(com_y_for_fit_v1))
    freq_for_fit_v2 = np.delete(freq_for_fit_v1, idx_nan)
    com_x_for_fit_v2 = np.delete(com_x_for_fit_v1, idx_nan)
    com_y_for_fit_v2 = np.delete(com_y_for_fit_v1, idx_nan)

    # linear fit
    if freq_for_fit_v2.size > max(int(len(idx_for_gt_freqthresh[0]) * min_freqfrac), 5):
    #if freq_for_fit_v1.size > 5:
        px = np.polyfit(1 / freq_for_fit_v2 ** 2, com_x_for_fit_v2, 1)
        py = np.polyfit(1 / freq_for_fit_v2 ** 2, com_y_for_fit_v2, 1)
    else:
        px = [np.nan, np.nan]
        py = [np.nan, np.nan]

    #com_x_fitted = px[0] * 1 / freqs_arr ** 2 + px[1]
    #com_y_fitted = py[0] * 1 / freqs_arr ** 2 + py[1]
    reftime = meta['header']['date-obs'][:19]

    if return_record:
        return {'Time':reftime, 'px0':px[0], 'px1':px[1], 'py0':py[0], 'py1':py[1]}
    else:
        return px, py


def save_refraction_fit_param(fname_in, fname_out, px, py):
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
    """
    # Copy the input file to the output file location
    copyfile(fname_in, fname_out)
    meta, data = ndfits.read(fname_in)
    freqs_arr = meta['ref_cfreqs']  # Hz
    com_x_fitted = px[0] * 1 / freqs_arr ** 2 + px[1]
    com_y_fitted = py[0] * 1 / freqs_arr ** 2 + py[1]

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


def apply_refra_coeff(fname_in, px, py, fname_out=None, verbose=False):
    """
    Apply refraction correction coefficients to level 1.0 fits file and create level 1.5 fits file
    Read in the fits file, apply the refraction correction coefficients by rolling the image pixels, update CRVALi and CRPIXi,
    and save the result to a new fits file
    
    :param fname_in: Path to the input FITS file.
    :type fname_in: str
    :param px: The fit parameters for the x-direction, expected to have at least 2 elements.
    :type px: list or np.ndarray
    :param py: The fit parameters for the y-direction, expected to have at least 2 elements.
    :type py: list or np.ndarray
    :param fname_out: Name of the output level 1.5 FITS file.
    :type fname_out: str

    :return fname_out: Name of the output level 1.5 FITS file.
    :rtype: str 

    """
    if fname_out is None:
        fname_out = './' + os.path.basename(fname_in).replace('lev1','lev1.5')
    copyfile(fname_in, fname_out)
    meta, data = ndfits.read(fname_in)
    freqs_arr = meta['ref_cfreqs']  # Hz
    com_x_fitted = px[0] * 1 / freqs_arr ** 2 + px[1]
    com_y_fitted = py[0] * 1 / freqs_arr ** 2 + py[1]

    datasize = data.shape
    new_data = np.zeros(datasize)
    old_crval1 = meta['header']["CRVAL1"]
    old_crval2 = meta['header']["CRVAL2"]
    delta_x = meta['header']["CDELT1"]
    delta_y = meta['header']["CDELT2"]
    nx = meta['header']["NAXIS1"]
    ny = meta['header']["NAXIS2"]

    # modify the data array move the center of the image to the fitted center
    for pol in range(datasize[0]):
        for chn in range(datasize[1]):
            datatmp = data[pol, chn, :, :]
            shift_x_tmp, shift_y_tmp = com_x_fitted[chn]-old_crval1, com_y_fitted[chn]-old_crval2
            datatmp = np.roll(datatmp, -int(np.round(shift_y_tmp / delta_y)), axis=0)
            datatmp = np.roll(datatmp, -int(np.round(shift_x_tmp / delta_x)), axis=1)
            new_data[pol, chn, :, :] = datatmp

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
        if verbose:
            print("FITS file successfully updated.")
        return fname_out
    else:
        print("Failed to update FITS file.")
        return False


def apply_refra_record(fname_in, refra_record, fname_out=None, interp='linear', max_dt=600.):
    """
    Use refraction correction record(s) to update level 1.0 fits file and create level 1.5 fits file
    Read in the level 1 fits file, apply the refraction correction record(s), do interpolation using nearby records, 
    and save the result to a new level 1.5 fits file
    
    :param fname_in: Path to the input FITS file.
    :type fname_in: str
    :param fname_out: Name of the output level 1.5 FITS file.
    :type fname_out: str
    :param refra_record: refraction correction coefficients record
    :type refra_record: a single dictionary or a list of dictionary
    :param interp: method of interpolation passed to scipy.interpolate.interp1d. Default is 'linear'
    :type interp: str
    :param max_dt: maximum time difference to perform the interpolation in seconds
    :type max_dt: float

    :return fname_out: Name of the output level 1.5 FITS file.
    :rtype: string 

    """
    from scipy import interpolate
    import pandas as pd
    if fname_out is None:
        fname_out = './' + os.path.basename(fname_in).replace('lev1','lev1.5')
    if isinstance(refra_record, dict):
        rec = refra_record
        if 'px0' in rec and 'px1' in rec and 'py0' in rec and 'py1' in rec:
            px = [rec['px0'], rec['px1']]
            py = [rec['py0'], rec['py1']]
            fname_out = apply_refra_coeff(fname_in, px, py, fname_out=fname_out)
            return fname_out
        else:
            print('The input refraction record does not have all the required keys. Abort.')
    elif isinstance(refra_record, list):
        rec_df = pd.DataFrame(refra_record)
    elif isinstance(refra_record, pd.DataFrame):
        rec_df = refra_record
    else:
        print('Input refra_record needs to be a dictionary, list or dictionaries, or a pandas.DataFrame. Abort.')
        return False

    # Now use the DataFrame record to do interpolation
    meta, data = ndfits.read(fname_in)
    time_in = Time(meta['header']['date-obs']).mjd
    times = Time(list(rec_df['Time'].values)).mjd
    dt_minute = np.min(np.abs(times-time_in)*24.*60.*60.)
    if dt_minute < max_dt:
        px0s = rec_df['px0'].values
        px1s = rec_df['px1'].values
        py0s = rec_df['py0'].values
        py1s = rec_df['py1'].values
        fx0 = interpolate.interp1d(times, px0s, kind=interp, fill_value="extrapolate")
        fx1 = interpolate.interp1d(times, px1s, kind=interp, fill_value="extrapolate")
        fy0 = interpolate.interp1d(times, py0s, kind=interp, fill_value="extrapolate")
        fy1 = interpolate.interp1d(times, py1s, kind=interp, fill_value="extrapolate")
        px0 = fx0(time_in).item()
        px1 = fx1(time_in).item()
        py0 = fy0(time_in).item()
        py1 = fy1(time_in).item()
        fname_out = apply_refra_coeff(fname_in, [px0, px1], [py0, py1], fname_out=fname_out)
        return fname_out
    else:
        print('Time difference between the input fits file and the record is greater than the set maximum {0:.1f} s. Abort.'.format(max_dt))
        return False



