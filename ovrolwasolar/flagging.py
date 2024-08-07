from casatasks import clearcal, ft, bandpass, applycal, flagdata, tclean, flagmanager, uvsub, gaincal, split, imstat, \
    gencal
from casatools import table, measures, componentlist, msmetadata
import math
import sys, os, time
import numpy as np
import matplotlib.pyplot as plt
from . import utils
import logging, glob
import h5py
#from sklearn.cluster import KMeans


tb = table()
me = measures()
cl = componentlist()
msmd = msmetadata()


def get_antids(msfile):
    """
    Read antenna ids from a measurement set and separate them to inner and expansion ones

    :param msfile: path to CASA measurement set
    :return: antenna ids for core antennas and expansion antennas
    """
    tb.open(msfile + '/ANTENNA')
    ms_ant_names = tb.getcol('NAME')
    tb.close()
    msmd = msmetadata()
    msmd.open(msfile)
    core_ant_name_list = ['LWA{0:03d}'.format(i + 1) for i in range(0, 256)]
    exp_ant_name_list = ['LWA{0:03d}'.format(i + 1) for i in range(256, 366)]
    core_ant_ids = []
    exp_ant_ids = []
    for ms_ant_name in ms_ant_names:
        if ms_ant_name in core_ant_name_list:
            core_ant_ids.append(msmd.antennaids(ms_ant_name)[0])
        if ms_ant_name in exp_ant_name_list:
            exp_ant_ids.append(msmd.antennaids(ms_ant_name)[0])

    msmd.close()
    return np.array(core_ant_ids), np.array(exp_ant_ids)


def gen_ant_flags_from_autocorr(msfile, antflagfile=None, datacolumn='DATA', tavg=False,
                                thresh_core=1.0, thresh_exp=1.0, flag_exp_with_core_stat=True,
                                flag_either_pol=True, doappend=False, debug=False, doplot=False):
    """
    Generates a text file containing the bad antennas.
    DOES NOT ACTUALLY APPLY FLAGS. CURRENTLY SHOULD ONLY BE RUN ON SINGLE SPW MSs.

    Adapted from the flag_ants_from_postcal_autocorr() module in
    https://github.com/ovro-lwa/distributed-pipeline/blob/main/orca/flagging/flag_bad_ants.py

    Args:
        msfile (str): Path to the measurement set file.
        datacolumn (str, optional): Specify which data column to use. Defaults to "DATA".
                                   Could be "CORRECTED_DATA" if the dataset is calibrated.
        tavg (bool, optional): If set to True, will time average before evaluating flags. Defaults to False.
        antflagfile (str, optional): Output file that contains the flagged antennas. If not defined,
                                     defaults to msfile.replace('.ms', 'antflags').
        thresh_core (float, optional): Threshold to use for flagging for core antennas. Defaults to 1.
        thresh_exp (float, optional): Threshold to use for flagging for expansion antennas. Defaults to 1.
        flag_exp_with_core_stat (bool, optional): If True, use statistics of core antennas to determine flags for outer antennas. Defaults to False.
        flag_either_pol (bool, optional): If True, the antenna will be flagged if either polarization is bad (OR scheme).
                                          If False, the antenna will be flagged only if both polarizations are bad (AND scheme).

    Returns:
        str: Path to the text file with the list of antennas to flag (antflagfile).
    """
    tb.open(msfile)
    tautos = tb.query('ANTENNA1=ANTENNA2')
    tb.close()
    msmd = msmetadata()
    msmd.open(msfile)
    # get data, either DATA or CORRECTED_DATA
    autos = tautos.getcol(datacolumn)
    autos_flags = tautos.getcol('FLAG')
    autos_antnums = tautos.getcol('ANTENNA1')

    shape = autos.shape
    # autos_corrected.shape = (Nants*Nints, Nchans, Ncorrs)
    if shape[2] > 4:
        autos = np.swapaxes(autos, 0, 2)
        autos_flags = np.swapaxes(autos_flags, 0, 2)
        print("Shape updated")
    print(autos.shape)
    Nants = np.unique(autos_antnums).shape[0]
    Nints = int(autos_antnums.shape[0] / Nants)
    Ncorrs = autos.shape[-1]
    # average over frequency, reorder
    autos_corrected_mask = np.ma.masked_array(autos, mask=autos_flags,
                                              fill_value=np.nan)
    # take average in channel
    autos_tseries = np.ma.mean(autos_corrected_mask, axis=1).reshape(Nints, Nants, Ncorrs).transpose(1, 0, 2)
    antnums_reorder = autos_antnums.reshape(Nints, Nants).transpose(1, 0)
    # autos_tseries.shape = (Nants, Nints, Ncorrs)
    # if msfile has Nints>1, use time series; else just take median
    if autos_tseries.shape[1] == 1:
        arr_to_evaluate = autos_tseries[:, 0, :]
    elif tavg:
        arr_to_evaluate = np.ma.mean(autos_tseries, axis=1)
    else:
        medant_tseries = np.ma.median(autos_tseries, axis=0)
        arr_to_evaluate = np.ma.std(autos_tseries / medant_tseries, axis=1)

    autos_ampdb = 10. * np.log10(np.abs(arr_to_evaluate / 1.e2))
    print('shape of arr_to_evaluate', arr_to_evaluate.shape)
    # separate out core and expansion antennas
    # inds_core = list(range(0, 56)) + list(range(64, 120)) + list(range(128, 184)) + list(range(192, 238))
    # inds_exp = list(range(56, 64)) + list(range(120, 128)) + list(range(184, 192)) + list(range(238, 246))
    inds_core, inds_exp = get_antids(msfile)
    medval_core = np.ma.median(autos_ampdb[inds_core, :], axis=0)
    medval_exp = np.ma.median(autos_ampdb[inds_exp, :], axis=0)
    stdval_core = np.ma.std(autos_ampdb[inds_core, :], axis=0)
    stdval_exp = np.ma.std(autos_ampdb[inds_exp, :], axis=0)
    if flag_exp_with_core_stat:
        print('!! Use core antenna statistics to flag outer antennas !!')
        logging.debug('!! Use core antenna statistics to flag outer antennas !!')
        medval_exp = medval_core
        stdval_exp = stdval_core
    if debug:
        print('=====Before filtering out those beyond 1 sigma=====')
        print('Median of core antennas', medval_core[0], medval_core[3])
        print('Standard deviation of core antennas', stdval_core[0], stdval_core[3])
        print('Median of outer antennas', medval_exp[0], medval_exp[3])
        print('Standard deviation of outer antennas', stdval_exp[0], stdval_exp[3])
    # find 1 sigma outliers, exclude, and recalculate stdval
    newinds_core = np.asarray(inds_core)[
        np.where(((autos_ampdb[inds_core, 0] < medval_core[0] + 1 * stdval_core[0]) &
                  (autos_ampdb[inds_core, 0] > medval_core[0] - 1 * stdval_core[0])) |
                 ((autos_ampdb[inds_core, 3] < medval_core[3] + 1 * stdval_core[3]) &
                  (autos_ampdb[inds_core, 3] > medval_core[3] - 1 * stdval_core[3])))]
    newinds_exp = np.asarray(inds_exp)[
        np.where(((autos_ampdb[inds_exp, 0] < medval_exp[0] + 2 * stdval_exp[0]) &
                  (autos_ampdb[inds_exp, 0] > medval_exp[0] - 2 * stdval_exp[0])) |
                 ((autos_ampdb[inds_exp, 3] < medval_exp[3] + 2 * stdval_exp[3]) &
                  (autos_ampdb[inds_exp, 3] > medval_exp[3] - 2 * stdval_exp[3])))]
    # exclude and recalculate
    medval_core = np.ma.median(autos_ampdb[newinds_core, :], axis=0)
    medval_exp = np.ma.median(autos_ampdb[newinds_exp, :], axis=0)
    stdval_core = np.ma.std(autos_ampdb[newinds_core, :], axis=0)
    stdval_exp = np.ma.std(autos_ampdb[newinds_exp, :], axis=0)
    if debug:
        print('=====After filtering out those beyond 1 sigma=====')
        print('Median of core antennas', medval_core[0], medval_core[3])
        print('Standard deviation of core antennas', stdval_core[0], stdval_core[3])
        print('Median of outer antennas', medval_exp[0], medval_exp[3])
        print('Standard deviation of outer antennas', stdval_exp[0], stdval_exp[3])

    if flag_either_pol:
        flagscore = np.asarray(inds_core)[
            np.where((autos_ampdb[inds_core, 0] > medval_core[0] + thresh_core * stdval_core[0]) |
                     (autos_ampdb[inds_core, 0] < medval_core[0] - thresh_core * stdval_core[0]) |
                     (autos_ampdb[inds_core, 3] > medval_core[3] + thresh_core * stdval_core[3]) |
                     (autos_ampdb[inds_core, 3] < medval_core[3] - thresh_core * stdval_core[3]))]
        flagsexp = np.asarray(inds_exp)[
            np.where((autos_ampdb[inds_exp, 0] > medval_exp[0] + thresh_exp * stdval_exp[0]) |
                     (autos_ampdb[inds_exp, 0] < medval_exp[0] - thresh_exp * stdval_exp[0]) |
                     (autos_ampdb[inds_exp, 3] > medval_exp[3] + thresh_exp * stdval_exp[3]) |
                     (autos_ampdb[inds_exp, 3] < medval_exp[3] - thresh_exp * stdval_exp[3]))]
    else:
        flagscore = np.asarray(inds_core)[
            np.where(((autos_ampdb[inds_core, 0] > medval_core[0] + thresh_core * stdval_core[0]) |
                      (autos_ampdb[inds_core, 0] < medval_core[0] - thresh_core * stdval_core[0])) &
                     ((autos_ampdb[inds_core, 3] > medval_core[3] + thresh_core * stdval_core[3]) |
                      (autos_ampdb[inds_core, 3] < medval_core[3] - thresh_core * stdval_core[3])))]
        flagsexp = np.asarray(inds_exp)[
            np.where(((autos_ampdb[inds_exp, 0] > medval_exp[0] + thresh_exp * stdval_exp[0]) |
                      (autos_ampdb[inds_exp, 0] < medval_exp[0] - thresh_exp * stdval_exp[0])) &
                     ((autos_ampdb[inds_exp, 3] > medval_exp[3] + thresh_exp * stdval_exp[3]) |
                      (autos_ampdb[inds_exp, 3] < medval_exp[3] - thresh_exp * stdval_exp[3])))]
    flagsall = np.sort(np.append(flagscore, flagsexp))
    print('{0:d} bad antennas found out of {1:d} antennas'.format(flagsall.size, Nants))
    if flagsall.size > 0:
        if antflagfile is None:
            antflagfile = os.path.splitext(os.path.abspath(msfile))[0] + '.badants'
        print('Writing flags to ' + antflagfile)
        if os.path.exists(antflagfile) and doappend:
            existingflags = np.genfromtxt(antflagfile, delimiter=',', dtype=int)
            flagsall = np.append(flagsall, existingflags)
            flagsall = np.unique(flagsall)
        flagsallstr = [str(flag) for flag in flagsall]
        flag_core_ids = ",".join([str(flag) for flag in np.sort(flagscore)])
        if len(flagscore) > 0:
            flag_core_names = msmd.antennanames(flagscore)
        else:
            flag_core_names = []
        flag_core_vals = autos_ampdb[flagscore]
        flag_exp_ids = ",".join([str(flag) for flag in np.sort(flagsexp)])
        if len(flagsexp) > 0:
            flag_exp_names = msmd.antennanames(flagsexp)
        else:
            flag_exp_names = []
        flag_exp_vals = autos_ampdb[flagsexp]
        flagsallstr2 = ",".join(flagsallstr)
        print('flagged core antenna ids: ', flag_core_ids)
        print('flagged core antenna names: ', flag_core_names)
        print('flagged outer antenna ids: ', flag_exp_ids)
        print('flagged outer antenna names: ', flag_exp_names)
        msmd.close()
        with open(antflagfile, 'w') as f:
            f.write(flagsallstr2)
        if doplot:
            fig = plt.figure(figsize=(12, 5))
            for i, n in enumerate([0, 3]):
                ax = fig.add_subplot(1, 2, i + 1)
                if n == 0:
                    ax.set_title('Auto-correlation in XX')
                    upper_bound = thresh_core + np.max(stdval_core)
                if n == 3:
                    ax.set_title('Auto-correlation in YY')
                    thresh = thresh_exp
                ax.plot(inds_core, autos_ampdb[inds_core, n], 'ro', fillstyle='none', label='Inner')
                ax.plot(flagscore, autos_ampdb[flagscore, n], 'ro', fillstyle='full', label='Flagged Inner')
                ax.plot(inds_exp, autos_ampdb[inds_exp, n], 'bo', fillstyle='none', label='Outer')
                ax.plot(flagsexp, autos_ampdb[flagsexp, n], 'bo', fillstyle='full', label='Flagged Outer')
                ax.plot([0, Nants], [medval_core[n], medval_core[n]], 'r-')
                ax.plot([0, Nants], [medval_core[n] + thresh_core + stdval_core[n],
                                     medval_core[n] + thresh_core + stdval_core[n]], 'r--')
                ax.plot([0, Nants], [medval_core[n] - thresh_core * stdval_core[n],
                                     medval_core[n] - thresh_core * stdval_core[n]], 'r--')
                ax.plot([0, Nants], [medval_exp[n], medval_exp[n]], 'b-')
                ax.plot([0, Nants], [medval_exp[n] + thresh_exp * stdval_exp[n],
                                     medval_exp[n] + thresh_exp * stdval_exp[n]], 'b--')
                ax.plot([0, Nants], [medval_exp[n] - thresh_exp * stdval_exp[n],
                                     medval_exp[n] - thresh_exp * stdval_exp[n]], 'b--')
                ax.set_xlabel('Antenna ID')
                ax.set_ylabel('dB (avg over channels)')
                ax.set_ylim([-30, 10])

                ax.legend()
            fig.tight_layout()
            plt.show()

        if debug:
            return antflagfile, medval_core, stdval_core, flag_core_ids, flag_core_names, flag_core_vals, \
                   medval_exp, stdval_exp, flag_exp_ids, flag_exp_names, flag_exp_vals
        else:
            return antflagfile
    else:
        if debug:
            return 0, medval_core, stdval_core, medval_exp, stdval_exp
        else:
            return 0


def gen_ant_flags_tst(msfile: str, debug: bool = False) -> str:
    """Generates a text file containing the bad antennas.
    DOES NOT ACTUALLY APPLY FLAGS.

    Adapted from the flag_bad_ants() module in
    https://github.com/ovro-lwa/distributed-pipeline/blob/main/orca/flagging/flag_bad_ants.py

    Comment BC (April 7, 2023): Does not seem to work well with lots of antennas out
    Args:
        :param msfile: msfile to generate
    Returns:
        Path to the text file with list of antennas to flag.
    """
    nant, nspw, nchan = utils.get_msinfo(msfile)
    tb.open(msfile)
    tautos = tb.query('ANTENNA1=ANTENNA2')

    # iterate over antenna, 1-->256
    datacolxx = np.zeros((nchan * nspw, nant))
    datacolyy = np.copy(datacolxx)
    for i in range(nspw):
        datacolxx[i * nchan:(i + 1) * nchan] = tb.getcol("DATA", nant * i, nant)[0]
        datacolyy[i * nchan:(i + 1) * nchan] = tb.getcol("DATA", nant * i, nant)[3]

    datacolxxamp = np.sqrt(np.real(datacolxx) ** 2. + np.imag(datacolxx) ** 2.)
    datacolyyamp = np.sqrt(np.real(datacolyy) ** 2. + np.imag(datacolyy) ** 2.)

    datacolxxampdb = 10 * np.log10(datacolxxamp / 1.e2)
    datacolyyampdb = 10 * np.log10(datacolyyamp / 1.e2)

    # median value for every antenna
    medamp_perantx = np.median(datacolxxampdb, axis=1)
    medamp_peranty = np.median(datacolyyampdb, axis=1)

    # get flags based on deviation from median amp
    xthresh_pos = np.median(medamp_perantx) + np.std(medamp_perantx)
    xthresh_neg = np.median(medamp_perantx) - 2 * np.std(medamp_perantx)
    ythresh_pos = np.median(medamp_peranty) + np.std(medamp_peranty)
    ythresh_neg = np.median(medamp_peranty) - 2 * np.std(medamp_peranty)
    flags = np.where((medamp_perantx > xthresh_pos) | (medamp_perantx < xthresh_neg) | \
                     (medamp_peranty > ythresh_pos) | (medamp_peranty < ythresh_neg) | \
                     np.isnan(medamp_perantx) | np.isnan(medamp_peranty))

    # use unflagged antennas to generate median spectrum
    flagmask = np.zeros((nchan * nspw, nant))
    flagmask[:, flags[0]] = 1
    datacolxxampdb_mask = np.ma.masked_array(datacolxxampdb, mask=flagmask, fill_value=np.nan)
    datacolyyampdb_mask = np.ma.masked_array(datacolyyampdb, mask=flagmask, fill_value=np.nan)

    medamp_allantsx = np.median(datacolxxampdb_mask, axis=1)
    medamp_allantsy = np.median(datacolyyampdb_mask, axis=1)

    stdarrayx = np.array([np.std(antarr / medamp_allantsx) for antarr in datacolxxampdb_mask.transpose()])
    stdarrayy = np.array([np.std(antarr / medamp_allantsy) for antarr in datacolyyampdb_mask.transpose()])

    # this threshold was manually selected...should be changed to something better at some point
    if nant > 256:
        thresh = 1
    else:
        thresh = 0.02
    flags2 = np.where((stdarrayx > thresh) | (stdarrayy > thresh))

    flagsall = np.sort(np.append(flags, flags2))
    flagsallstr = [str(flag) for flag in flagsall]
    flagsallstr2 = ",".join(flagsallstr)

    antflagfile = os.path.dirname(os.path.abspath(msfile)) + '/flag_bad_ants.ants'
    with open(antflagfile, 'w') as f:
        f.write(flagsallstr2)

    tb.close()
    if debug:
        return medamp_perantx, medamp_peranty, stdarrayx, stdarrayy
    else:
        return antflagfile


def flag_bad_ants(msfile, antflagfile=None, datacolumn='DATA', thresh_core=1.0, thresh_exp=1.0, clearflags=True):
    """
    Read the text file that contains flags for bad antennas, and apply the flags
    
    :param msfile: input CASA ms visibility for calibration
    :param thresh: Threshold to use for flagging. Default is 10.
    """
    if clearflags:
        flaglist = flagmanager(msfile, mode='list')
        # check if previous flags exist. If so, restore to original state
        if len(flaglist) > 1:
            logging.debug('Clearing all previous flags')
            flagmanager(msfile, mode='restore', versionname=flaglist[0]['name'])
    if antflagfile is None:
        logging.debug('Antenna flag file not supplied.')
        antflagfile = os.path.splitext(os.path.abspath(msfile))[0] + '.badants'
        logging.debug('Generating antenna flags from auto-correlation')
        res = gen_ant_flags_from_autocorr(msfile, antflagfile=antflagfile, datacolumn=datacolumn,
                                          thresh_core=thresh_core, thresh_exp=thresh_exp)
    if os.path.isfile(antflagfile):
        with open(antflagfile, 'r') as f:
            antenna_list = f.readline()
            print('Applying flags for these antennas')
            print(antenna_list)
            logging.debug('Flagging antennas ' + antenna_list)
        flagdata(vis=msfile, mode='manual', antenna=antenna_list)
    else:
        logging.debug("No flag is found. Do nothing")
        print("No flag is found. Do nothing")
    return antflagfile


def func_baseline_flagging(msfile,verbose=False,n_clusters = 128,extend_flg=True,
                           combine_chans=True):
    """
    Perform baseline flagging using KMeans clustering and outlier removal

    :param msfile: path to CASA measurement set
    :param verbose: if True, print the percentage of flagged data
    :param n_clusters: number of clusters to use for KMeans clustering
    """
    from sklearn.cluster import KMeans
    tb = table()
    msmd = msmetadata()
    tb.open(msfile)
    msmd.open(msfile)

    visibility_data = tb.getcol('DATA')

    # Extract UVW Column
    uvw_data = tb.getcol('UVW')

    u_col = uvw_data[0, :]
    v_col = uvw_data[1, :]
    w_col = uvw_data[2, :]

    # Extract FLAG Column
    flag_data = tb.getcol('FLAG')
    new_flag_data = flag_data.copy()

    # Close the table
    tb.close()

    # flag the autocorr
    flag_data[:,:,np.sqrt(u_col**2+v_col**2)<1] = True
    if verbose:
        print("Previously flagged {}% of the data".format(np.sum(flag_data)/flag_data.size*100))   


    uv_data = np.column_stack((u_col, v_col))
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(uv_data)
    labels = kmeans.labels_


    n_ch = visibility_data.shape[1]
    n_times = visibility_data.shape[2]

    labels_allch = np.tile(labels, n_ch)
    u_col_allch = np.tile(u_col, n_ch)
    v_col_allch = np.tile(v_col, n_ch)

    if combine_chans:
        flg_combincol = flag_data[0,:,:].reshape(n_ch*n_times)

        stokes_I = 0.5 * (visibility_data[0,:,:] + visibility_data[3,:,:])
        Stokes_I_amp = np.abs(stokes_I)
        stokes_I_amp_combincol = Stokes_I_amp.reshape(n_ch*n_times)

        labels_allch_combincol = labels_allch.reshape(n_ch*n_times)
        u_col_combincol = u_col_allch.reshape(n_ch*n_times)
        v_col_combincol = v_col_allch.reshape(n_ch*n_times)

        u_collect_after_rm_outliers = []
        v_collect_after_rm_outliers = []
        stokes_I_collect_after_rm_outliers = []

        u_outliers = []
        v_outliers = []
        stokes_I_outliers = []

        outlier_idx = []

        for i in range(n_clusters):
            this_label_idx = np.where(labels_allch_combincol == i)
            goodvis_this_label_idx = np.where(~flg_combincol[this_label_idx])[0]
            u_col_this_label = u_col_combincol[this_label_idx][goodvis_this_label_idx]
            v_col_this_label = v_col_combincol[this_label_idx][goodvis_this_label_idx]

            if goodvis_this_label_idx.shape[0] > 100:
                stokes_I_this_label = stokes_I_amp_combincol[this_label_idx][goodvis_this_label_idx]
                Q1 = np.percentile(stokes_I_this_label, 25)
                Q3 = np.percentile(stokes_I_this_label, 75)
                IQR = Q3 - Q1
                outlier_step = 1.2 * IQR
                outlier_list_col = ((stokes_I_this_label < Q1 - outlier_step) | (stokes_I_this_label > Q3 + outlier_step))
                u_collect_after_rm_outliers.append(u_col_this_label[~outlier_list_col])
                v_collect_after_rm_outliers.append(v_col_this_label[~outlier_list_col])
                stokes_I_collect_after_rm_outliers.append(stokes_I_this_label[~outlier_list_col])
                u_outliers.append(u_col_this_label[outlier_list_col])
                v_outliers.append(v_col_this_label[outlier_list_col])
                stokes_I_outliers.append(stokes_I_this_label[outlier_list_col])
                outlier_idx.append(this_label_idx[0][goodvis_this_label_idx[outlier_list_col]])

            
        u_collect_after_rm_outliers = np.concatenate(u_collect_after_rm_outliers)
        v_collect_after_rm_outliers = np.concatenate(v_collect_after_rm_outliers)
        stokes_I_collect_after_rm_outliers = np.concatenate(stokes_I_collect_after_rm_outliers)

        u_outliers = np.concatenate(u_outliers)
        v_outliers = np.concatenate(v_outliers)
        stokes_I_outliers = np.concatenate(stokes_I_outliers)
        outlier_idx = np.concatenate(outlier_idx)

        flg_combincol[outlier_idx] = True
        flg_chs = flg_combincol.reshape(n_ch,n_times)
        new_flag_data[0,:,:] = (flg_chs +new_flag_data[0,:,:])>0.5

    else:
        for ch_idx in range(n_ch):
            flg_1col = flag_data[0,ch_idx,:]

            stokes_I = 0.5 * (visibility_data[0,ch_idx,:] + visibility_data[3,ch_idx,:])
            Stokes_I_amp = np.abs(stokes_I)


            # repeat the process to remove the outliers for each cluster

            u_collect_after_rm_outliers = []
            v_collect_after_rm_outliers = []
            stokes_I_collect_after_rm_outliers = []

            u_outliers = []
            v_outliers = []
            stokes_I_outliers = []

            outlier_idx = []

            for i in range(n_clusters):
                this_label_idx = np.where(labels == i)
                goodvis_this_label_idx = np.where(~flg_1col[this_label_idx])[0]
                u_col_this_label = u_col[this_label_idx][goodvis_this_label_idx]
                v_col_this_label = v_col[this_label_idx][goodvis_this_label_idx]

                if goodvis_this_label_idx.shape[0] > 100:
                    stokes_I_this_label = Stokes_I_amp[this_label_idx][goodvis_this_label_idx]
                    Q1 = np.percentile(stokes_I_this_label, 25)
                    Q3 = np.percentile(stokes_I_this_label, 75)
                    IQR = Q3 - Q1
                    outlier_step = 1.2 * IQR
                    outlier_list_col = ((stokes_I_this_label < Q1 - outlier_step) | (stokes_I_this_label > Q3 + outlier_step))
                    u_collect_after_rm_outliers.append(u_col_this_label[~outlier_list_col])
                    v_collect_after_rm_outliers.append(v_col_this_label[~outlier_list_col])
                    stokes_I_collect_after_rm_outliers.append(stokes_I_this_label[~outlier_list_col])
                    u_outliers.append(u_col_this_label[outlier_list_col])
                    v_outliers.append(v_col_this_label[outlier_list_col])
                    stokes_I_outliers.append(stokes_I_this_label[outlier_list_col])
                    outlier_idx.append(this_label_idx[0][goodvis_this_label_idx[outlier_list_col]])

                
            u_collect_after_rm_outliers = np.concatenate(u_collect_after_rm_outliers)
            v_collect_after_rm_outliers = np.concatenate(v_collect_after_rm_outliers)
            stokes_I_collect_after_rm_outliers = np.concatenate(stokes_I_collect_after_rm_outliers)

            u_outliers = np.concatenate(u_outliers)
            v_outliers = np.concatenate(v_outliers)
            stokes_I_outliers = np.concatenate(stokes_I_outliers)
            outlier_idx = np.concatenate(outlier_idx)

            new_flag_data[0,ch_idx,outlier_idx] = True

    for idx in new_flag_data.shape[0]:
        new_flag_data[idx,:,:] = new_flag_data[0,:,:]

    if extend_flg:
        e_new_flag = new_flag_data.copy()
        idx_extend  = np.mean(new_flag_data, axis=1)[0]>0.50
        e_new_flag[:,:,idx_extend] = True

    for idx in new_flag_data.shape[0]:
        new_flag_data[idx,ch_idx,outlier_idx] = True

    if verbose:
        print("Flagged {}% of the data".format(np.sum(new_flag_data)/new_flag_data.size*100))

    return new_flag_data


def perform_baseline_flagging(msfile, overwrite=True, new_msfile=None, verbose=True, 
                              n_clusters=128, combine_chans=True, extend_flg=False):
    new_flag_data=func_baseline_flagging(msfile,verbose,n_clusters,combine_chans=True) 
    if new_msfile is None:
        ms_new = msfile.replace('.ms', '_bflagged.ms')
    else:
        ms_new = new_msfile
    if overwrite:
        ms_new = msfile
    tb.open(ms_new, nomodify=False)
    tb.putcol('FLAG', new_flag_data)
    tb.close()


def make_cross_coor_flagging(ms_file, outh5):
    """
    create cross-correlation matrix for flagging
    recorded as

    in uv index:
    when idx_ant1 < idx_ant2, the entry is flag ratio
    when idx_ant1 >= idx_ant2, the entry is amplitude of correlation
    
    return a n_ant*n_ant matrix
    """

    tb.open(ms_file)

    # Extract DATA Column
    visibility_data = tb.getcol('DATA')

    # extract ant1 and2 columns
    ant1 = tb.getcol('ANTENNA1')
    ant2 = tb.getcol('ANTENNA2')
    stokes_I = 0.5 * (visibility_data[0,:,:] + visibility_data[3,:,:])

    # Extract FLAG Column
    flag_data = tb.getcol('FLAG')

    # Close the table
    tb.close()
    
    unique_ant1 = np.unique(ant1)
    num_ant = unique_ant1.shape[-1]    

    img_cross = np.zeros((num_ant, num_ant))
    for idx in range(stokes_I.shape[1]):
        idx1 = np.max([ant1[idx], ant2[idx]])
        idx2 = np.min([ant1[idx], ant2[idx]])
        img_cross[idx2,idx1]= np.mean(np.abs(flag_data[0,:,idx].squeeze()), axis=0)
        img_cross[idx1,idx2]= np.mean(np.abs(stokes_I[:,idx]), axis=0)
    
    # save to hdf5 file
    with h5py.File(outh5, 'w') as f:
        f.create_dataset('cormat', data=img_cross)
        f.attrs['timeobs'] = msmd.sourcetimes()['0']['value']
