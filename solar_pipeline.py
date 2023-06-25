"""
Pipeline for calibrating and imaging solar data.
It was initially adapted from Marin Anderson's script named /opt/astro/utils/bin/gen_model_ms.py
    on astm.lwa.ovro.caltech.edu in August 2022
Certain functions are adapted from the orca repository at https://github.com/ovro-lwa/distributed-pipeline

Requirements:
- A modular installation of CASA 6: https://casadocs.readthedocs.io/en/stable/notebooks/introduction.html#Modular-Packages
- A working version of wsclean for imaging (i.e., "wsclean" defined in the search path)
"""

from casatasks import clearcal, ft, bandpass, applycal, flagdata, tclean, flagmanager, uvsub, gaincal, split, imstat, \
    gencal
from casatools import table, measures, componentlist, msmetadata
import math
import sys, os, time
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.wcs import WCS
from astropy.io import fits
import matplotlib.pyplot as plt
import utils
import logging, glob
from file_handler import File_Handler
from primary_beam import woody_beam as beam 
import primary_beam
from generate_calibrator_model import model_generation
import generate_calibrator_model
tb = table()
me = measures()
cl = componentlist()
msmd = msmetadata()


def get_sun_pos(msfile, str_output=True):
    """
    Return J2000 RA and DEC coordinates of the solar disk center
    :param msfile: input CASA measurement set
    :param str_output: if True, return coordinate in string form acceptable by CASA tclean
        if False, return a dictionary in CASA measures format: https://casa.nrao.edu/docs/casaref/measures.measure.html
    :return: solar disk center coordinate in string or dictionary format
    """
    tb.open(msfile)
    t0 = tb.getcell('TIME', 0)
    tb.close()
    ovro = me.observatory('OVRO_MMA')
    timeutc = me.epoch('UTC', '%fs' % t0)
    me.doframe(ovro)
    me.doframe(timeutc)
    d0 = me.direction('SUN')
    d0_j2000 = me.measure(d0, 'J2000')
    if str_output:
        d0_j2000_str = 'J2000 %frad %frad' % (d0_j2000['m0']['value'], d0_j2000['m1']['value'])
        return d0_j2000_str
    return d0_j2000


def get_msinfo(msfile):
    """
    Return some basic information of an OVRO-LWA measurement set
    :param msfile: path to CASA measurement set
    :return: number of antennas, number of spectral windows, number of channels
    """
    msmd = msmetadata()
    msmd.open(msfile)
    nant = msmd.nantennas()  # number of antennas
    nspw = msmd.nspw()  # number of spectral windows
    nchan = msmd.nchan(0)  # number of channels of the first spw
    msmd.close()
    return nant, nspw, nchan


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
                                flag_either_pol=False, doappend=False, debug=False, doplot=False):
    """Generates a text file containing the bad antennas.
    DOES NOT ACTUALLY APPLY FLAGS. CURRENTLY SHOULD ONLY BE RUN ON SINGLE SPW MSs.

    Adapted from the flag_ants_from_postcal_autocorr() module in
    https://github.com/ovro-lwa/distributed-pipeline/blob/main/orca/flagging/flag_bad_ants.py
    
    Args:
        :param msfile: string
        :param datacolumn: specify which data column to use. Default to "DATA".
            Could be "CORRECTED_DATA" if the dataset is calibrated
        :param tavg: If set to True, will time average before evaluating flags.
        :param antflagfile: Output file that contains the flagged antennas. If not defined, use msfile.replace('.ms', 'antflags')
        :param thresh_core: Threshold to use for flagging for core antennas. Default is 1.
        :param thresh_exp: Threshold to use for flagging for expansion antennas. Default is 1.
        :param flag_exp_with_core_stat: If True, use statistics of core antennas to determine flags for outer antennas
        :param flag_either_pol: If True, the antenna will be flagged if either polarization is bad (OR scheme).
                              If False, the antenna will be flagged only if both polarizations are bad (AND scheme).

    Returns:
        Path to the text file with the list of antennas to flag (antflagfile).
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
        logging.info('!! Use core antenna statistics to flag outer antennas !!')
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
        flag_core_names = msmd.antennanames(flagscore)
        flag_core_vals = autos_ampdb[flagscore]
        flag_exp_ids = ",".join([str(flag) for flag in np.sort(flagsexp)])
        flag_exp_names = msmd.antennanames(flagsexp)
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
        msfile: msfile to generate
    Returns:
        Path to the text file with list of antennas to flag.
    """
    nant, nspw, nchan = get_msinfo(msfile)
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
        logging.info('Generating antenna flags from auto-correlation')
        res = gen_ant_flags_from_autocorr(msfile, antflagfile=antflagfile, datacolumn=datacolumn,
                                          thresh_core=thresh_core, thresh_exp=thresh_exp)
    if os.path.isfile(antflagfile):
        with open(antflagfile, 'r') as f:
            antenna_list = f.readline()
            print('Applying flags for these antennas')
            print(antenna_list)
            logging.info('Flagging antennas ' + antenna_list)
        flagdata(vis=msfile, mode='manual', antenna=antenna_list)
    else:
        logging.info("No flag is found. Do nothing")
        print("No flag is found. Do nothing")
    return antflagfile


def gen_calibration(msfile, modelcl=None, uvrange='>20lambda', bcaltb=None, logging_level='info', caltable_fold='caltables'):
    """
    This function is for doing initial self-calibrations using strong sources that are above the horizon
    It is recommended to use a dataset observed at night when the Sun is not in the field of view
    :param uvrange: uv range to consider for calibration. Following CASA's syntax, e.g., '>1lambda'
    :param msfile: input CASA ms visibility for calibration
    :param modelcl: input model of strong sources as a component list, produced from gen_model_cl()
    """

    if not modelcl or not (os.path.exists(modelcl)):
        print('Model component list does not exist. Generating one from scratch.')
        logging.info('Model component list does not exist. Generating one from scratch.')
       
       
        md=model_generation(vis=msfile,separate_pol=True)
        #md.point_source_model_needed=True  	    
        modelcl, ft_needed = md.gen_model_cl()
    else:
        ft_needed = True

    if ft_needed == True:
        # Put the component list to the model column
        clearcal(msfile, addmodel=True)
        ft(msfile, complist=modelcl, usescratch=True)
    # Now do a bandpass calibration using the model component list

    if not bcaltb:
        bcaltb = caltable_fold + "/" + os.path.basename(msfile).replace('.ms', '.bcal')

    logging.info("Generating bandpass solution")
    bandpass(msfile, caltable=bcaltb, uvrange=uvrange, combine='scan,field,obs', fillgaps=0)
    logging.debug("Applying the bandpass solutions")
    applycal(vis=msfile, gaintable=bcaltb)
    logging.debug("Doing a rflag run on corrected data")
    flagdata(vis=msfile, mode='rflag', datacolumn='corrected')
    logging.debug("Finding updated and final bandpass table")
    bandpass(msfile, caltable=bcaltb, uvrange=uvrange, combine='scan,field,obs', fillgaps=0)

    if logging_level == 'debug':
        utils.get_flagged_solution_num(bcaltb)
    return bcaltb


def apply_calibration(msfile, gaintable=None, doantflag=False, doflag=False, antflagfile=None, do_solar_imaging=True,
                      imagename='test'):
    if doantflag:
        logging.info("Flagging using auro-correlation")
        flag_bad_ants(msfile, antflagfile=antflagfile)
    if not gaintable:
        logging.error("No bandpass table found. Proceed with extreme caution")
        print('No calibration table is provided. Abort... ')
    else:
        if type(gaintable) == str:
            gaintable = [gaintable]
    # Apply the calibration
    clearcal(msfile)
    applycal(msfile, gaintable=gaintable, flagbackup=True, applymode='calflag')
    if doflag == True:
        logging.debug("Running rflag on corrected data")
        flagdata(vis=msfile, mode='rflag', datacolumn='corrected')
    sunpos = get_sun_pos(msfile)
    if do_solar_imaging:
        tclean(msfile, imagename=imagename, imsize=[512], cell=['1arcmin'],
               weighting='uniform', phasecenter=sunpos, niter=500)
        print('Solar image made {0:s}.image'.format(imagename))


def make_fullsky_image(msfile, imagename="allsky", imsize=4096, cell='2arcmin',
                       minuv=10,pol='I'):  ### minuv: minimum uv in lambda
    """
    Make all sky image with wsclean
    :param msfile: path to CASA measurement set
    :param imagename: output image name
    :param imsize: size of the image in pixels
    :param cell: pixel scale
    :param minuv: minimum uv to consider for imaging (in # of wavelengths)
    :return: produces wsclean images (fits), PSF, etc.
    """
    os.system("wsclean -no-update-model-required -weight uniform" +
              " -name " + imagename + " -size " + str(imsize) + " " + str(imsize) + " -scale " + cell +
              " -minuv-l " + str(minuv) + " -niter 1000 -pol "+pol+' '+ msfile)


def get_solar_loc_pix(msfile, image="allsky"):
    """
    Get the x, y pixel location of the Sun from an all-sky image
    :param msfile: path to CASA measurement set
    :param image: all sky image made from the measurement set
    :return: pixel value in X and Y for solar disk center
    """
    from astropy.wcs.utils import skycoord_to_pixel
    m = get_sun_pos(msfile, str_output=False)
    ra = m['m0']['value']
    dec = m['m1']['value']
    coord = SkyCoord(ra * u.rad, dec * u.rad, frame='icrs')
    logging.info('RA, Dec of Sun is radians:' + str(ra) + "," + str(dec))
    head=fits.getheader(image)
    w = WCS(head)
    pix = skycoord_to_pixel(coord, w)
    x = int(pix[0])
    y = int(pix[1])
    logging.info('RA, Dec of Sun is ' + str(ra) + "pix," + str(dec) + ",pix in imagename " + image)
    return x, y


def get_nonsolar_sources_loc_pix(msfile, image="allsky", verbose=False, min_beam_val=1e-6):
    """
    Converting the RA & DEC coordinates of nonsolar sources to image coordinates in X and Y
    :param image: input CASA image
    :return: an updated directionary of strong sources with 'xpix' and 'ypix' added
    """
    from astropy.wcs.utils import skycoord_to_pixel
    srcs = utils.get_strong_source_list()
    tb.open(msfile)
    t0 = tb.getcell('TIME', 0)
    tb.close()
    # me.set_data_path('/opt/astro/casa-data')
    ovro = me.observatory('OVRO_MMA')
    time = me.epoch('UTC', '%fs' % t0)
    me.doframe(ovro)
    me.doframe(time)
    

    for i in range(len(srcs) - 1, -1, -1):
        src = srcs[i]
        coord = src['position'].split()
        d0 = None
        if len(coord) == 1:
            d0 = me.direction(coord[0])
            d0_j2000 = me.measure(d0, 'J2000')
            src['position'] = 'J2000 %frad %frad' % (d0_j2000['m0']['value'], d0_j2000['m1']['value'])
        elif len(coord) == 3:
            coord[2] = generate_calibrator_model.conv_deg(coord[2])
            d0 = me.direction(coord[0], coord[1], coord[2])
            d0_j2000 = me.measure(d0, 'J2000')
        else:
            raise Exception("Unknown direction")
        d = me.measure(d0, 'AZEL')
        elev = d['m1']['value']*180/np.pi
        az=d['m0']['value']*180/np.pi
        scale=np.sin(elev*np.pi/180)**1.6  ### sufficient for doing this check
        if elev > 0 and scale > min_beam_val:
            ra = d0_j2000['m0']['value']
            dec = d0_j2000['m1']['value']
            coord = SkyCoord(ra * u.rad, dec * u.rad, frame='icrs')
            
            head = fits.getheader(image)
            
            w = WCS(head)
            pix = skycoord_to_pixel(coord, w)
            x = int(pix[0])
            y = int(pix[1])
            srcs[i]['xpix'] = x
            srcs[i]['ypix'] = y
            logging.debug('Found source {0:s} at pix x {1:d}, y {2:d}'.format(srcs[i]['label'], x, y))
            if verbose:
                print('Found source {0:s} at pix x {1:d}, y {2:d}'.format(srcs[i]['label'], x, y))
        else:
            logging.debug('Source {0:s} has a <0 elevation or very low gain'.format(srcs[i]['label']))
            if verbose:
                print('Source {0:s} has a <0 elevation or very low gain'.format(srcs[i]['label']))
            del srcs[i]
    return srcs


def gen_nonsolar_source_model(msfile, imagename="allsky", outimage=None, sol_area=400., src_area=200.,
                              remove_strong_sources_only=True, verbose=True,pol='I'):
    """
    Take the full sky image, remove non-solar sources from the image
    :param msfile: path to CASA measurement set
    :param imagename: input all sky image
    :param outimage: output all sky image without other sources
    :param sol_area: size around the Sun in arcmin to be left alone
    :param src_area: size around the source to be taken away
    :param remove_strong_sources_only: If True, remove only known strong sources.
        If False, remove everything other than Sun.
    :param verbose: Toggle to print out more information
    :return: FITS image with non-solar sources removed
    """
    imagename1=imagename 
    if pol=='I':
        imagename=imagename+"-image.fits"
    else:
        imagename=imagename+"-XX-image.fits"
    if os.path.isfile(imagename)==False:
        imagename=imagename+"-I-image.fits"
    solx, soly = get_solar_loc_pix(msfile, imagename)
    srcs = get_nonsolar_sources_loc_pix(msfile, imagename)
    
    head = fits.getheader(imagename)
    if head['cunit1'] == 'deg':
        dx = np.abs(head['cdelt1'] * 60.)
    else:
        print(head['cunit1'] + ' not recognized as "deg". Model could be wrong.')
    if head['cunit2'] == 'deg':
        dy = np.abs(head['cdelt2'] * 60.)
    else:
        print(head['cunit2'] + ' not recognized as "deg". Model could be wrong.')
   
    imagename=imagename1
    for pola in ['I','XX','YY']:
        if pol=='I' and pola=='I':
            prefix=''
        elif pola=='XX' and pol!='I':
            prefix='-XX'
        elif pola=='YY' and pol!='I':
            prefix='-YY'
        else:
            continue
        print (pola,pol) 
        data = fits.getdata(imagename + prefix+"-model.fits")
        head=fits.getheader(imagename + prefix+"-model.fits")
        if remove_strong_sources_only:
            new_data = np.zeros_like(data)
            src_area_xpix = src_area / dx
            src_area_ypix = src_area / dy
            for s in srcs:
                src_x = s['xpix']
                src_y = s['ypix']
                bbox = [[src_y - src_area_ypix // 2, src_y + src_area_ypix // 2],
                        [src_x - src_area_xpix // 2, src_x + src_area_xpix // 2]]
                slicey, slicex = slice(int(bbox[0][0]), int(bbox[0][1]) + 1), slice(int(bbox[1][0]), int(bbox[1][1]) + 1)
                new_data[0, 0, slicey, slicex] = data[0, 0, slicey, slicex]
                if verbose:
                    print('Adding source {0:s} to model at x {1:d}, y {2:d} '
                          'with flux {3:.1f} Jy'.format(s['label'], src_x, src_y, np.max(data[0, 0, slicey, slicex])))
        else:
            new_data = np.copy(data)
            sol_area_xpix = int(sol_area / dx)
            sol_area_ypix = int(sol_area / dy)
            new_data[0, 0, soly - sol_area_ypix // 2:soly + sol_area_ypix // 2 + 1,
            solx - sol_area_xpix // 2:solx + sol_area_xpix // 2 + 1] = 0.0000

        if not outimage:
            outimage = imagename + "_no_sun"
        print (outimage+prefix+'-model.fits')
        fits.writeto(outimage + prefix+'-model.fits', new_data, header=head, overwrite=True)
    return outimage


def predict_model(msfile, outms, image="_no_sun",pol='I'):
    """
    Predict a model measurement set from an image. In the pipeline, it is
    used for transforming a model all sky image without the Sun to the output ms, and write it into the model column
    :param msfile: input CASA measurement set
    :param outms: output CASA measurement set
    :param image: input all sky image with non-solar sources, generated by gen_nonsolar_source_model()
    :return: N/A, but with an output CASA measurement set written into the same area as in the input ms
    """
    os.system("cp -r " + msfile + " " + outms)
    clearcal(outms, addmodel=True)
    os.system("wsclean -predict -pol "+pol+" -name " + image + " " + outms)


def remove_nonsolar_sources(msfile, imagename='allsky', imsize=4096, cell='2arcmin', minuv=0,
                            remove_strong_sources_only=True,pol='I'):
    """
    Wrapping for removing the nonsolar sources from the solar measurement set
    :param msfile: input CASA measurement set
    :param imagename: name of the all sky image
    :param imsize: size of the image in pixels
    :param cell: pixel scale
    :param minuv: minimum uv to consider for imaging (in # of wavelengths)
    :return: a CASA measurement set with non-solar sources removed. Default name is "*_sun_only.ms"
    """
    outms = msfile[:-3] + "_sun_only.ms"
    if os.path.isdir(outms):
        return outms
    run_wsclean(msfile=msfile, imagename=imagename, imsize=imsize, cell=cell, uvrange=minuv, predict=False,
                automask_thresh=5,pol=pol)
    image_nosun = gen_nonsolar_source_model(msfile, imagename=imagename,
                                            remove_strong_sources_only=remove_strong_sources_only,pol=pol)
    predict_model(msfile, outms="temp.ms", image=image_nosun,pol=pol)
    uvsub("temp.ms")
    split(vis="temp.ms", outputvis=outms, datacolumn='corrected')
    os.system("rm -rf temp.ms")
    return outms


def make_solar_image(msfile, imagename='sun_only',
                     imsize=512, cell='1arcmin', niter=500, uvrange='', psfcutoff=0.5):
    """
    Simple wrapper of CASA's tclean to make a solar image center at the solar disk center
    :param msfile: input CASA measurement set
    :param imagename: output image name
    :param imsize: size of the image in pixels
    :param cell: pixel scale
    :param niter: number of iterations
    :param uvrange: uvrange following tclean's syntax
    :param psfcutoff: cutoff level to fit the PSF
    :return: resulting CASA image
    """
    sunpos = get_sun_pos(msfile)
    tclean(msfile, imagename=imagename, uvrange=uvrange, imsize=imsize, cell=cell,
           weighting='uniform', phasecenter=sunpos, niter=niter, psfcutoff=psfcutoff)


def correct_ms_bug(msfile):
    """
    Temporary fix for the visibility files produced by the current pipeline.
    Update: initially found in August 2022. Not needed as of April 2023.
    :param msfile: input CASA measurement set
    """
    tb.open(msfile + "/SPECTRAL_WINDOW", nomodify=False)
    meas_freq_ref = tb.getcol('MEAS_FREQ_REF')
    if meas_freq_ref[0] == 0:
        meas_freq_ref[0] = 1
    tb.putcol('MEAS_FREQ_REF', meas_freq_ref)
    tb.flush()
    tb.close()


def do_selfcal(msfile, num_phase_cal=3, num_apcal=5, applymode='calflag', logging_level='info',
               ms_keyword='di_selfcal_time',pol='I'):
    logging.info('The plan is to do ' + str(num_phase_cal) + " rounds of phase selfcal")
    logging.info('The plan is to do ' + str(num_apcal) + " rounds of amplitude-phase selfcal")
    
    num_pol=2
    if pol=='XX,YY':
        num_pol=4

    max1 = np.zeros(num_pol)
    min1 = np.zeros(num_pol)
    
    for i in range(num_phase_cal):
        imagename = msfile[:-3] + "_self" + str(i)
        run_wsclean(msfile, imagename=imagename,pol=pol)
        
        
        good = utils.check_image_quality(imagename, max1, min1)
        
        print(good)
        logging.debug('Maximum pixel values are: ' + str(max1[0]) + "," + str(max1[1]))
        logging.debug('Minimum pixel values around peaks are: ' + str(min1[0]) + "," + str(min1[1]))
        if not good:
            logging.info('Dynamic range has reduced. Doing a round of flagging')
            flagdata(vis=msfile, mode='rflag', datacolumn='corrected')
            run_wsclean(msfile, imagename=imagename,pol=pol)
            
            good = utils.check_image_quality(imagename, max1, min1)
            
            print(good)
            logging.debug('Maximum pixel values are: ' + str(max1[0]) + "," + str(max1[1]))
            logging.debug('Minimum pixel values around peaks are: ' + str(min1[0]) + "," + str(min1[1]))
            if not good:
                logging.info('Flagging could not solve the issue. Restoring flags, applying last good solutions.')
                utils.restore_flag(msfile)
                logging.debug('Restoring flags')
                os.system("rm -rf " + imagename + "-*.fits")
                caltable = msfile[:-3] + "_self" + str(i - 1) + ".gcal"
                os.system("rm -rf " + caltable)
                imagename = msfile[:-3] + "_self" + str(i - 2)
                caltable = imagename + ".gcal"
                if os.path.isdir(caltable):
                    logging.info("Applying " + caltable)
                    applycal(vis=msfile, gaintable=caltable, calwt=[False], applymode=applymode)
                    os.system("cp -r " + caltable + " caltables/")
                else:
                    logging.warning("No caltable found. Setting corrected data to DATA")
                    clearcal(msfile)
                return good
        logging.debug("Finding gain solutions and writing in into " + imagename + ".gcal")
        gaincal(vis=msfile, caltable=imagename + ".gcal", uvrange=">10lambda",
                calmode='p', solmode='L1R', rmsthresh=[10, 8, 6])
        utils.put_keyword(imagename + ".gcal", ms_keyword, utils.get_keyword(msfile, ms_keyword))
        if logging_level == 'debug' or logging_level == 'DEBUG':
            utils.get_flagged_solution_num(imagename + ".gcal")
        logging.debug("Applying solutions")
        applycal(vis=msfile, gaintable=imagename + ".gcal", calwt=[False], applymode=applymode)

    logging.info("Phase self-calibration finished successfully")

    if num_phase_cal > 0:
        final_phase_caltable = imagename + ".gcal"
    else:
        final_phase_caltable = ''
    for i in range(num_phase_cal, num_phase_cal + num_apcal):
        imagename = msfile[:-3] + "_self" + str(i)
        run_wsclean(msfile, imagename=imagename,pol=pol)
        
        good = utils.check_image_quality(imagename, max1, min1)
        
        logging.debug('Maximum pixel values are: ' + str(max1[0]) + "," + str(max1[1]))
        logging.debug('Minimum pixel values around peaks are: ' + str(min1[0]) + "," + str(min1[1]))
        if not good:
            logging.info('Dynamic range has reduced. Doing a round of flagging')
            flagdata(vis=msfile, mode='rflag', datacolumn='corrected')
            run_wsclean(msfile, imagename=imagename,pol=pol)
            
            good = utils.check_image_quality(imagename, max1, min1)
            
            print(good)
            if not good:
                logging.info('Flagging could not solve the issue. Restoring flags, applying last good solutions.')
                utils.restore_flag(msfile)
                os.system("rm -rf " + imagename + "-*.fits")
                caltable = msfile[:-3] + "_self" + str(i - 1) + "_ap_over_p.gcal"
                os.system("rm -rf " + caltable)
                imagename = msfile[:-3] + "_self" + str(i - 2)
                caltable = imagename + "_ap_over_p.gcal"
                if os.path.isdir(caltable):
                    logging.info("Applying " + caltable + " and " + final_phase_caltable)
                    if num_phase_cal > 0:
                        applycal(vis=msfile, gaintable=[caltable, final_phase_caltable], calwt=[False, False],
                                 applymode=applymode)
                        os.system("cp -r " + final_phase_caltable + " caltables/")
                    else:
                        applycal(vis=msfile, gaintable=[caltable], calwt=[False, False], applymode=applymode)
                    os.system("cp -r " + caltable + " caltables/")

                else:
                    logging.warning("No good aplitude-phase selfcal solution found.")
                    if num_phase_cal > 0:
                        logging.info("Applying " + final_phase_caltable)
                        applycal(vis=msfile, gaintable=[final_phase_caltable], calwt=[False], applymode=applymode)
                        os.system("cp -r " + final_phase_caltable + " caltables/")
                    else:
                        logging.warning("No caltable found. Setting corrected data to DATA")
                        clearcal(msfile)
                return good
        caltable = imagename + "_ap_over_p.gcal"

        gaincal(vis=msfile, caltable=caltable, uvrange=">10lambda",
                calmode='ap', solnorm=True, normtype='median', solmode='L1R',
                rmsthresh=[10, 8, 6], gaintable=final_phase_caltable)
        utils.put_keyword(caltable, ms_keyword, utils.get_keyword(msfile, ms_keyword))
        if logging_level == 'debug' or logging_level == 'DEBUG':
            utils.get_flagged_solution_num(imagename + "_ap_over_p.gcal")
        applycal(vis=msfile, gaintable=[caltable, final_phase_caltable], calwt=[False, False], applymode=applymode)
        if i == num_phase_cal:
            flagdata(vis=msfile, mode='rflag', datacolumn='corrected')
    logging.debug('Flagging on the residual')
    flagdata(vis=msfile, mode='rflag', datacolumn='residual')
    os.system("cp -r " + caltable + " caltables/")
    os.system("cp -r " + final_phase_caltable + " caltables/")
    return True


def run_wsclean(msfile, imagename, automask_thresh=8, imsize=4096, cell='2arcmin', uvrange='10',
                predict=True,pol='I'):  ### uvrange is in lambda units
    logging.debug("Running WSCLEAN")
    os.system("wsclean -no-dirty -no-update-model-required -no-negative -size " + str(imsize) + " " + \
              str(imsize) + " -scale " + cell + " -weight uniform -minuv-l " + str(uvrange) + " -auto-mask " + str(
        automask_thresh) + \
              " -niter 100000 -name " + imagename + " -mgain 0.7 -beam-fitting-size 2 -pol "+pol+' ' + msfile)
    if predict:
        logging.debug("Predicting model visibilities from " + imagename + " in " + msfile)
        os.system("wsclean -predict -pol "+pol+" "+ "-name " + imagename + " " + msfile)


def change_phasecenter(msfile):
    m = get_sun_pos(msfile, str_output=False)
    ra = m['m0']['value']  ### ra in radians
    dec = m['m1']['value']  ### dec in radians
    logging.debug('Solar ra in radians: ' + str(m['m0']['value']))
    logging.debug('Solar dec in radians: ' + str(m['m1']['value']))
    print(ra, dec)
    neg = False
    if ra < 0:
        neg = True
        ra = -ra
    ra = ra * 180 / (np.pi * 15)  ### ra in hours
    dec = dec * 180 / np.pi  ### dec in deg
    print(ra)
    temp = int(ra)
    print(temp)
    ra1 = str(temp) + "h"
    ra = ra - temp
    temp = int(ra * 60)
    ra1 = ra1 + str(temp) + "m"
    ra = (ra * 60 - temp)
    ra1 = ra1 + str(ra) + "s"
    if neg == True:
        ra1 = '-' + ra1
    print(dec)
    neg = False
    if dec < 0:
        neg = True
        dec = -dec
    temp = int(dec)
    dec1 = str(temp) + "d"
    dec = dec - temp
    temp = int(dec * 60)
    dec1 = dec1 + str(temp) + "m"
    dec = dec * 60 - temp
    dec1 = dec1 + str(dec) + "s"
    if neg == True:
        dec1 = '-' + dec1
    print(ra1)
    print(dec1)
    logging.info("Changing the phasecenter to " + ra1 + " " + dec1)
    os.system("chgcentre " + msfile + " " + ra1 + " " + dec1)


def get_point_flux(modelcl, src,pol=''):
    tb.open(modelcl)
    flux = tb.getcol('Flux')
    names = tb.getcol('Label')
    tb.close()
    for i, name in enumerate(names):
        if name == src['label']:
            if pol=='':
            	return abs(np.real(flux[0, i]))
            elif pol=='-XX':
                return (np.real(flux[0, i])+np.real(flux[1, i]))  ##X=(I+Q)
            elif pol=='-YY':
                return (np.real(flux[0, i])-np.real(flux[1, i]))  ##y=(I-Q)
    logging.warning("There is no matching source in the Component list " + \
                    "corresponding to " + src['label'])
    return -1


def correct_flux_scaling(msfile, src_area=100, min_beam_val=0.1, caltable_suffix='fluxscale',pol='I'):
    import glob
    
    os.system("rm -rf calibrator-model.fits calibrator-XX-model.fits calibrator-YY-model.fits")
    mstime_str = utils.get_timestr_from_name(msfile)
    di_selfcal_str, success = utils.get_keyword(msfile, 'di_selfcal_time', return_status=True)

    if di_selfcal_str == mstime_str and success:
        if pol=='I':
            md=model_generation(vis=msfile,pol=pol,separate_pol=False)
        else:
            md=model_generation(vis=msfile,pol=pol,separate_pol=True)
        md.predict=False
        md.model=False
        md.min_beam_val=min_beam_val
        modelcl, ft_needed = md.gen_model_cl()
        if pol=='I':
            images = glob.glob(msfile[:-3] + "_self*-image.fits")
        else:
            images= glob.glob(msfile[:-3] + "_self*XX-image.fits")
        num_image = len(images)
        
        
        mean_factor=[]
        pol_prefix=[]
        for pola in ['I','XX','YY']:
            if pol=='I' and pola=='I':
                prefix=''
            elif pola=='XX' and pol!='I':
                prefix='-XX'
            elif pola=='YY' and pol!='I':
                prefix='-YY'
            else:
                continue
            final_image = msfile[:-3] + "_self" + str(num_image - 1) + prefix+"-image.fits"
        
            srcs = get_nonsolar_sources_loc_pix(msfile, final_image, min_beam_val=min_beam_val)
            head = fits.getheader(final_image)

            if head['cunit1'] == 'deg':
                dx = np.abs(head['cdelt1'] * 60.)
            elif head['cunit1'] == 'asec':
                dx = np.abs(head['cdelt1'] / 60.)
            else:
                logging.warning(head['cunit1'] + ' not recognized as "deg" or "asec". Model could be wrong.')
                print(head['cunit1'] + ' not recognized as "deg" or "asec". Model could be wrong.')
            if head['cunit2'] == 'deg':
                dy = np.abs(head['cdelt2'] * 60.)
            elif head['cunit2'] == 'asec':
                dx = np.abs(head['cdelt2'] / 60.)
            else:
                logging.warning(head['cunit2'] + ' not recognized as "deg" or asec. Model could be wrong.')
                print(head['cunit2'] + ' not recognized as "deg" or "asec". Model could be wrong.')
            src_area_xpix = src_area / dx
            src_area_ypix = src_area / dy
            scaling_factor = []
            for s in srcs:
                src_x = s['xpix']
                src_y = s['ypix']
                bbox = [[src_y - src_area_ypix // 2, src_y + src_area_ypix // 2],
                        [src_x - src_area_xpix // 2, src_x + src_area_xpix // 2]]

                if os.path.isfile('calibrator'+prefix+'-model.fits') == False:
                    model_flux = get_point_flux(modelcl,
                                                s,prefix)  ### if wsclean failed, then Component List was generated in gen_model_cl
                else:
        
                    model_flux = imstat(imagename='calibrator'+prefix+'-model.fits', box=str(src_x - src_area_xpix // 2) + "," + \
                                                                               str(src_y - src_area_ypix // 2) + "," + \
                                                                               str(src_x + src_area_xpix // 2) + "," + \
                                                                               str(src_y + src_area_ypix // 2))['flux'][0]
                if model_flux < 0:
                    logging.warning('Model flux is negative. Picking flux from point source model')
                    model_flux = get_point_flux(modelcl,
                                                s,prefix)  ### if model had negative, then Component List was generated in gen_model_cl
                logging.info('Model flux of ' + s['label'] + ' is  ' + str(model_flux))
                image_flux = imstat(imagename=final_image, box=str(src_x - src_area_xpix // 2) + "," + \
                                                               str(src_y - src_area_ypix // 2) + "," + \
                                                               str(src_x + src_area_xpix // 2) + "," + \
                                                               str(src_y + src_area_ypix // 2))['flux'][0]
                logging.info('Model flux of ' + s['label'] + ' is  ' + str(image_flux))
                # print (image_flux)
                print(s['label'], image_flux, model_flux)
                if (model_flux > 0 and image_flux > 0):
                    scaling_factor.append(model_flux / image_flux)
                    logging.info('Scaling factor obtained from ' + s['label'] + ' is ' + str(scaling_factor[-1]))
                else:
                    logging.warning('Scaling factor is not calculated for ' + s[
                        'label'] + ' as either/both model and image flux is negative')
            if len(scaling_factor) > 0:
                mean_factor.append(np.mean(np.array(scaling_factor)))
                print(scaling_factor)
                print(mean_factor)
                if prefix=='-XX':
                    pol_prefix.append('X')
                elif prefix=='-YY':
                    pol_prefix.append('Y')
                else:
                    pol_prefix.append('')
                logging.info('Scaling factor is for '+prefix+":" + str(mean_factor))

        logging.debug("Generating caltable for fluxscaling. Filename is " + msfile[:-3] + "." + caltable_suffix)
        caltable = msfile[:-3] + "." + caltable_suffix
        pol=','.join(pol_prefix)
        
        gencal(vis=msfile, caltable=caltable, caltype='amp', parameter=np.sqrt(1. / np.array(mean_factor) ),pol=pol)

        os.system("cp -r " + caltable + " caltables/")
    elif success == True:
        caltable = glob.glob("caltables/" + di_selfcal_str + "*.fluxscale")[0]
        logging.info("Applying {0:s} for doing fluxscaling".format(caltable))
    else:
        caltable = msfile[:-3] + "." + caltable_suffix
        gencal(vis=msfile, caltable=caltable, caltype='amp', parameter=1)
        logging.warning("Could not find appropriate flux scaling factor. No correction will be done.")

    DI_val = utils.get_keyword(msfile, 'di_selfcal_time')

    logging.debug('Correcting the DATA with the scaling factor')
    temp_file = 'temp_' + msfile

    split(vis=msfile, outputvis=temp_file)

    applycal(vis=temp_file, gaintable=caltable, calwt=False)

    os.system("rm -rf " + msfile + "*")

    split(vis=temp_file, outputvis=msfile)
    os.system("rm -rf " + temp_file + "*")
    utils.put_keyword(msfile, 'di_selfcal_time', DI_val)
    return


def correct_primary_beam(msfile, imagename,pol='I'):
    m = get_sun_pos(msfile, str_output=False)
    logging.debug('Solar ra: ' + str(m['m0']['value']))
    logging.debug('Solar dec: ' + str(m['m1']['value']))
    d = me.measure(m, 'AZEL')
    logging.debug('Solar azimuth: ' + str(d['m0']['value']))
    logging.debug('Solar elevation: ' + str(d['m1']['value']))
    elev = d['m1']['value']*180/np.pi
    az=d['m0']['value']*180/np.pi
    pb=beam(msfile=msfile)
    jones_matrix=pb.srcIQUV(az=az,el=elev)
    
    if pol=='I':
        scale=primary_beam.primary_beam_correction_val('I',jones_matrix)
        logging.info('The Stokes I beam correction factor is ' + str(round(scale, 4)))
        hdu = fits.open(imagename+ "-image.fits", mode='update')
        hdu[0].data /= scale
        hdu.flush()
        hdu.close()
    elif pol=='XX,YY':
        for n,pola in enumerate(['XX','YY']):
            scale=primary_beam.primary_beam_correction_val(pola,jones_matrix)
            logging.info('The Stokes '+pola+' beam correction factor is ' + str(round(scale, 4)))
            hdu = fits.open(imagename+ "-"+pola+"-image.fits", mode='update')
            hdu[0].data /= scale
            hdu.flush()
            hdu.close()
    return


def do_bandpass_correction(solar_ms, calib_ms=None, bcal=None, caltable_fold='caltables', logging_level='info',pol='I'):
    solar_ms1 = solar_ms[:-3] + "_calibrated.ms"
    if os.path.isdir(solar_ms1):
        return solar_ms1
    if not bcal or os.path.isdir(bcal) == False:
        logging.debug('Bandpass table not supplied or is not present on disc. Creating one' + \
                      ' from the supplied MS')
        if os.path.exists(calib_ms):
            logging.debug('Flagging all data which are zero')
            flagdata(vis=calib_ms, mode='clip', clipzeros=True)
            logging.debug('Flagging antennas before calibration.')
            flag_bad_ants(calib_ms)
            bcal = gen_calibration(calib_ms, logging_level=logging_level, caltable_fold=caltable_fold)
            logging.info('Bandpass calibration table generated using ' + calib_ms)
        else:
            print('Neither calib_ms nor bcal exists. Need to provide calibrations to continue. Abort..')
            logging.error('Neither calib_ms nor bcal exists. Need to provide calibrations to continue. Abort..')
    # correct_ms_bug(solar_ms)

    apply_calibration(solar_ms, gaintable=bcal, doantflag=True, doflag=True, do_solar_imaging=False)
    split(vis=solar_ms, outputvis=solar_ms[:-3] + "_calibrated.ms")
    logging.info('Splitted the input solar MS into a file named ' + solar_ms[:-3] + "_calibrated.ms")
    solar_ms = solar_ms[:-3] + "_calibrated.ms"
    return solar_ms


def do_fresh_selfcal(solar_ms, num_phase_cal=3, num_apcal=5, logging_level='info',pol='I'):
    """
    Do fresh self-calibration if no self-calibration tables are found
    :param solar_ms: input solar visibility
    :param num_phase_cal: (maximum) rounds of phase-only selfcalibration. Default to 3
    :param num_apcal: (maximum) rounds of ampitude and phase selfcalibration. Default to 5
    :param logging_level: type of logging, default to "info"
    :return: N/A
    """
    logging.info('Starting to do direction independent Stokes I selfcal')
    success = do_selfcal(solar_ms, num_phase_cal=num_phase_cal, num_apcal=num_apcal, logging_level=logging_level,pol=pol)
    if not success:
#TODO Understand why this step is needed
        logging.info('Starting fresh selfcal as DR decreased significantly')
        clearcal(solar_ms)
        success = do_selfcal(solar_ms, num_phase_cal=num_phase_cal, num_apcal=num_apcal, logging_level=logging_level,pol=pol)
    return


def DI_selfcal(solar_ms, solint_full_selfcal=14400, solint_partial_selfcal=3600,
               full_di_selfcal_rounds=[1,1], partial_di_selfcal_rounds=[1, 1], logging_level='info',pol='I'):
    """
    Directional-independent self-calibration (full sky)
    :param solar_ms: input solar visibility
    :param solint_full_selfcal: interval for doing full self-calibration in seconds. Default to 4 hours
    :param solint_partial_selfcal: interval for doing partial self-calibration in seconds. Default to 1 hour.
    :param full_di_selfcal_rounds: [rounds of phase-only selfcal, rounds of amp-phase selfcal]
            for full selfcalibration runs
    :param partial_di_selfcal_rounds: [rounds of phase-only selfcal, rounds of amp-phase selfcal]
            for partial selfcalibration runs
    :param logging_level: level of logging
    :return: N/A
    """

    solar_ms1 = solar_ms[:-3] + "_selfcalibrated.ms"
    if os.path.isdir(solar_ms1) == True:
        return solar_ms1
    
    sep = 100000000
    prior_selfcal = False
    caltables = []

    mstime = utils.get_time_from_name(solar_ms)
    mstime_str = utils.get_timestr_from_name(solar_ms)

    caltables = glob.glob("caltables/*.gcal")
    if len(caltables) != 0:
        prior_selfcal = True

    if prior_selfcal:
        dd_cal = glob.glob("caltables/*sun_only*.gcal")
        di_cal = [cal for cal in caltables if cal not in dd_cal]
        print(di_cal)
        selfcal_time = utils.get_selfcal_time_to_apply(solar_ms, di_cal)
        print(selfcal_time)

        caltables = glob.glob("caltables/" + selfcal_time + "*.gcal")
        dd_cal = glob.glob("caltables/" + selfcal_time + "*sun_only*.gcal")
        di_cal = [cal for cal in caltables if cal not in dd_cal]

        if len(di_cal) != 0:
            di_selfcal_time_str, success = utils.get_keyword(di_cal[0], 'di_selfcal_time', return_status=True)
            print(di_selfcal_time_str, success)
            if success:
                di_selfcal_time = utils.get_time_from_name(di_selfcal_time_str)

                sep = abs((di_selfcal_time - mstime).value * 86400)  ### in seconds

                applycal(solar_ms, gaintable=di_cal, calwt=[False] * len(di_cal))
                flagdata(vis=solar_ms, mode='rflag', datacolumn='corrected')

                if sep < solint_partial_selfcal:
                    logging.info('No direction independent Stokes I selfcal after applying ' + di_selfcal_time_str)
                    success = utils.put_keyword(solar_ms, 'di_selfcal_time', di_selfcal_time_str, return_status=True)


                elif sep > solint_partial_selfcal and sep < solint_full_selfcal:
                    # Partical selfcal does one additional round of ap self-calibration
                    success = utils.put_keyword(solar_ms, 'di_selfcal_time', mstime_str, return_status=True)
                    logging.info(
                        'Starting to do direction independent Stokes I selfcal after applying ' + di_selfcal_time_str)
                    success = do_selfcal(solar_ms, num_phase_cal=0,
                                         num_apcal=partial_di_selfcal_rounds[1], logging_level=logging_level,pol=pol)
                    datacolumn = 'corrected'

                else:
                    # Full selfcal does 5 additional rounds of ap self-calibration
                    success = utils.put_keyword(solar_ms, 'di_selfcal_time', mstime_str, return_status=True)
                    logging.info(
                        'Starting to do direction independent Stokes I selfcal after applying ' + di_selfcal_time_str)
                    success = do_selfcal(solar_ms, num_phase_cal=0,
                                         num_apcal=full_di_selfcal_rounds[1], logging_level=logging_level,pol=pol)
                    datacolumn = 'corrected'
                    if success == False:
                        clearcal(solar_ms)
                        success = do_selfcal(solar_ms, logging_level=logging_level,pol=pol)
            else:
                success = utils.put_keyword(solar_ms, 'di_selfcal_time', mstime_str, return_status=True)
                logging.info(
                    'Starting to do direction independent Stokes I selfcal as I failed to retrieve the keyword for DI selfcal')
                do_fresh_selfcal(solar_ms, num_phase_cal=full_di_selfcal_rounds[0],
                                 num_apcal=full_di_selfcal_rounds[1], logging_level=logging_level,pol=pol)
        else:
            success = utils.put_keyword(solar_ms, 'di_selfcal_time', mstime_str, return_status=True)
            logging.info(
                'Starting to do direction independent Stokes I selfcal as mysteriously I did not find a suitable caltable')
            do_fresh_selfcal(solar_ms, num_phase_cal=full_di_selfcal_rounds[0],
                             num_apcal=full_di_selfcal_rounds[1], logging_level=logging_level,pol=pol)
    else:
        success = utils.put_keyword(solar_ms, 'di_selfcal_time', mstime_str, return_status=True)
        logging.info('Starting to do direction independent Stokes I selfcal')
        do_fresh_selfcal(solar_ms, num_phase_cal=full_di_selfcal_rounds[0],
                         num_apcal=full_di_selfcal_rounds[1], logging_level=logging_level,pol=pol)
    
    logging.info('Doing a flux scaling using background strong sources')
    correct_flux_scaling(solar_ms, min_beam_val=0.1,pol=pol)

    logging.info('Splitted the selfcalibrated MS into a file named ' + solar_ms[:-3] + "_selfcalibrated.ms")

    solar_ms_slfcaled = solar_ms[:-3] + "_selfcalibrated.ms"
    split(vis=solar_ms, outputvis=solar_ms_slfcaled, datacolumn='data')
    return solar_ms_slfcaled


def DD_selfcal(solar_ms, solint_full_selfcal=1800, solint_partial_selfcal=600,
               full_dd_selfcal_rounds=[3, 5], partial_dd_selfcal_rounds=[1, 1],
               logging_level='info',pol='I'):
    """
    Directional-dependent self-calibration on the Sun only
    :param solar_ms: input solar visibility
    :param solint_full_selfcal: interval for doing full self-calibration in seconds. Default to 30 min
    :param solint_partial_selfcal: interval for doing partial self-calibration in seconds. Default to 10 min.
    :param full_dd_selfcal_rounds: [rounds of phase-only selfcal, rounds of amp-phase selfcal]
            for full selfcalibration runs
    :param partial_dd_selfcal_rounds: [rounds of phase-only selfcal, rounds of amp-phase selfcal]
            for partial selfcalibration runs
    :param logging_level: level of logging
    :return: N/A
    """

    solar_ms1 = solar_ms[:-3] + "_sun_selfcalibrated.ms"
    if os.path.isdir(solar_ms1):
        return solar_ms1

    selfcal_time = utils.get_selfcal_time_to_apply(solar_ms, glob.glob("caltables/*.gcal"))
    mstime = utils.get_time_from_name(solar_ms)
    mstime_str = utils.get_timestr_from_name(solar_ms)

    sep = 100000000
    prior_selfcal = False

    caltables = glob.glob("caltables/" + selfcal_time + "*sun_only*.gcal")

    if len(caltables) != 0:
        prior_selfcal = True

    if prior_selfcal:
        dd_selfcal_time_str, success = utils.get_keyword(caltables[0], 'dd_selfcal_time', return_status=True)

        if success:
            dd_selfcal_time = utils.get_time_from_name(dd_selfcal_time_str)

            sep = abs((dd_selfcal_time - mstime).value * 86400)  ### in seconds

            applycal(solar_ms, gaintable=caltables, calwt=[False] * len(caltables), applymode='calonly')
            flagdata(vis=solar_ms, mode='rflag', datacolumn='corrected')

            if sep < solint_partial_selfcal:
                logging.info('No direction dependent Stokes I selfcal after applying ' + dd_selfcal_time_str)
                success = utils.put_keyword(solar_ms, 'dd_selfcal_time', dd_selfcal_time_str, return_status=True)

            elif sep > solint_partial_selfcal and sep < solint_full_selfcal:
                success = utils.put_keyword(solar_ms, 'dd_selfcal_time', mstime_str, return_status=True)
                logging.info(
                    'Starting to do direction dependent Stokes I selfcal after applying ' + dd_selfcal_time_str)
                success = do_selfcal(solar_ms, num_phase_cal=partial_dd_selfcal_rounds[0],
                                     num_apcal=partial_dd_selfcal_rounds[1], applymode='calonly',
                                     logging_level=logging_level, ms_keyword='dd_selfcal_time',pol=pol)
                datacolumn = 'corrected'


            else:
                success = utils.put_keyword(solar_ms, 'dd_selfcal_time', mstime_str, return_status=True)
                logging.info(
                    'Starting to do direction dependent Stokes I selfcal after applying ' + dd_selfcal_time_str)
                success = do_selfcal(solar_ms, num_phase_cal=full_dd_selfcal_rounds[0],
                                     num_apcal=full_dd_selfcal_rounds[1], applymode='calonly',
                                     logging_level=logging_level, ms_keyword='dd_selfcal_time',pol=pol)
                datacolumn = 'corrected'
        else:
            success = utils.put_keyword(solar_ms, 'dd_selfcal_time', mstime_str, return_status=True)
            logging.info(
                'Starting to do direction dependent Stokes I selfcal as I failed to retrieve the keyword for DD selfcal')
            success = do_selfcal(solar_ms, num_phase_cal=full_dd_selfcal_rounds[0],
                                 num_apcal=full_dd_selfcal_rounds[1], applymode='calonly',
                                 logging_level=logging_level, ms_keyword='dd_selfcal_time',pol=pol)



    else:
        success = utils.put_keyword(solar_ms, 'dd_selfcal_time', mstime_str, return_status=True)
        logging.info('Starting to do direction dependent Stokes I selfcal')
        success = do_selfcal(solar_ms, num_phase_cal=full_dd_selfcal_rounds[0], num_apcal=full_dd_selfcal_rounds[1],
                             applymode='calonly', logging_level=logging_level,
                             ms_keyword='dd_selfcal_time',pol=pol)

    logging.info('Splitted the selfcalibrated MS into a file named ' + solar_ms[:-3] + "_sun_selfcalibrated.ms")

    split(vis=solar_ms, outputvis=solar_ms[:-3] + "_sun_selfcalibrated.ms")
    solar_ms = solar_ms[:-3] + "_sun_selfcalibrated.ms"
    return solar_ms


def image_ms(solar_ms, calib_ms=None, bcal=None, selfcal=False, imagename='sun_only',
             imsize=1024, cell='1arcmin', logfile='analysis.log', logging_level='info',
             caltable_fold='caltables', full_di_selfcal_rounds=[4, 8], partial_di_selfcal_rounds=[0, 1],
             full_dd_selfcal_rounds=[2, 1], partial_dd_selfcal_rounds=[0, 1], do_final_imaging=True,pol='I'):
    """
    Pipeline to calibrate and imaging a solar visibility
    :param solar_ms: input solar measurement set
    :param calib_ms: (optional) input measurement set for generating the calibrations, usually is one observed at night
    :param bcal: (optional) bandpass calibration table. If not provided, use calib_ms to generate one.
    :param full_di_selfcal_rounds: [rounds of phase-only selfcal, rounds of amp-phase selfcal]
            for directional independent (full sky) full selfcalibration runs
    :param partial_di_selfcal_rounds: [rounds of phase-only selfcal, rounds of amp-phase selfcal]
            for directional independent (full sky) partial selfcalibration runs
    :param full_dd_selfcal_rounds: [rounds of phase-only selfcal, rounds of amp-phase selfcal]
            for directional-dependent full selfcalibration runs
    :param partial_dd_selfcal_rounds: [rounds of phase-only selfcal, rounds of amp-phase selfcal]
            for directional-dependent partial selfcalibration runs
    """

    if not os.path.isdir(caltable_fold):
        os.mkdir(caltable_fold)
    if os.path.isfile(imagename + "-image.fits"):
        return

    solar_ms = do_bandpass_correction(solar_ms, calib_ms=calib_ms, bcal=bcal, caltable_fold=caltable_fold,pol=pol)

    logging.info('Analysing ' + solar_ms)
    if selfcal:
        outms_di = DI_selfcal(solar_ms, logging_level=logging_level, full_di_selfcal_rounds=full_di_selfcal_rounds,
                              partial_di_selfcal_rounds=partial_di_selfcal_rounds,pol=pol)
        logging.info('Removing the strong sources in the sky')
        outms_di_ = remove_nonsolar_sources(outms_di,pol=pol)
        logging.info('The strong source subtracted MS is ' + outms_di_)
        logging.info('Starting to do Stokes I selfcal towards direction of sun')
        outms_dd = DD_selfcal(outms_di_, logging_level=logging_level, full_dd_selfcal_rounds=full_dd_selfcal_rounds,
                              partial_dd_selfcal_rounds=partial_dd_selfcal_rounds,pol=pol)
        logging.info('Removing almost all sources in the sky except Sun')
        outms = remove_nonsolar_sources(outms_dd, imagename='for_weak_source_subtraction',
                                        remove_strong_sources_only=False,pol=pol)
        logging.info('The source subtracted MS is ' + outms)
    else:
        logging.info('Removing almost all sources in the sky except Sun')
        outms = remove_nonsolar_sources(solar_ms,pol=pol)
        logging.info('The source subtracted MS is ' + outms)

    logging.info('Changing the phasecenter to position of Sun')
    change_phasecenter(outms)

    if do_final_imaging:
        logging.info('Generating final solar centered image')
        run_wsclean(outms, imagename=imagename, automask_thresh=5, uvrange='0', predict=False, imsize=imsize, cell=cell,pol=pol)
        logging.info('Correcting for the primary beam at the location of Sun')
        correct_primary_beam(outms, imagename,pol=pol)
        # make_solar_image(outms, imagename=imagename, imsize=imsize, cell=cell)
        helio_image = utils.convert_to_heliocentric_coords(outms, imagename+"-image.fits")
        logging.info('Imaging completed for ' + solar_ms)
        return outms, helio_image
    else:
        return outms


def solar_pipeline(time_duration, calib_time_duration, freqstr, filepath, time_integration=8, time_cadence=100,
                   observation_integration=8,
                   calib_ms=None, bcal=None, selfcal=False, imagename='sun_only',
                   imsize=512, cell='1arcmin', logfile='analysis.log', logging_level='info',
                   caltable_fold='caltables',pol='I'):
    if logging_level == 'info' or logging_level == 'INFO':
        logging.basicConfig(filename=logfile, level=logging.INFO)
    elif logging_level == 'warning' or logging_level == 'WARNING':
        logging.basicConfig(filename=logfile, level=logging.WARNING)
    elif logging_level == 'critical' or logging_level == 'CRITICAL':
        logging.basicConfig(filename=logfile, level=logging.CRITICAL)
    elif logging_level == 'error' or logging_level == 'ERROR':
        logging.basicConfig(filename=logfile, level=logging.ERROR)
    else:
        logging.basicConfig(filename=logfile, level=logging.DEBUG)

    fp = File_Handler(time_duration=time_duration, freqstr=freqstr, file_path=filepath, \
                      time_integration=time_integration, time_cadence=time_cadence)

    calib_fp = File_Handler(time_duration=calib_time_duration, freqstr=freqstr, file_path=filepath)

    print_str = 'Start the pipeline for imaging {0:s}'.format(time_duration)
    logging.info(print_str)
    try:
        print_str = 'Frequencies to be analysed: {0:s}'.format(','.join(freqstr))
    except:
        print_str = 'Frequencies to be analysed: {0:s}'.format(freqstr)

    logging.info(print_str)
    print_str = 'Chosen time integration and time cadence are {0:d} and {0:d}'.format(time_integration, time_cadence)

    calib_fp.start = calib_fp.parse_duration()
    calib_fp.end = calib_fp.parse_duration(get_end=True)
    calib_fp.get_selfcal_times_paths()

    calib_filename = calib_fp.get_current_file_for_selfcal(freqstr[0])

    fp.start = fp.parse_duration()
    fp.end = fp.parse_duration(get_end=True)

    fp.get_selfcal_times_paths()

    filename = fp.get_current_file_for_selfcal(freqstr[0])
    while filename is not None:
        calib_file = glob.glob(caltable_fold + '/*.bcal')
        if len(calib_file) != 0:
            bcal = calib_file[0]
        imagename = "sun_only_" + filename[:-3]
        outms, helio_image = image_ms(filename, calib_ms=calib_filename, bcal=bcal, selfcal=True,
                                    imagename=imagename, do_final_imaging=True,pol=pol)
        filename = fp.get_current_file_for_selfcal(freqstr[0])

    filename = fp.get_current_file_for_selfcal(freqstr[0])
    while filename is not None:
        imagename = "sun_only_" + filename[:-3]
        outms, helio_image = image_ms(filename, calib_ms=calib_ms, bcal=bcal, selfcal=True,
                                    imagename=imagename, do_final_imaging=True,pol=pol)
        filename = fp.get_current_file_for_imaging(freqstr[0])


def apply_solutions_and_image(msname, bcal, imagename):
    logging.info('Analysing ' + msname)
    apply_calibration(msname, gaintable=bcal, doantflag=True, doflag=True, do_solar_imaging=False)
    split(vis=msname, outputvis=msname[:-3] + "_calibrated.ms")
    msname = msname[:-3] + "_calibrated.ms"
    selfcal_time = utils.get_selfcal_time_to_apply(msname)
    logging.info('Will apply selfcal solutions from ' + selfcal_time)
    caltables = glob.glob("caltables/" + selfcal_time + "*.gcal")
    dd_cal = glob.glob("caltables/" + selfcal_time + "*sun_only*.gcal")
    di_cal = [i for i in caltables if i not in dd_cal]
    fluxscale_cal = glob.glob("caltables/" + selfcal_time + "*.fluxscale")
    di_cal.append(fluxscale_cal[0])
    applycal(msname, gaintable=di_cal, calwt=[False] * len(di_cal))
    flagdata(vis=msname, mode='rflag', datacolumn='corrected')
    split(vis=msname, outputvis=msname[:-3] + "_selfcalibrated.ms")
    solar_ms = msname[:-3] + "_selfcalibrated.ms"
    outms = remove_nonsolar_sources(solar_ms)
    solar_ms = outms
    num_dd_cal = len(dd_cal)
    if num_dd_cal != 0:
        applycal(solar_ms, gaintable=dd_cal, calwt=[False] * len(dd_cal), applymode='calonly')
        flagdata(vis=solar_ms, mode='rflag', datacolumn='corrected')
        split(vis=solar_ms, outputvis=solar_ms[:-3] + "_sun_selfcalibrated.ms")
    else:
        split(vis = solar_ms, outputvis=solar_ms[:-3]+"_sun_selfcalibrated.ms",datacolumn='data')
    outms = solar_ms[:-3] + "_sun_selfcalibrated.ms"
    outms = remove_nonsolar_sources(outms,imagename='for_weak_source_subtraction',remove_strong_sources_only=False)
    change_phasecenter(outms)
    run_wsclean(outms, imagename=imagename, automask_thresh=5, uvrange='0', predict=False, imsize=1024, cell='1arcmin')
    correct_primary_beam(outms, imagename + "-image.fits")
    logging.info('Imaging completed for ' + msname)
