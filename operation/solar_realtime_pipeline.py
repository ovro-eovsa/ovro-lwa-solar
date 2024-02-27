#!/usr/bin/env python
# Script for calibrating and imaging the OVRO-LWA slow visibility data on 2023 May 10
# Author: Bin Chen (bin.chen@njit.edu) on 2023 August 21

import os, sys, glob, getopt
#sys.path.append('/opt/devel/bin.chen/ovro-lwa-solar')
from ovrolwasolar import solar_pipeline as sp
from ovrolwasolar.primary_beam import analytic_beam as beam
from ovrolwasolar import utils, calibration, flagging
from casatasks import clearcal, applycal, flagdata, tclean, exportfits, imsubimage
from casatools import msmetadata, quanta, measures, table
from suncasa.utils import helioimage2fits as hf
from suncasa.io import ndfits
from ovrolwasolar import file_handler
import logging
import timeit
import multiprocessing
from astropy.time import Time, TimeDelta
import astropy.units as u
from sunpy.coordinates import frames
from astropy.coordinates import SkyCoord, EarthLocation, get_body, AltAz
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
from suncasa.utils import plot_mapX as pmX
import sunpy.map as smap
import shlex, subprocess
from functools import partial
from time import sleep
import socket
from matplotlib.patches import Ellipse
import argparse
matplotlib.use('agg')

msmd = msmetadata()
qa = quanta()
me = measures()
tb = table()

def sun_riseset(date=Time.now(), observatory='ovro'):
    '''
    Given a date in Time object, determine the sun rise and set time as viewed from OVRO
    '''
    try:
        date_mjd = Time(date).mjd
    except Exception as e:
        logging.error(e)

    obs = EarthLocation.of_site(observatory)
    t0 = Time(int(date_mjd) + 13. / 24., format='mjd')
    sun_loc = get_body('sun', t0, location=obs)
    alt = sun_loc.transform_to(AltAz(obstime=t0, location=obs)).alt.degree
    while alt < 10.:
        t0 += TimeDelta(60., format='sec')
        alt = sun_loc.transform_to(AltAz(obstime=t0, location=obs)).alt.degree

    t1 = Time(int(date_mjd) + 22. / 24., format='mjd')
    sun_loc = get_body('sun', t1, location=obs)
    alt = sun_loc.transform_to(AltAz(obstime=t1, location=obs)).alt.degree
    while alt > 10.:
        t1 += TimeDelta(60., format='sec')
        alt = sun_loc.transform_to(AltAz(obstime=t1, location=obs)).alt.degree

    return t0, t1


#### define data files #####
def list_msfiles_old(intime, server='lwacalim', distributed=True, file_path='slow',
            nodes = [1, 2, 3, 4, 5, 6, 7, 8], time_interval='10s'):
    """
    Return a list of visibilities to be copied for pipeline processing for a given time
    :param intime: astropy Time object
    :param time_interval: Options are '10s', '1min', '10min', '1hr', '1day'
    :param nband_min: minimum number of available subbands acceptable
    """
    intimestr = intime.isot[:-4].replace('-','').replace(':','').replace('T','_')
    if time_interval == '10s':
        tstr = intimestr[:-1]
    if time_interval == '1min':
        tstr = intimestr[:-2]
    if time_interval == '10min':
        tstr = intimestr[:-3]
    if time_interval == '1hr':
        tstr = intimestr[:-4]
    if time_interval == '1day':
        tstr = intimestr[:9]

    msfiles = []
    if not distributed:
        args = ['ssh', '{}'.format(server), 'ls', '{}'.format(file_path), '|', 'grep', '{}'.format(tstr)]
        p = subprocess.run(args, capture_output=True)
        filenames = p.stdout.decode('utf-8').split('\n')[:-1]
        for filename in filenames:
            if filename[-6:] == 'MHz.ms':
                pathstr = '{0:s}:{1:s}/{2:s}'.format(server, file_path, filename)
                tmpstr = filename[:15].replace('_', 'T')
                timestr = tmpstr[:4] + '-' + tmpstr[4:6] + '-' + tmpstr[6:11] + ':' + tmpstr[11:13] + ':' + tmpstr[13:]
                freqstr = filename[16:21]
                msfiles.append({'path': pathstr, 'name': filename, 'time': timestr, 'freq': freqstr})
    else:
        processes=[]
        for i in nodes:
            args = ['ssh', '{0:s}0{1:d}'.format(server, i), 'ls', '/data0{0:d}/{1:s}'.format(i, file_path), '|', 'grep', '{}'.format(tstr)]
            p = subprocess.Popen(args, stdout=subprocess.PIPE)
            processes.append(p)
            filenames = p.communicate()[0].decode('utf-8').split('\n')[:-1]
            for filename in filenames:
                if filename[-6:] == 'MHz.ms':
                    pathstr = '{0:s}0{1:d}:/data0{2:d}/{3:s}/{4:s}'.format(server, i, i, file_path, filename)
                    tmpstr = filename[:15].replace('_', 'T')
                    timestr = tmpstr[:4] + '-' + tmpstr[4:6] + '-' + tmpstr[6:11] + ':' + tmpstr[11:13] + ':' + tmpstr[13:]
                    freqstr = filename[16:21]
                    msfiles.append({'path': pathstr, 'name': filename, 'time': timestr, 'freq': freqstr})
    return msfiles

def list_msfiles(intime, lustre=True, file_path='slow', server=None, time_interval='10s', 
                 bands=['32MHz', '36MHz', '41MHz', '46MHz', '50MHz', '55MHz', '59MHz', '64MHz', '69MHz', '73MHz', '78MHz', '82MHz']):
    """
    Return a list of visibilities to be copied for pipeline processing for a given time
    :param intime: astropy Time object
    :param server: name of the server to list available data
    :param lustre: if True, specific to lustre system on lwacalim nodes. If not, try your luck in combination with file_path
    :param file_path: file path to the data files. For lustre, it is either 'slow' or 'fast'. For other servers, provide full path to data.
    :param time_interval: Options are '10s', '1min', '10min'
    :param bands: bands to list/download. Default to 12 bands above 30 MHz. Full list of available bands is
            ['13MHz', '18MHz', '23MHz', '27MHz', '32MHz', '36MHz', '41MHz', '46MHz', '50MHz', '55MHz', '59MHz', '64MHz', '69MHz', '73MHz', '78MHz', '82MHz']
    """
    intimestr = intime.isot[:-4].replace('-','').replace(':','').replace('T','_')
    datestr = intime.isot[:10]
    hourstr = intime.isot[11:13]
    if time_interval == '10s':
        tstr = intimestr[:-1]
    if time_interval == '1min':
        tstr = intimestr[:-2]
    if time_interval == '10min':
        tstr = intimestr[:-3]

    msfiles = []
    if lustre:
        processes=[]
        for b in bands:
            pathstr = '/lustre/pipeline/{0:s}/{1:s}/{2:s}/{3:s}/'.format(file_path, b, datestr, hourstr)
            if server:
                cmd = 'ssh ' + server + ' ls ' + pathstr + ' | grep ' + tstr
            else:
                cmd = 'ls ' + pathstr + ' | grep ' + tstr
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
            processes.append(p)
            filenames = p.communicate()[0].decode('utf-8').split('\n')[:-1]
            for filename in filenames:
                if filename[-6:] == 'MHz.ms':
                    filestr = pathstr + filename
                    tmpstr = filename[:15].replace('_', 'T')
                    timestr = tmpstr[:4] + '-' + tmpstr[4:6] + '-' + tmpstr[6:11] + ':' + tmpstr[11:13] + ':' + tmpstr[13:]
                    freqstr = filename[16:21]
                    msfiles.append({'path': filestr, 'name': filename, 'time': timestr, 'freq': freqstr})
    else:
        if server:
            cmd = 'ssh ' + server + ' ls ' + file_path + ' | grep ' + tstr
        else:
            cmd = 'ls ' + file_path + ' | grep ' + tstr
        p = subprocess.run(cmd, capture_output=True, shell=True)
        filenames = p.stdout.decode('utf-8').split('\n')[:-1]
        for filename in filenames:
            if filename[-6:] == 'MHz.ms':
                pathstr = '{0:s}:{1:s}/{2:s}'.format(server, file_path, filename)
                tmpstr = filename[:15].replace('_', 'T')
                timestr = tmpstr[:4] + '-' + tmpstr[4:6] + '-' + tmpstr[6:11] + ':' + tmpstr[11:13] + ':' + tmpstr[13:]
                freqstr = filename[16:21]
                msfiles.append({'path': pathstr, 'name': filename, 'time': timestr, 'freq': freqstr})
    return msfiles

def download_msfiles_cmd(msfile_path, server, destination):
    if server:
        p = subprocess.Popen(shlex.split('rsync -az --numeric-ids --info=progress2 --no-perms --no-owner --no-group {0:s}:{1:s} {2:s}'.format(server, msfile_path, destination)))
    else:
        p = subprocess.Popen(shlex.split('rsync -az --numeric-ids --info=progress2 --no-perms --no-owner --no-group {0:s} {1:s}'.format(msfile_path, destination)))
    std_out, std_err = p.communicate()
    if std_err:
        print(std_err)

def download_msfiles(msfiles, destination='/fast/bin.chen/realtime_pipeline/slow_working/', bands=None, verbose=True, server=None, maxthread=6):
    from multiprocessing.pool import ThreadPool
    """
    Parallelized downloading for msfiles returned from list_msfiles() to a destination.
    """
    inmsfiles_path = [f['path'] for f in msfiles]
    inmsfiles_name = [f['name'] for f in msfiles]
    inmsfiles_band = [f['freq'] for f in msfiles]
    omsfiles_path = []
    omsfiles_name = []
    if bands is None:
        omsfiles_path = inmsfiles_path
        omsfiles_name = inmsfiles_name
    else:
        for bd in bands:
            if bd in inmsfiles_band:
                idx = inmsfiles_band.index(bd)
                #omsfiles_server.append(inmsfiles_server[idx])
                omsfiles_path.append(inmsfiles_path[idx])
                omsfiles_name.append(inmsfiles_name[idx])

    nfile = len(omsfiles_path)
    if nfile == 0:
        print('No files to download. Abort...')
        return -1
    time_bg = timeit.default_timer() 
    if verbose:
        print('I am going to download {0:d} files'.format(nfile))

    tp = ThreadPool(maxthread)
    for omsfile_path in omsfiles_path:
        tp.apply_async(download_msfiles_cmd, args=(omsfile_path, server, destination))

    tp.close()
    tp.join()

    time_completed = timeit.default_timer() 
    if verbose:
        print('Downloading {0:d} files took in {1:.1f} s'.format(nfile, time_completed-time_bg))
    omsfiles = [destination + n for n in omsfiles_name]
    return omsfiles


def download_timerange(starttime, endtime, download_interval='1min', destination='/fast/bin.chen/20231027/slow/', 
                server=None, file_path='slow', bands=None, verbose=True, maxthread=5):
    time_bg = timeit.default_timer() 
    t_start = Time(starttime)
    t_end = Time(endtime)
    print('Start time: ', t_start.isot)
    print('End time: ', t_end.isot)
    if not os.path.exists(destination):
        os.makedirs(destination)

    if download_interval == '10s':
        dt = TimeDelta(10., format='sec')
    if download_interval == '1min':
        dt = TimeDelta(60., format='sec')
    if download_interval == '10min':
        dt = TimeDelta(600., format='sec')
    nt = int(np.ceil((t_end - t_start) / dt))
    print('====Will download {0:d} times at an interval of {1:s}===='.format(nt, download_interval))
    for i in range(nt):
        intime = t_start + i * dt
        msfiles = list_msfiles(intime, server=server, file_path=file_path, time_interval='10s', bands=bands)
        if verbose:
            print('Downloading time ', intime.isot)
        download_msfiles(msfiles, destination=destination, bands=bands, verbose=verbose, server=server, maxthread=maxthread)
    time_completed = timeit.default_timer() 
    if verbose:
        print('====Downloading all {0:d} times took {1:.1f} s===='.format(nt, time_completed-time_bg))


def download_calibms(calib_time, download_fold = '/lustre/bin.chen/realtime_pipeline/ms_calib/', doflag=True,
        bands=['13MHz', '18MHz', '23MHz', '27MHz', '32MHz', '36MHz', '41MHz', '46MHz', '50MHz', '55MHz', '59MHz', '64MHz', '69MHz', '73MHz', '78MHz', '82MHz']):
    """
    Function to download calibration ms files for all or selected bands based on a given time
    :param calib_time: time selected for generating the calibration tables in astropy Time format
    :param download_fold: directory to hold the downloaded msfiles 
    :param bands: band selection. Default to use all 16 bands.
    """
    if type(calib_time) == str:
        try:
            calib_time = Time(calib_time)
        except:
            print('The input time needs to be astropy.time.Time format')
    print(socket.gethostname(), '=======Calibrating Time {0:s}======='.format(calib_time.isot))
    ms_calib0 = list_msfiles(calib_time, file_path='slow', bands=bands)
    ms_calib = download_msfiles(ms_calib0, destination=download_fold, bands=bands)
    ms_calib.sort()
    if doflag:
        for ms_calib_ in ms_calib:
           antflagfile = flagging.flag_bad_ants(ms_calib_)
    return ms_calib


def gen_caltables(ms_calib, caltable_fold = '/lustre/bin.chen/realtime_pipeline/caltables/',
        bcaltb=None, uvrange='>10lambda', refant='202', flag_outrigger=True,
        beam_caltable_fold = '/lustre/bin.chen/realtime_pipeline/caltables_beam/'):
    """
    Function to generate calibration tables for a list of calibration ms files
    :param ms_calib: list of ms files used for calibration
    :param caltable_fold: directory to hold these calibration tables
    :param bcaltb: name of the calibration tables. Use the default if None
    :param uvrange: uv range to be used, default to '>10lambda'
    :param refant: reference antenna, default to '202'
    :param flag_outrigger: if True, flag all outrigger antennas. These would be used for beamforming.
    :param beam_caltable_fold: directory to hold the calibration tables for beamforming (after flagging the outriggers).
    """
    import pandas as pd
    bcaltbs = []
    chan_freqs = []
    # TODO: somehow the parallel processing failed if flagging has run. I have no idea why. Returning to the slow serial processing.
    #pool = multiprocessing.pool.Pool(processes=len(ms_calib))
    #gen_calib_partial = partial(calibration.gen_calibration, uvrange=uvrange, caltable_fold=caltable_fold,
    #                refant=refant)
    #result = pool.map_async(gen_calib_partial, ms_calib)
    #timeout = 2000.
    #result.wait(timeout=timeout)
    #bcaltbs = result.get()
    #pool.close()
    #pool.join()
    for ms_calib_ in ms_calib:
        try:
            bcaltb = calibration.gen_calibration(ms_calib_, uvrange=uvrange, caltable_fold=caltable_fold, refant=refant)
            msmd.open(ms_calib_)
            chan_freqs.append(msmd.chanfreqs(0))
            msmd.done()
            bcaltbs.append(bcaltb)
        except Exception as e:
            print('Something is wrong when making calibrations for ', ms_calib_)
            print(e)
        
    chan_freqs = np.concatenate(chan_freqs)

    if flag_outrigger:
        core_ant_ids, exp_ant_ids = flagging.get_antids(ms_calib[0])
        bcaltbs_bm = []
        bmcalfac = []
        for bcaltb in bcaltbs:
            bcaltb_bm = beam_caltable_fold + '/' + os.path.basename(bcaltb)
            os.system('cp -r ' + bcaltb + ' ' + bcaltb_bm)
            tb.open(bcaltb_bm, nomodify=False)
            flags = tb.getcol('FLAG')
            npol, nch, nant = flags.shape
            for exp_ant_id in exp_ant_ids:
                flags[:, :, exp_ant_id] = True
            num_ant_per_chan = nant - np.min(np.sum(flags, axis=2), axis=0)
            bmcalfac_per_chan = num_ant_per_chan ** 2
            bmcalfac.append(bmcalfac_per_chan)
            tb.putcol('FLAG', flags)
            tb.close()
            bcaltbs_bm.append(bcaltb_bm)

        bmcalfac = np.concatenate(bmcalfac)
        # write channel frequencies and corresponding beam scaling factors into a csv file
        df = pd.DataFrame({"chan_freqs":chan_freqs, "beam_calfac":bmcalfac})
        bcalfac_file = beam_caltable_fold + '/' + os.path.basename(bcaltb)[:15] + '_bmcalfac.csv'
        df.to_csv(bcalfac_file, index=False)
        return bcaltbs, bcaltbs_bm, bcalfac_file
    else:
        return bcaltbs



def run_calib(msfile, msfiles_cal=None, bcal_tables=None, do_selfcal=True, num_phase_cal=0, num_apcal=1, caltable_folder=None, logger_file=None, visdir_slfcaled=None, flagdir=None):
    msmd.open(msfile)
    trange = msmd.timerangeforobs(0)
    btime = qa.time(trange['begin']['m0'],form='fits')[0]
    etime = qa.time(trange['end']['m0'],form='fits')[0]
    msmd.close()
    cfreqidx = os.path.basename(msfile).find('MHz') - 2
    cfreq = os.path.basename(msfile)[cfreqidx:cfreqidx+2]+'MHz'
    msfile_cal_ = [m for m in msfiles_cal if cfreq in m]
    bcal_tables_ = [m for m in bcal_tables if cfreq in m]
    #### Generate calibrations ####
    imagename = os.path.basename(msfile)[:-3]+'_sun_selfcal'
    if len(bcal_tables_) > 0:
        bcal_table = bcal_tables_[0]
        print('Found calibration table {0:s}'.format(bcal_table))
        try:
            outms, tmp = sp.image_ms_quick(msfile, calib_ms=None, bcal=bcal_table, do_selfcal=do_selfcal, imagename=imagename, logging_level='info', 
                        num_phase_cal=num_phase_cal, num_apcal=num_apcal,
                        logfile=logger_file, caltable_folder=caltable_folder, do_final_imaging=False, do_fluxscaling=False, freqbin=1)
            os.system('cp -r '+ outms + ' ' + visdir_slfcaled + '/')
            if os.path.exists(msfile.replace('.ms', '.badants')):
                os.system('cp '+ msfile.replace('.ms', '.badants') + ' ' + flagdir + '/')
            msfile_slfcaled = visdir_slfcaled + '/' + os.path.basename(outms)
            return msfile_slfcaled
        except Exception as e:
            logging.error(e)
            return -1
    elif len(msfile_cal_) > 0:
        msfile_cal = msfile_cal_[0]
        try:
            outms, tmp = sp.image_ms_quick(msfile, calib_ms=msfile_cal, do_selfcal=do_selfcal, imagename=imagename, logging_level='info', 
                        num_phase_cal=num_phase_cal, num_apcal=num_apcal,
                        logfile=logger_file, caltable_folder=caltable_folder, do_final_imaging=False, do_fluxscaling=False, freqbin=1)
            os.system('cp -r '+ outms + ' ' + visdir_slfcaled + '/')
            msfile_slfcaled = visdir_slfcaled + '/' + os.path.basename(outms)
            return msfile_slfcaled
        except Exception as e:
            logging.error(e)
            return -1
    else:
        print('No night time ms or caltable available for {0:s}. Skip...'.format(msfile))
        return -1


def run_imager(msfile_slfcaled, imagedir_allch=None, ephem=None, nch_out=12):
    blc = int(512 - 128)
    trc = int(512 + 128 - 1)
    region='box [ [ {0:d}pix , {1:d}pix] , [{2:d}pix, {3:d}pix ] ]'.format(blc, blc, trc, trc)
    try:
        msmd.open(msfile_slfcaled)
        trange = msmd.timerangeforobs(0)
        msmd.close()
        btime = Time(trange['begin']['m0']['value'], format='mjd')
        etime = Time(trange['end']['m0']['value'], format='mjd')
        tref_mjd = (btime.mjd + etime.mjd) / 2. 
        tref = Time(tref_mjd, format='mjd')
        tref_str = btime.isot+'~'+etime.isot
        msinfo = hf.read_msinfo(msfile_slfcaled, verbose=True)
        timeutc = me.epoch('UTC', '%fd' % tref.mjd)
        ovro = me.observatory('OVRO_MMA')
        me.doframe(ovro)
        me.doframe(timeutc)
        d0 = me.direction('SUN')
        d0_j2000 = me.measure(d0, 'J2000')
        azel = me.measure(d0, 'AZEL')
        elev = np.degrees(azel['m1']['value'])
        az = np.degrees(azel['m0']['value'])
        pb = beam(msfile_slfcaled)
        pb.srcjones(az=[az],el=[elev])
        jones_matrices = pb.get_source_pol_factors(pb.jones_matrices[0,:,:])
        sclfactor = 1. / jones_matrices[0][0]
        helio_imagename = imagedir_allch + os.path.basename(msfile_slfcaled).replace('.ms','.sun') 
        default_wscleancmd = ("wsclean -j 4 -mem 2 -no-reorder -no-dirty -size 1024 1024 -scale 1arcmin -weight briggs -0.5 -minuv-l 10 -auto-threshold 3 -name " + 
                helio_imagename + " -niter 10000 -mgain 0.8 -beam-fitting-size 1 -pol I -join-channels -channels-out " + str(nch_out) + ' ' + msfile_slfcaled)
 
        os.system(default_wscleancmd)

        outfits = glob.glob(helio_imagename + '*-image.fits')
        outfits.sort()
        outfits_helio = hf.imreg(msfile_slfcaled, outfits, ephem=ephem, msinfo=msinfo, timerange=[tref_str] * len(outfits), 
                usephacenter=True, verbose=True, toTb=True, subregion=region, sclfactor=sclfactor)
        return outfits_helio
    except Exception as e:
        logging.error(e)
        #logging.error('{0:s}: Imaging for {1:s} failed'.format(socket.gethostname(), msfile_slfcaled))
        return -1

def pipeline_quick(image_time=Time.now() - TimeDelta(20., format='sec'), server='lwacalim', file_path='slow', 
            distributed=True, min_nband=6, nch_out=12, do_selfcal=True, num_phase_cal=0, num_apcal=1, overwrite_ms=False, delete_ms_slfcaled=False,
            logger_file=None, compress_fits=True,
            proc_dir = '/fast/bin.chen/realtime_pipeline/',
            save_dir = '/lustre/bin.chen/realtime_pipeline/',
            calib_file = '20240117_145752',
            delete_working_ms=True):
    """
    Pipeline for processing and imaging slow visibility data
    :param time_start: start time of the visibility data to be processed
    :param time_end: end time of the visibility data to be processed
    :param image_interval: time interval between adjacent visibilities to be processed
    :param server: server name on which the data is stored
    :param file_path: path to the data w.r.t. the server
    :param distributed: if true, assume the data is on distributed lwacalim nodes
    :param min_nband: minimum number of bands to be processed. Will skip if less than that.
    :param calib_file: calibration file to be used. Format yyyymmdd_hhmmss
    :param nch_out: number of channels to be imaged
    :param do_selfcal: if True, do selfcalibration
    :param num_phase_cal: number of phase calibration iterations
    :param num_apcal: number of amplitude calibration iterations
    :param overwrite_ms: if True, overwrite the ms files in the working directory
    :param delete_ms_slfcaled: if True, delete the selfcalibrated ms files after imaging
    :param logger_file: name of the log file
    :param compress_fits: if True, compress the fits (not loseless, reduce bitlen, but it's OK to do this in most of the cases of solar imaging) files after imaging
    :param proc_dir: directory to hold the working files
    :param save_dir: directory to hold the final products
    :param calib_file: calibration file to be used. Format yyyymmdd_hhmmss
    :param delete_working_ms: if True, delete the working ms files after imaging (set False for debugging purpose)
    """

    time_begin = timeit.default_timer() 
    bands = ['32MHz', '36MHz', '41MHz', '46MHz', '50MHz', '55MHz', '59MHz', '64MHz', '69MHz', '73MHz', '78MHz', '82MHz']

    visdir_calib = proc_dir + 'slow_calib/'
    caltable_folder = proc_dir + 'caltables/'
    visdir_work = proc_dir + 'slow_working/'
    visdir_slfcaled = proc_dir + 'slow_slfcaled/'
    imagedir_allch = proc_dir + 'images_allch/'
    flagdir = save_dir + 'flags/'

    imagedir_allch_combined = save_dir + 'fits/'
    fig_mfs_dir = save_dir + 'figs_mfs/'

    ## Night-time MS files used for calibration ##
    msfiles_cal = glob.glob(visdir_calib + calib_file + '_*MHz.ms')
    msfiles_cal.sort()

    bcal_tables = glob.glob(caltable_folder + calib_file + '_*MHz.bcal')
    bcal_tables.sort()

    if not os.path.exists(visdir_work):
        os.makedirs(visdir_work)

    if not os.path.exists(visdir_slfcaled):
        os.makedirs(visdir_slfcaled)

    if not os.path.exists(caltable_folder):
        os.makedirs(caltable_folder)

    if not os.path.exists(flagdir):
        os.makedirs(flagdir)

    if not os.path.exists(imagedir_allch):
        os.makedirs(imagedir_allch)

    if not os.path.exists(imagedir_allch_combined):
        os.makedirs(imagedir_allch_combined)

    if not os.path.exists(fig_mfs_dir):
        os.makedirs(fig_mfs_dir)

    try:
        print(socket.gethostname(), '=======Processing Time {0:s}======='.format(image_time.isot))
        #logging.info('=======Processing Time {0:s}======='.format(image_time.isot))
        msfiles0 = list_msfiles(image_time, file_path=file_path)
        if len(msfiles0) < min_nband:
            print('This time only has {0:d} subbands. Check nearby +-10s time.'.format(len(msfiles0)))
            image_time_before = image_time - TimeDelta(10., format='sec')
            msfiles0_before = list_msfiles(image_time_before, file_path=file_path)
            image_time_after = image_time + TimeDelta(10., format='sec')
            msfiles0_after = list_msfiles(image_time_after, file_path=file_path)
            if len(msfiles0_before) < min_nband and len(msfiles0_after) < min_nband:
                print('I cannot find a nearby time with at least {0:d} available subbands. Abort and wait for next time interval.'.format(min_nband))
                return False
            else:
                if len(msfiles0_before) > len(msfiles0_after):
                    msfiles0 = msfiles0_before
                    image_time = image_time_before
                else:
                    msfiles0 = msfiles0_after
                    image_time = image_time_after
        
        msfiles0_freq = [f['freq'] for f in msfiles0]
        msfiles0_name = [f['name'] for f in msfiles0]
        timestr = msfiles0_name[0][:15]
        msfiles_slfcaled = glob.glob(visdir_slfcaled + '/' + timestr + '_*MHz*.ms')
        msfiles_slfcaled.sort()
        if len(msfiles_slfcaled) == 0 or overwrite_ms:
            #msfiles0 = glob.glob(datadir_orig + timestr + '_*MHz.ms')
            #msfiles0.sort()
            # skip the first two bands (18-32 MHz)
            # msfiles0 = msfiles0[2:]

            #### copy files over to the working directory ####
            print('==Copying file over to working directory==')
            logging.debug('====Copying file over to working directory====')
            time1 = timeit.default_timer()
            #msfiles = []
            #for msfile0 in msfiles0:
            #    os.system('cp -r '+ msfile0 + ' ' + visdir_work + '/')
            #    msfiles.append(visdir_work + '/' + os.path.basename(msfile0))
            msfiles = download_msfiles(msfiles0, destination=visdir_work, bands=bands)
            time2 = timeit.default_timer()
            logging.debug('Time taken to copy files is {0:.1f} s'.format(time2-time1))

            fitsfiles=[]
            msfiles_slfcaled = []

            # parallelized calibration, selfcalibration, and source subtraction
            logging.debug('Starting to calibrate all {0:d} bands'.format(len(msfiles)))
            time_cal1 = timeit.default_timer()
            pool = multiprocessing.pool.Pool(processes=len(msfiles))
            #result = pool.map_async(run_calib, msfiles)
            run_calib_partial = partial(run_calib, msfiles_cal=msfiles_cal, bcal_tables=bcal_tables, do_selfcal=do_selfcal, num_phase_cal=num_phase_cal, num_apcal=num_apcal, 
                    logger_file=logger_file, caltable_folder=caltable_folder, visdir_slfcaled=visdir_slfcaled, flagdir=flagdir)
            result = pool.map_async(run_calib_partial, msfiles)
            timeout = 2000.
            result.wait(timeout=timeout)
            if result.ready():
                time_cal2 = timeit.default_timer()
                logging.debug('Calibration for all {0:d} bands is done in {1:.1f} s'.format(len(msfiles), time_cal2-time_cal1))
            else:
                logging.debug('Calibration for certain bands is incomplete in {0:.1f} s'.format(timeout))
                logging.debug('Proceed anyway')
                
            msfiles_slfcaled = result.get()
            pool.close()
            pool.join()
            if delete_working_ms:
                os.system('rm -rf '+ visdir_work + '/' + timestr + '*')
        else:
            logging.debug('=====Selfcalibrated ms already exist for {0:s}. Proceed with imaging.========'.format(timestr)) 
            if delete_working_ms:
                os.system('rm -rf '+ visdir_work + '/' + timestr + '*')


        # Do imaging
        print('======= processed selfcaled ms files =====')
        success = [type(m) is str for m in msfiles_slfcaled]
        msfiles_slfcaled_success = []
        for m in msfiles_slfcaled:
            if type(m) is str:
                msfiles_slfcaled_success.append(m)
        if sum(success) > 4:
            logging.info('{0:s}: Successfuly selfcalibrated {1:d} out of {2:d} bands'.format(socket.gethostname(), len(msfiles_slfcaled_success), len(bands)))
            time_img1 = timeit.default_timer()
            for i, m in enumerate(msfiles_slfcaled_success):
                try:
                    msmd.open(m)
                    trange = msmd.timerangeforobs(0)
                    msmd.close()
                    break
                except Exception as e:
                    if i < len(msfiles_slfcaled_success): 
                        logging.error('Reading file {0:s} has error {1:s}. Will try the next one'.format(m, e))
                        continue
                    else:
                        logging.error('Nothing seems to work. I will abort and continue to the next time')
                        os.system('rm -rf '+ visdir_slfcaled + '/' + timestr + '_*MHz*.ms')
                        os.system('rm -rf '+ caltable_folder + '/' + timestr + '_*MHz*')
                        return False

            btime = Time(trange['begin']['m0']['value'], format='mjd')
            etime = Time(trange['end']['m0']['value'], format='mjd')
            tref_mjd = (btime.mjd + etime.mjd) / 2. 
            tref = Time(tref_mjd, format='mjd')
            ephem = hf.read_horizons(tref, dur=1./60./24., observatory='OVRO_MMA')
            pool = multiprocessing.pool.Pool(processes=len(msfiles_slfcaled_success))
            run_imager_partial = partial(run_imager, imagedir_allch=imagedir_allch, ephem=ephem, nch_out=nch_out)
            result = pool.map_async(run_imager_partial, msfiles_slfcaled_success)
            timeout = 200.
            result.wait(timeout=timeout)
            if result.ready():
                time_img2 = timeit.default_timer()
                logging.debug('Imaging for all {0:d} bands is done in {1:.1f} s'.format(len(msfiles_slfcaled_success), time_img2-time_img1))
            else:
                logging.debug('Imaging for certain bands is incomplete in {0:.1f} s'.format(timeout))
                logging.debug('Proceed anyway')

            fitsfiles = result.get()
            pool.close()
            pool.join()
        else:
            logging.error('For time {0:s}, less than 4 bands out of {1:d} bands were calibrated successfully. Abort....'.format(timestr, len(bands)))
            os.system('rm -rf '+ visdir_slfcaled + '/' + timestr + '_*MHz*.ms')
            os.system('rm -rf '+ caltable_folder + '/' + timestr + '_*MHz*')
            return False

        if delete_ms_slfcaled:
            os.system('rm -rf '+ visdir_slfcaled + '/' + timestr + '_*MHz*.ms')
            os.system('rm -rf '+ caltable_folder + '/' + timestr + '_*MHz*')

        if 'fitsfiles' in locals() and len(fitsfiles) > 1:
            ## define subdirectories for storing the fits and png files
            datedir = btime.isot[:10].replace('-','/')+'/'
            imagedir_allch_combined_sub = imagedir_allch_combined + '/' + datedir
            fig_mfs_dir_sub = fig_mfs_dir + '/' + datedir
            if not os.path.exists(imagedir_allch_combined_sub):
               os.makedirs(imagedir_allch_combined_sub)
            if not os.path.exists(fig_mfs_dir_sub):
                os.makedirs(fig_mfs_dir_sub)

            ## Wrap images
            timestr_iso = btime.isot[:-4].replace(':','')+'Z'
            # multi-frequency synthesis images
            fits_mfs = imagedir_allch_combined_sub + '/ovro-lwa.lev1_mfs_10s.' + timestr_iso + '.image.fits' 
            #fitsfiles_mfs = glob.glob(imagedir_allch + '/' + timestr+ '*MFS-image.fits')
            fitsfiles_mfs = []
            for f in fitsfiles:
                if type(f) is list:
                    if 'MFS' in f[-1]:
                        fitsfiles_mfs.append(f[-1])
                else:
                    continue
            fitsfiles_mfs.sort()
            #ndfits.wrap(fitsfiles_mfs, outfitsfile=fits_mfs, docompress=compress_fits)
            ndfits.wrap(fitsfiles_mfs, outfitsfile=fits_mfs)
            # fine channel spectral images
            fits_fch = imagedir_allch_combined_sub + '/ovro-lwa.lev1_fch_10s.' + timestr_iso + '.image.fits' 
            #fitsfiles_fch = list(set(glob.glob(imagedir_allch + '/' + timestr + '*-image.fits'))-set(glob.glob(imagedir_allch + '/' + timestr + '*MFS-image.fits')))
            fitsfiles_fch = []
            for f in fitsfiles:
                if type(f) is list:
                    fitsfiles_fch += f[:-1]
                else:
                    continue
            fitsfiles_fch.sort()
            #ndfits.wrap(fitsfiles_fch, outfitsfile=fits_fch, docompress=compress_fits)
            ndfits.wrap(fitsfiles_fch, outfitsfile=fits_fch)
            os.system('rm -rf '+imagedir_allch + '*')

            # Plot mfs images (1 image per subband)
            fig = plt.figure(figsize=(15., 8.))
            fov = 8000
            gs = gridspec.GridSpec(3, 4, left=0.05, right=0.95, top=0.94, bottom=0.10, wspace=0.3, hspace=0.4)
            meta, rdata = ndfits.read(fits_mfs)
            freqs_mhz = meta['ref_cfreqs']/1e6
            freqs_plt = [34.1, 38.7, 43.2, 47.8, 52.4, 57.0, 61.6, 66.2, 70.8, 75.4, 80.0, 84.5]
            for i in range(12):
                ax = fig.add_subplot(gs[i])
                freq_plt = freqs_plt[i]
                if np.min(np.abs(freqs_mhz - freq_plt)) < 2.:
                    bd = np.argmin(np.abs(freqs_mhz - freq_plt)) 
                    rmap_plt_ = smap.Map(np.squeeze(rdata[0, bd, :, :]/1e6), meta['header'])
                    rmap_plt = pmX.Sunmap(rmap_plt_)
                    im = rmap_plt.imshow(axes=ax, cmap='hinodexrt')
                    rmap_plt.draw_limb(ls='-', color='w', alpha=0.5)

                    bmaj,bmin,bpa = meta['cbmaj'][bd],meta['cbmin'][bd],meta['cbpa'][bd]
                    beam0 = Ellipse((-fov/2*0.75, -fov/2*0.75), bmaj*3600,
                            #bmin*3600, angle=(-bpa),  fc='None', lw=2, ec='w')
                            bmin*3600, angle=-(90.-bpa),  fc='None', lw=2, ec='w')

                    ax.add_artist(beam0)

                    cbar = plt.colorbar(im)
                    cbar.set_label(r'$T_B$ (MK)')
                    freq_mhz = meta['ref_cfreqs'][bd]/1e6
                    #ax.set_title('OVRO-LWA {0:.0f} MHz'.format(freq_mhz))
                    ax.text(0.02, 0.98, '{0:.0f} MHz'.format(freq_mhz), color='w', ha='left', va='top', fontsize=12, transform=ax.transAxes)
                else:
                    coord = SkyCoord(0*u.arcsec, 0*u.arcsec, obstime=tref.isot, observer='earth', frame=frames.Helioprojective)
                    header = smap.make_fitswcs_header(np.zeros((256, 256)), coord,
                                           reference_pixel=[128, 128]*u.pixel,
                                           scale=[60, 60]*u.arcsec/u.pixel,
                                           telescope='OVRO-LWA', instrument='',
                                           wavelength=freq_plt*1e6*u.Hz)
                    empty_map_ = smap.Map(np.zeros((256, 256)), header)
                    empty_map = pmX.Sunmap(empty_map_)
                    im = empty_map.imshow(axes=ax, cmap='hinodexrt', vmin=0, vmax=1.)
                    empty_map.draw_limb(ls='-', color='w', alpha=0.5)
                    cbar = plt.colorbar(im)
                    cbar.set_label(r'$T_B$ (MK)')
                    ax.text(0.02, 0.98, '{0:.0f} MHz (no data)'.format(freq_plt), color='w', ha='left', va='top', fontsize=12, transform=ax.transAxes)
                ax.set_xlim([-fov/2, fov/2])
                ax.set_ylim([-fov/2, fov/2])
            fig.suptitle('OVRO-LWA Images at ' + tref.isot[:-4], fontsize=15)
            text1 = fig.text(0.01, 0.01, 'OVRO-LWA Solar Team (NJIT Solar Radio Group)', fontsize=12, ha='left', va='bottom')
            text2 = fig.text(0.99, 0.01, 'OVRO Long Wavelength Array (Caltech)', fontsize=12, ha='right', va='bottom')
            fig.savefig(fig_mfs_dir_sub + '/' + os.path.basename(fits_mfs).replace('.image.fits', '.png'))
            plt.close()
            time_completed= timeit.default_timer() 
            logging.debug('====All processing for time {0:s} is done in {1:.1f} minutes'.format(timestr, (time_completed-time_begin)/60.))
            return True
        else:
            time_exit = timeit.default_timer()
            logging.error('====Processing for time {0:s} failed in {1:.1f} minutes'.format(timestr, (time_exit-time_begin)/60.))
            return False
    except Exception as e:
        logging.error(e)
        time_exit = timeit.default_timer()
        logging.error('====Processing for time {0:s} failed in {1:.1f} minutes'.format(timestr, (time_exit-time_begin)/60.))
        return False



def run_pipeline(time_start=Time.now(), time_end=None, time_interval=600., delay_from_now=180., do_selfcal=True, num_phase_cal=0, num_apcal=1, 
        server='lwacalim', file_path='slow', multinode=True, nodes=10, firstnode=0, delete_ms_slfcaled=True, 
        logger_file='/fast/bin.chen/realtime_pipeline/realtime_calib-imaging_parallel.log',
        proc_dir = '/fast/bin.chen/realtime_pipeline/',
        save_dir = '/lustre/bin.chen/realtime_pipeline/',
        calib_file = '20240117_145752', delete_working_ms=True):
    '''
    Main routine to run the pipeline. Note each time stamp takes about 8.5 minutes to complete.
    "time_interval" needs to be set to something greater than that. 600 is recommended.
    :param time_start: time for starting the pipeline. astropy.time.Time object.
    :param time_end: time for ending the pipeline. astropy.time.Time object. 
                If not specified (default), it will never end until being killed manually.
    :param time_interval: interval between adjacent processing times in seconds for each session
    :param delay_from_now: delay of the newest time to process compared to now.
    :param delete_ms_slfcaled: whether or not to delete the self-calibrated measurement sets.
    :param multinode: if True, will delay the start time by the node
    :param nodes: number of nodes to be used. Default 10 (lwacalim[00-09])
    :param firstnodes: first node to be used. Default 0 (lwacalim00)
    :param calib_file: calibration file to be used. Format yyyymmdd_hhmmss
    '''
    logging.basicConfig(filename=logger_file, filemode='at',
        format='%(asctime)s %(funcName)s %(lineno)d %(levelname)-8s %(message)s',
        level=20,
        datefmt='%Y-%m-%d %H:%M:%S', force=True)
    try:
        time_start = Time(time_start)
    except Exception as e:
        logging.error(e)
        raise e
    logging.info('{0:s}: I am asked to start imaging for {1:s}'.format(socket.gethostname(), time_start.isot))
    if multinode:
        nodenum = int(socket.gethostname()[-2:])
        delay_by_node = (nodenum - firstnode) * (time_interval/nodes)
    else:
        delay_by_node = 0. 
    #while time_start > t_rise and time_start < Time.now() - TimeDelta(15.,format='sec'): 
    # find out when the Sun is high enough in the sky
    (t_rise, t_set) = sun_riseset(time_start)
    if time_start < t_rise:
        twait = t_rise - time_start
        logging.info('{0:s}: Start time {1:s} is before sunrise. Wait for {2:.1f} hours to start.'.format(socket.gethostname(), time_start.isot, twait.value * 24.))
        time_start += TimeDelta(twait.sec + 60., format='sec')
        sleep(twait.sec + 60.)
    else:
        logging.info("{0:s}: Start time {1:s} is after today's sunrise at {2:s}. Will try to proceed.".format(socket.gethostname(), time_start.isot, t_rise.isot))
    time_start += TimeDelta(delay_by_node, format='sec')
    logging.info('{0:s}: Delay {1:.1f} min to {2:s}'.format(socket.gethostname(), delay_by_node / 60., time_start.isot))
    sleep(delay_by_node)
    while True:
        if time_end:
            if time_start > Time(time_end):
                logging.info('The new imaging time now passes the provided end time. Ending the pipeline.'.format(Time(time_start).isot, Time(time_end).isot))
                print('The new imaging time now passes the provided end time. Ending the pipeline.'.format(Time(time_start).isot, Time(time_end).isot))
                break
        time1 = timeit.default_timer()
        if time_start > Time.now() - TimeDelta(delay_from_now, format='sec'):
            twait = time_start - Time.now()
            logging.info('{0:s}: Start time {1:s} is too close to current time. Wait {2:.1f} m to start.'.format(socket.gethostname(), time_start.isot, (twait.sec + delay_from_now) / 60.))
            sleep(twait.sec + delay_from_now)
        logging.info('{0:s}: Start processing {1:s}'.format(socket.gethostname(), time_start.isot))
        res = pipeline_quick(time_start, do_selfcal=do_selfcal, num_phase_cal=num_phase_cal, num_apcal=num_apcal, server=server, file_path=file_path, 
                delete_ms_slfcaled=delete_ms_slfcaled, logger_file=logger_file, proc_dir=proc_dir, save_dir=save_dir, calib_file=calib_file, 
                delete_working_ms=delete_working_ms)
        time2 = timeit.default_timer()
        if res:
            logging.info('{0:s}: Processing {1:s} was successful within {2:.1f}m'.format(socket.gethostname(), time_start.isot, (time2-time1)/60.))
        else:
            logging.info('{0:s}: Processing {1:s} was unsuccessful!!!'.format(socket.gethostname(), time_start.isot))

        if (time_interval - (time2-time1)) < 0:
            logging.info('{0:s}: Warning!! Processing {1:s} took {2:.1f}m to complete. This node may be falling behind'.format(socket.gethostname(), time_start.isot, (time2-time1)/60.))

        time_start += TimeDelta(time_interval, format='sec')

        if time_start > t_set:
            (t_rise_next, t_set_next) = sun_riseset(t_set + TimeDelta(6./24., format='jd'))
            twait = t_rise_next - time_start
            logging.info('{0:s}: Sun is setting. Done for the day. Wait for {1:.1f} hours to start.'.format(socket.gethostname(), twait.value * 24.)) 
            time_start += TimeDelta(twait.sec + 60. + delay_by_node, format='sec')
            t_rise = t_rise_next
            t_set = t_set_next
            sleep(twait.sec + 60. + delay_by_node)


if __name__=='__main__':
    """
    Main routine of running the realtime pipeline. Example call
        pdsh -w lwacalim[00-09] 'conda activate suncasa && cd /fast/bin.chen/ && python /opt/devel/bin.chen/ovro-lwa-solar/solar_realtime_pipeline.py 2023-11-21T15:50'
    Sometimes afer killing the pipeline (with ctrl c), one need to remove the temporary files and kill all the processes before restarting.
        pdsh -w lwacalim[00-09] 'rm -rf /fast/bin.chen/realtime_pipeline/slow_working/*'
        pdsh -w lwacalim[00-09] 'rm -rf /fast/bin.chen/realtime_pipeline/slow_slfcaled/*'
        pdsh -w lwacalim[00-09] 'pkill -u bin.chen -f wsclean'
        pdsh -w lwacalim[00-09] 'pkill -u bin.chen -f python'
    """
    parser = argparse.ArgumentParser(description='Solar realtime pipeline')
    parser.add_argument('prefix', type=str, help='Timestamp for the start time. Format YYYY-MM-DDTHH:MM')
    parser.add_argument('--end_time', default=None, help='End time in format YYYY-MM-DDTHH:MM')
    parser.add_argument('--interval', default=600., help='Time interval in seconds')
    parser.add_argument('--nodes', default=10, help='Number of nodes to use')
    parser.add_argument('--delay', default=60, help='Delay from current time in seconds')
    parser.add_argument('--proc_dir', default='/fast/bin.chen/realtime_pipeline/', help='Directory for processing')
    parser.add_argument('--save_dir', default='/lustre/bin.chen/realtime_pipeline/', help='Directory for saving fits files')
    parser.add_argument('--calib_file', default='20240117_145752', help='Calibration file to be used yyyymmdd_hhmmss')
    parser.add_argument('--logger_file', default='/fast/bin.chen/realtime_pipeline/realtime_calib-imaging_parallel.log', help='Directory for saving fits files')
                        
    args = parser.parse_args()
    try:
        run_pipeline(args.prefix, time_end=Time(args.end_time), time_interval=float(args.interval), nodes=int(args.nodes), delay_from_now=float(args.delay),
                     proc_dir=args.proc_dir, save_dir=args.save_dir, calib_file=args.calib_file, logger_file=args.logger_file)
    except Exception as e:
        logging.error(e)
        raise e



