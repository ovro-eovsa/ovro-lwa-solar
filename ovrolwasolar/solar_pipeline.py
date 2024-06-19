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
import astropy.units as u
from astropy.io import fits
from . import utils,flagging,calibration,selfcal,source_subtraction,deconvolve,flux_scaling
import logging, glob
from .file_handler import File_Handler
import timeit
from line_profiler import profile

tb = table()
me = measures()
cl = componentlist()
msmd = msmetadata()


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


def change_phasecenter(msfile):
    """
    Change the phasecenter of the measurement set to the position of the Sun.
    """
    m = utils.get_sun_pos(msfile, str_output=False)
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
    logging.debug("Changing the phasecenter to " + ra1 + " " + dec1)
    os.system("chgcentre " + msfile + " " + ra1 + " " + dec1)


@profile
def image_ms(solar_ms, calib_ms=None, bcal=None, do_selfcal=True, imagename='sun_only',
             imsize=1024, cell='1arcmin', logfile='analysis.log', logging_level='info',
             caltable_folder='caltables', full_di_selfcal_rounds=[3,2], partial_di_selfcal_rounds=[0, 1],
             full_dd_selfcal_rounds=[1, 1], partial_dd_selfcal_rounds=[0, 1], do_final_imaging=True, pol='I', 
             solint_full_DI_selfcal=14400, solint_partial_DI_selfcal=3600, solint_full_DD_selfcal=1800, solint_partial_DD_selfcal=600,
             fast_vis=False, fast_vis_image_model_subtraction=False, delete=True,
             refant='202', overwrite=False, do_fluxscaling=False, apply_primary_beam=True, 
             delete_allsky=True, sky_image=None):

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
    :param solint_full_DI_selfcal: Time after which a full direction independent selfcal will be done
    :param solint_partial_DI_selfcal: Time after which a partial direction independent selfcal will
            be done
    :param solint_full_DD_selfcal: same as DI_selfcal but for direction dependent one
    :param fast_vis: Do special analysis for fast visibility imaging   
    :param fast_vis_image_model_subtraction: If False, the strong source model will be subtracted.
            Otherwise WSClean will be run using the full MS and the average sky model will be subtracted.    
    """
    
    if logging_level.lower() == 'info':
        logging.basicConfig(filename=logfile,
            format='%(asctime)s %(levelname)-8s %(message)s',
            filemode='w', level=logging.INFO,
            datefmt='%Y-%m-%d %H:%M:%S')

    if logging_level.lower() == 'debug':
        logging.basicConfig(filename=logfile,
            format='%(asctime)s %(levelname)-8s %(message)s',
            filemode='w', level=logging.DEBUG,
            datefmt='%Y-%m-%d %H:%M:%S')
    
        
    if not os.path.isdir(caltable_folder):
        os.mkdir(caltable_folder)
    if os.path.isfile(imagename + "-image.fits"):
        if not overwrite:
            return None, imagename + "-image.helio.fits"
    
    if fast_vis:
        utils.make_wsclean_compatible(solar_ms)
        utils.swap_fastms_pols(solar_ms)
        utils.correct_fastms_amplitude_scale(solar_ms)

    logging.info('==========Working on a new solar ms file {0:s}============='.format(solar_ms))
    time_begin=timeit.default_timer()
    time1=timeit.default_timer()
    solar_ms = calibration.do_bandpass_correction(solar_ms, calib_ms=calib_ms, bcal=bcal, caltable_folder=caltable_folder, fast_vis=fast_vis)
    time2=timeit.default_timer()
    logging.info('Time taken to do the bandpass correction is: {0:.1f} s'.format(time2-time1))
    time1=time2
    logging.info('Analysing ' + solar_ms)
    if do_selfcal:
        outms_di = selfcal.DI_selfcal(solar_ms, logging_level=logging_level, full_di_selfcal_rounds=full_di_selfcal_rounds,
                              partial_di_selfcal_rounds=partial_di_selfcal_rounds, pol=pol, refant=refant, do_fluxscaling=do_fluxscaling,
                              solint_full_selfcal=solint_full_DI_selfcal, solint_partial_selfcal=solint_partial_DI_selfcal, 
                              fast_vis=fast_vis,calib_ms=calib_ms)

        time2=timeit.default_timer()
        logging.info('Time taken for DI selfcal and fluxscaling is: {0:.1f} s'.format(time2-time1))
        time1=time2
        print (outms_di)
        logging.info('Removing the strong sources in the sky')
        outms_di_ = source_subtraction.remove_nonsolar_sources(outms_di,pol=pol, fast_vis=fast_vis,\
                                        fast_vis_image_model_subtraction=fast_vis_image_model_subtraction,\
                                        delete_allsky=delete_allsky, skyimage=sky_image)
        time2=timeit.default_timer()
        logging.info('Time taken for strong source removal is: {0:.1f} s'.format(time2-time1)) 
        time1=time2
        logging.info('The strong source subtracted MS is ' + outms_di_)
        logging.info('Starting to do Stokes I selfcal towards direction of sun')
        
        if not fast_vis:
            outms_dd = selfcal.DD_selfcal(outms_di_, logging_level=logging_level, full_dd_selfcal_rounds=full_dd_selfcal_rounds,
                                  partial_dd_selfcal_rounds=partial_dd_selfcal_rounds, pol=pol, refant=refant, 
                                  solint_full_selfcal=solint_full_DD_selfcal, solint_partial_selfcal=solint_partial_DD_selfcal)
            time2=timeit.default_timer()
            logging.info('Time taken for DD selfcal is: {0:.1f} s'.format(time2-time1))
            time1=time2
            logging.info('Removing almost all sources in the sky except Sun')
            print ('Removing almost all sources in the sky except Sun')
            outms = source_subtraction.remove_nonsolar_sources(outms_dd, remove_strong_sources_only=False, pol=pol)
            time2=timeit.default_timer()
            logging.info('Time taken for weak source removal is: {0:.1f} s'.format(time2-time1)) 
            time1=time2
            logging.info('The source subtracted MS is ' + outms)
        else:
            outms = selfcal.DD_selfcal(outms_di_, logging_level=logging_level, full_dd_selfcal_rounds=full_dd_selfcal_rounds,
                                  partial_dd_selfcal_rounds=partial_dd_selfcal_rounds, pol=pol, refant=refant, 
                                  solint_full_selfcal=solint_full_DD_selfcal, solint_partial_selfcal=solint_partial_DD_selfcal,
                                  fast_vis=fast_vis, calib_ms=calib_ms)
    else:
        logging.info('Removing almost all sources in the sky except Sun')
        outms = source_subtraction.remove_nonsolar_sources(solar_ms,pol=pol)
        logging.info('The source subtracted MS is ' + outms)

    logging.info('Changing the phasecenter to position of Sun')
    change_phasecenter(outms)
    if do_final_imaging:
        time1=timeit.default_timer()
        logging.info('Generating final solar centered image')
        if not fast_vis:
            deconvolve.run_wsclean(outms, imagename=imagename, auto_mask=5, minuv_l='0', predict=False, 
                                   size=imsize, scale=cell, pol=pol, fast_vis=fast_vis)
        else:
            num_fields=utils.get_total_fields(outms)
            deconvolve.run_wsclean(outms, imagename=imagename, auto_mask=5, minuv_l='0', predict=False,
                                   size=imsize , scale=cell, pol=pol, fast_vis=fast_vis, 
                                   field=','.join([str(i) for i in range(num_fields)]))
        if apply_primary_beam:
            utils.correct_primary_beam(outms, imagename, pol=pol, fast_vis=fast_vis)
        if not fast_vis:
            image_list=[]
            for n,pola in enumerate(['I','Q','U','V','XX','YY']):
                if os.path.isfile(imagename+ "-"+pola+"-image.fits"):
                    image_list.append(imagename+ "-"+pola+"-image.fits")
                   
            if os.path.isfile(imagename+"-image.fits"):
                image_list.append(imagename+ "-image.fits")
            helio_image = utils.convert_to_heliocentric_coords(outms, image_list)
        else:
            num_fields=utils.get_total_fields(outms)
            image_names=utils.collect_fast_fits(imagename, pol)
            
            image_list=[]
            for name in image_names:
                if os.path.isfile(name):
                    image_list.append(name)
            helio_image = utils.convert_to_heliocentric_coords(outms, image_list)    
        logging.info('Imaging completed for ' + solar_ms)

        time2=timeit.default_timer()
        logging.info('Time taken for producing final image: {0:.1f} s'.format(time2-time1))
        time_end=timeit.default_timer()
        logging.info('Time taken to complete all processing: {0:.1f} s'.format(time_end-time_begin)) 
        return outms, helio_image
    else:
        if delete==True:
            os.system("rm -rf *model*")
        time_end=timeit.default_timer()
        logging.info('Time taken to complete all processing: {0:.1f} s'.format(time_end-time_begin)) 
        return outms, None



def manual_split_corrected_ms(vis, outputvis):
    tb.open(vis, nomodify=False)
    try:
        corrected_data = tb.getcol('CORRECTED_DATA')
        tb.putcol('DATA', corrected_data)
        tb.flush()
    except Exception as e:
        logging.debug("Hand split method did not work")
        raise e
    finally:
        tb.close() 
    os.system("mv " + vis + " " + outputvis)
    return outputvis   
  
@profile
def image_ms_quick(solar_ms, calib_ms=None, bcal=None, do_selfcal=True, imagename='sun_only',
             imsize=1024, cell='1arcmin', logfile='analysis.log', logging_level='info',
             caltable_folder='caltables/', num_phase_cal=1, num_apcal=1, freqbin=4,
             do_fluxscaling=False, do_final_imaging=True, pol='I', delete=True,
             refant='202', niter0=600, niter_incr=200, overwrite=False,
             auto_pix_fov=False, fast_vis=False, fast_vis_image_model_subtraction=False,
             delete_allsky=True, sky_image=None, quiet=True):
    """
    Pipeline to calibrate and imaging a solar visibility. 
    This is the version that optimizes the speed with a somewhat reduced image dynamic range.

    :param solar_ms: input solar measurement set
    :param calib_ms: (optional) input measurement set for generating the calibrations, usually is one observed at night
    :param bcal: (optional) bandpass calibration table. If not provided, use calib_ms to generate one.
    :param full_di_selfcal_rounds: [rounds of phase-only selfcal, rounds of amp-phase selfcal]
            for directional independent (full sky) full selfcalibration runs
    """
    
    if logging_level.lower() == 'info':
        logging.basicConfig(filename=logfile,
            format='%(asctime)s %(levelname)-8s %(message)s',
            level=logging.INFO,
            datefmt='%Y-%m-%d %H:%M:%S')
        
    if not os.path.isdir(caltable_folder):
        os.mkdir(caltable_folder)
    if os.path.isfile(imagename + "-image.fits"):
        if not overwrite:
            return None, imagename + "-image.helio.fits"

    logging.debug('==========Working on a new solar ms file {0:s}============='.format(solar_ms))
    
    if fast_vis:
        utils.make_wsclean_compatible(solar_ms)
        utils.swap_fastms_pols(solar_ms)
        utils.correct_fastms_amplitude_scale(solar_ms)
        
    time_begin=timeit.default_timer()
    time1=timeit.default_timer()
    solar_ms = calibration.do_bandpass_correction(solar_ms, calib_ms=calib_ms, bcal=bcal, \
                        caltable_folder=caltable_folder, freqbin=freqbin, fast_vis=fast_vis)
    time2=timeit.default_timer()
    logging.debug('Time taken to do the bandpass correction is: {0:.1f} s'.format(time2-time1)) 
    time1=time2
    logging.debug('Analysing ' + solar_ms)
    if do_selfcal:
        #outms_di = selfcal.DI_selfcal(solar_ms, logging_level=logging_level, full_di_selfcal_rounds=full_di_selfcal_rounds,
        #                      partial_di_selfcal_rounds=[0, 0], pol=pol, refant=refant, caltable_folder=caltable_folder)
        mstime_str = utils.get_timestr_from_name(solar_ms)
        success = utils.put_keyword(solar_ms, 'di_selfcal_time', mstime_str, return_status=True)
        success = selfcal.do_selfcal(solar_ms, num_phase_cal=num_phase_cal, num_apcal=num_apcal, logging_level=logging_level, pol=pol,
            refant=refant, niter0=niter0, niter_incr=niter_incr, caltable_folder=caltable_folder, auto_pix_fov=auto_pix_fov, quiet=quiet)
        outms_di = solar_ms[:-3] + "_selfcalibrated.ms"
        if do_fluxscaling:
            logging.debug('Doing a flux scaling using background strong sources')
            fc=flux_scaling.flux_scaling(vis=solar_ms, min_beam_val=0.1, pol=pol)
            fc.correct_flux_scaling()
            logging.debug('Splitted the selfcalibrated MS into a file named ' + solar_ms[:-3] + "_selfcalibrated.ms")
            split(vis=solar_ms, outputvis=outms_di, datacolumn='data')
        else:
            logging.debug('Splitted the selfcalibrated MS into a file named ' + solar_ms[:-3] + "_selfcalibrated.ms")
            #split(vis=solar_ms, outputvis=outms_di, datacolumn='corrected')
            ##### putting in a hand-split in an effort to run the realtime pipeline continuously
            manual_split_corrected_ms(solar_ms, outms_di)
            
        time2=timeit.default_timer()
        logging.debug('Time taken for selfcal and fluxscaling is: {0:.1f} s'.format(time2-time1))
        print(outms_di)
    else:
        outms_di = solar_ms

    # Do non-solar source removal
    time1=time2
    print('Removing non-solar sources in the sky')
    outms = source_subtraction.remove_nonsolar_sources(outms_di, remove_strong_sources_only=True, niter=1000, \
                                pol=pol, fast_vis= fast_vis, fast_vis_image_model_subtraction=fast_vis_image_model_subtraction,
                                delete_allsky=delete_allsky, skyimage=sky_image)
    time2=timeit.default_timer()
    logging.debug('Time taken for non-solar source removal is {0:.1f} s'.format(time2-time1))
    logging.debug('The source subtracted MS is ' + outms)
    time1=time2

    logging.debug('Changing the phasecenter to position of Sun')
    change_phasecenter(outms)
    time2=timeit.default_timer()
    logging.debug('Time taken for changing phasecenter: {0:.1f} s'.format(time2-time1))
    time1=time2
    if do_final_imaging:
        logging.debug('Generating final solar centered image')
        deconvolve.run_wsclean(outms, imagename=imagename, auto_mask=5, minuv_l='0', predict=False, 
                               size=imsize, scale=cell, pol=pol)
        logging.debug('Correcting for the primary beam at the location of Sun')
        utils.correct_primary_beam(outms, imagename, pol=pol)
        for n,pola in enumerate(['I','Q','U','V','XX','YY']):
            if os.path.isfile(imagename+ "-"+pola+"-image.fits"):
                helio_image = utils.convert_to_heliocentric_coords(outms, imagename+ "-"+pola+"-image.fits")
            elif pola=='I' and os.path.isfile(imagename+"-image.fits"):
                helio_image = utils.convert_to_heliocentric_coords(outms, imagename+"-image.fits")
        time2=timeit.default_timer()
        logging.debug('Imaging completed for ' + solar_ms)
        if delete==True:
            os.system("rm -rf *model*")
        return outms, helio_image
        
    else:
        if delete==True:
            os.system("rm -rf *model*")
        logging.debug('Time taken for producing final image: {0:.1f} s'.format(time2-time1))
        time_end=timeit.default_timer()
        logging.debug('Time taken to complete all processing: {0:.1f} s'.format(time_end-time_begin)) 
        return outms, None
        
   
@profile
def solar_pipeline(time_duration, calib_time_duration, freqstr, filepath, time_integration=8, time_cadence=100,
                   observation_integration=8,
                   calib_ms=None, bcal=None, selfcal=False, imagename='sun_only',
                   imsize=512, cell='1arcmin', logfile='analysis.log', logging_level='info',
                   caltable_folder='caltables',pol='I',refant='202'):
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
        calib_file = glob.glob(caltable_folder + '/*.bcal')
        if len(calib_file) != 0:
            bcal = calib_file[0]
        imagename = "sun_only_" + filename[:-3]
        outms, helio_image = image_ms(filename, calib_ms=calib_filename, bcal=bcal, selfcal=True,
                                    imagename=imagename, do_final_imaging=True, pol=pol, refant=refant)
        filename = fp.get_current_file_for_selfcal(freqstr[0])

    filename = fp.get_current_file_for_selfcal(freqstr[0])
    while filename is not None:
        imagename = "sun_only_" + filename[:-3]
        outms, helio_image = image_ms(filename, calib_ms=calib_ms, bcal=bcal, selfcal=True,
                                    imagename=imagename, do_final_imaging=True, pol=pol, refant=refant)
        filename = fp.get_current_file_for_imaging(freqstr[0])


def apply_solutions_and_image(msname, bcal, imagename):
    logging.info('Analysing ' + msname)
    calibration.apply_calibration(msname, gaintable=bcal, doantflag=True, doflag=True, do_solar_imaging=False)
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
    outms = source_subtraction.remove_nonsolar_sources(solar_ms)
    solar_ms = outms
    num_dd_cal = len(dd_cal)
    if num_dd_cal != 0:
        applycal(solar_ms, gaintable=dd_cal, calwt=[False] * len(dd_cal), applymode='calonly')
        flagdata(vis=solar_ms, mode='rflag', datacolumn='corrected')
        split(vis=solar_ms, outputvis=solar_ms[:-3] + "_sun_selfcalibrated.ms")
    else:
        split(vis = solar_ms, outputvis=solar_ms[:-3]+"_sun_selfcalibrated.ms",datacolumn='data')
    outms = solar_ms[:-3] + "_sun_selfcalibrated.ms"
    outms = source_subtraction.remove_nonsolar_sources(outms, remove_strong_sources_only=False)
    change_phasecenter(outms)
    deconvolve.run_wsclean(outms, imagename=imagename, auto_mask=5, minuv_l='0', predict=False, 
                           size=1024, scale='1arcmin')
    utils.correct_primary_beam(outms, imagename + "-image.fits")
    logging.info('Imaging completed for ' + msname)
