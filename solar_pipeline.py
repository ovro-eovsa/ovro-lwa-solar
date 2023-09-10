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
import utils,flagging,calibration,selfcal,source_subtraction,deconvolve
import logging, glob
from file_handler import File_Handler
from primary_beam import analytic_beam as beam 
import primary_beam
from generate_calibrator_model import model_generation
import generate_calibrator_model
import timeit

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
    print(ra1)
    print(dec1)
    logging.info("Changing the phasecenter to " + ra1 + " " + dec1)
    os.system("chgcentre " + msfile + " " + ra1 + " " + dec1)


def correct_primary_beam(msfile, imagename,pol='I',fast_vis=False):
    m = utils.get_sun_pos(msfile, str_output=False)
    logging.debug('Solar ra: ' + str(m['m0']['value']))
    logging.debug('Solar dec: ' + str(m['m1']['value']))
    d = me.measure(m, 'AZEL')
    logging.debug('Solar azimuth: ' + str(d['m0']['value']))
    logging.debug('Solar elevation: ' + str(d['m1']['value']))
    elev = d['m1']['value']*180/np.pi
    az=d['m0']['value']*180/np.pi
    pb=beam(msfile=msfile)
    pb.srcjones(az=[az],el=[elev])
    jones_matrices=pb.get_source_pol_factors(pb.jones_matrices[0,:,:])

    md=generate_calibrator_model.model_generation(vis=msfile)
    if fast_vis==False:
        if pol=='I':
            scale=md.primary_beam_value(0,jones_matrices)
            logging.info('The Stokes I beam correction factor is ' + str(round(scale, 4)))
            hdu = fits.open(imagename+ "-image.fits", mode='update')
            hdu[0].data /= scale
            hdu.flush()
            hdu.close()
        else:
            for pola in ['I','Q','U','V','XX','YY']:
                if pola=='I' or pola=='XX' or pola=='YY':
                    n=0
                elif pola=='Q':
                    n=1
                elif pola=='U':
                    n=2
                else:
                    n==3
                scale=md.primary_beam_value(n,jones_matrices)
                logging.info('The Stokes '+pola+' beam correction factor is ' + str(round(scale, 4)))
                if os.path.isfile(imagename+ "-"+pola+"-image.fits"):
                    hdu = fits.open(imagename+ "-"+pola+"-image.fits", mode='update')
                    hdu[0].data /= scale
                    hdu.flush()
                    hdu.close()
                elif pola=='I' and os.path.isfile(imagename+"-image.fits"):
                    hdu = fits.open(imagename+"-image.fits", mode='update')
                    hdu[0].data /= scale
                    hdu.flush()
                    hdu.close()
    else:
        image_names=utils.get_fast_vis_imagenames(msfile,imagename,pol)
        for name in image_names:
            if os.path.isfile(name[1]):
                if pol=='I':
                    scale=md.primary_beam_value(0,jones_matrices)
                else:
                    pola=name[1].split('-')[-1]
                    if pola=='I' or pola=='XX' or pola=='YY':
                        n=0
                    elif pola=='Q':
                        n=1
                    elif pola=='U':
                        n=2
                    else:
                        n==3
                    scale=md.primary_beam_value(n,jones_matrices)
                hdu = fits.open(name[1], mode='update')
                hdu[0].data /= scale
                hdu.flush()
                hdu.close()
    return


def image_ms(solar_ms, calib_ms=None, bcal=None, do_selfcal=True, imagename='sun_only',
             imsize=1024, cell='1arcmin', logfile='analysis.log', logging_level='info',
             caltable_fold='caltables', full_di_selfcal_rounds=[2,1], partial_di_selfcal_rounds=[0, 1],
             full_dd_selfcal_rounds=[1, 1], partial_dd_selfcal_rounds=[0, 1], do_final_imaging=True,pol='I',overwrite=False,\
             solint_full_DI_selfcal=14400, solint_partial_DI_selfcal=3600, solint_full_DD_selfcal=1800, solint_partial_DD_selfcal=600,\
             fast_vis=False, fast_vis_image_model_subtraction=False):
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
            level=logging.INFO,
            datefmt='%Y-%m-%d %H:%M:%S')
    
    
    
        
    if not os.path.isdir(caltable_fold):
        os.mkdir(caltable_fold)
    if os.path.isfile(imagename + "-image.fits"):
        if not overwrite:
            return None, imagename + "-image.helio.fits"
            
    utils.make_wsclean_compatible(solar_ms)

    time1=timeit.default_timer()
    solar_ms = calibration.do_bandpass_correction(solar_ms, calib_ms=calib_ms, bcal=bcal, caltable_fold=caltable_fold, fast_vis=fast_vis)
    time2=timeit.default_timer()
    logging.info('Time taken to do the bandpass correction is: '+str(time2-time1)+"seconds")
    time1=time2
    logging.info('Analysing ' + solar_ms)
    if do_selfcal:
        outms_di = selfcal.DI_selfcal(solar_ms, logging_level=logging_level, full_di_selfcal_rounds=full_di_selfcal_rounds,
                              partial_di_selfcal_rounds=partial_di_selfcal_rounds,pol=pol,solint_full_selfcal=solint_full_DI_selfcal,\
                              solint_partial_selfcal=solint_partial_DI_selfcal, fast_vis=fast_vis,calib_ms=calib_ms)
        time2=timeit.default_timer()
        logging.info('Time taken for DI selfcal and fluxscaling is: '+str(time2-time1)+"seconds")
        time1=time2
        print (outms_di)
        logging.info('Removing the strong sources in the sky')
        outms_di_ = source_subtraction.remove_nonsolar_sources(outms_di,pol=pol, fast_vis=fast_vis,\
                                        fast_vis_image_model_subtraction=fast_vis_image_model_subtraction)
        time2=timeit.default_timer()
        logging.info('Time taken for strong source removal is: '+str(time2-time1)+"seconds")
        time1=time2
        logging.info('The strong source subtracted MS is ' + outms_di_)
        logging.info('Starting to do Stokes I selfcal towards direction of sun')
        
        if fast_vis==False:
            outms_dd = selfcal.DD_selfcal(outms_di_, logging_level=logging_level, full_dd_selfcal_rounds=full_dd_selfcal_rounds,
                                  partial_dd_selfcal_rounds=partial_dd_selfcal_rounds,pol=pol, solint_full_selfcal=solint_full_DD_selfcal,\
                                  solint_partial_selfcal=solint_partial_DD_selfcal)
            time2=timeit.default_timer()
            logging.info('Time taken for DD selfcal is: '+str(time2-time1)+"seconds")
            time1=time2
            logging.info('Removing almost all sources in the sky except Sun')
            print ('Removing almost all sources in the sky except Sun')
            outms = source_subtraction.remove_nonsolar_sources(outms_dd, imagename='for_weak_source_subtraction',
                                            remove_strong_sources_only=False,pol=pol)
            time2=timeit.default_timer()
            logging.info('Time taken for weak source removal is: '+str(time2-time1)+"seconds")
            time1=time2
            logging.info('The source subtracted MS is ' + outms)
        else:
            outms = selfcal.DD_selfcal(outms_di_, logging_level=logging_level, full_dd_selfcal_rounds=full_dd_selfcal_rounds,
                                  partial_dd_selfcal_rounds=partial_dd_selfcal_rounds,pol=pol, solint_full_selfcal=solint_full_DD_selfcal,\
                                  solint_partial_selfcal=solint_partial_DD_selfcal,fast_vis=fast_vis,calib_ms=calib_ms)
            
    else:
        logging.info('Removing almost all sources in the sky except Sun')
        outms = source_subtraction.remove_nonsolar_sources(solar_ms,pol=pol)
        logging.info('The source subtracted MS is ' + outms)

    logging.info('Changing the phasecenter to position of Sun')
    change_phasecenter(outms)
    time2=timeit.default_timer()
    logging.info('Time taken for changing phasecenter: '+str(time2-time1)+"seconds")
    time1=time2
    if do_final_imaging:
        logging.info('Generating final solar centered image')
        if fast_vis==False:
            deconvolve.run_wsclean(outms, imagename=imagename, automask_thresh=5, uvrange='0', predict=False, \
                                imsize=imsize, cell=cell,pol=pol, fast_vis=fast_vis)
        else:
            num_fields=utils.get_total_fields(outms)
            deconvolve.run_wsclean(outms, imagename=imagename, automask_thresh=5, uvrange='0', predict=False, \
                                imsize=imsize, cell=cell,pol=pol, fast_vis=fast_vis,field=','.join([str(i) for i in range(num_fields)]))
            
        time2=timeit.default_timer()
        logging.info('Correcting for the primary beam at the location of Sun')
        logging.info('Time taken for producing final image: '+str(time2-time1)+"seconds")
        time1=time2
        correct_primary_beam(outms, imagename,pol=pol,fast_vis=fast_vis)  ### suport fast_vis images
        
        time2=timeit.default_timer()
        logging.info('Time taken for primary beam correction: '+str(time2-time1)+"seconds")
        time1=time2
        # make_solar_image(outms, imagename=imagename, imsize=imsize, cell=cell)
        
        if fast_vis==False:
            for n,pola in enumerate(['I','Q','U','V','XX','YY']):
                if os.path.isfile(imagename+ "-"+pola+"-image.fits"):
                    helio_image = utils.convert_to_heliocentric_coords(outms, imagename+ "-"+pola+"-image.fits")
                elif pola=='I' and os.path.isfile(imagename+"-image.fits"):
                    helio_image = utils.convert_to_heliocentric_coords(outms, imagename+"-image.fits")
        else:
            image_names=utils.get_fast_vis_imagenames(outms,imagename,pol)
            for name in image_names:
                if os.path.isfile(name[1]):
                    helio_image = utils.convert_to_heliocentric_coords(outms, name[1])    
            
        time2=timeit.default_timer()
        logging.info('Time taken for converting to heliocentric image: '+str(time2-time1)+"seconds")
        logging.info('Imaging completed for ' + solar_ms)
        return outms, helio_image
        
    else:
        return outms, None


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
