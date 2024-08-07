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
from . import utils,flagging,calibration,deconvolve
from . import flux_scaling
import logging, glob
from .file_handler import File_Handler
from .primary_beam import analytic_beam as beam 
from . import primary_beam
from .generate_calibrator_model import model_generation
from . import generate_calibrator_model
import timeit
tb = table()
me = measures()
cl = componentlist()
msmd = msmetadata()


def do_selfcal(msfile, num_phase_cal=2, num_apcal=2, applymode='calflag', logging_level='info', caltable_folder='caltables/',
               ms_keyword='di_selfcal_time',pol='I', refant='202', niter0=1000, niter_incr=500, auto_pix_fov=False, \
               bandpass_selfcal=False):
    
    time1=timeit.default_timer()          
    logging.debug('The plan is to do ' + str(num_phase_cal) + " rounds of phase selfcal")
    logging.debug('The plan is to do ' + str(num_apcal) + " rounds of amplitude-phase selfcal")
    
    if pol!='I':
        pol='XX,YY'
        
    num_pol=2
    if pol=='XX,YY':
        num_pol=4

    max1 = np.zeros(num_pol)
    min1 = np.zeros(num_pol)

    if num_phase_cal!=0:
        niters = np.arange(num_phase_cal) * niter_incr + niter0
    else:
        niters = np.arange(2) * niter_incr + niter0
    
    for i in range(num_phase_cal):
        imagename = msfile[:-3] + "_self" + str(i)
        deconvolve.run_wsclean(msfile, imagename=imagename, niter=niters[i], 
                               mgain=0.9, auto_mask=False, auto_threshold=False, pol=pol, auto_pix_fov=auto_pix_fov)
        good = utils.check_image_quality(imagename, max1, min1)
       
        print(good)
        logging.debug('Maximum pixel values are: ' + str(max1[0]) + "," + str(max1[1]))
        logging.debug('Minimum pixel values around peaks are: ' + str(min1[0]) + "," + str(min1[1]))
        if not good:
            logging.debug('Dynamic range has reduced. Doing a round of flagging')
            flagdata(vis=msfile, mode='rflag', datacolumn='corrected')
            deconvolve.run_wsclean(msfile, imagename=imagename, niter=niters[i], mgain=0.9, 
                                   auto_mask=False, auto_threshold=False, pol=pol, auto_pix_fov=auto_pix_fov)
            
            good = utils.check_image_quality(imagename, max1, min1,reorder=False)
            
            print(good)
            logging.debug('Maximum pixel values are: ' + str(max1[0]) + "," + str(max1[1]))
            logging.debug('Minimum pixel values around peaks are: ' + str(min1[0]) + "," + str(min1[1]))
            if not good:
                logging.debug('Flagging could not solve the issue. Restoring flags, applying last good solutions.')
                utils.restore_flag(msfile)
                logging.debug('Restoring flags')
                os.system("rm -rf " + imagename + "-*.fits")
                caltable = msfile[:-3] + "_self" + str(i - 1) + ".gcal"
                os.system("rm -rf " + caltable)
                imagename = msfile[:-3] + "_self" + str(i - 2)
                caltable = imagename + ".gcal"
                if os.path.isdir(caltable):
                    logging.debug("Applying " + caltable)
                    applycal(vis=msfile, gaintable=caltable, calwt=[False], applymode=applymode)
                    os.system("cp -r " + caltable + " " + caltable_folder)
                else:
                    logging.warning("No caltable found. Setting corrected data to DATA")
                    clearcal(msfile)
                return good
        logging.debug("Finding gain solutions and writing in into " + imagename + ".gcal")
        time1=timeit.default_timer()
        if not bandpass_selfcal:
            gaincal(vis=msfile, caltable=imagename + ".gcal", uvrange=">10lambda",
                calmode='p', solmode='L1R', rmsthresh=[10, 8, 6], refant=refant)
        else:
            calibration.find_bandpass_sol(msfile,caltable=imagename + ".gcal", uvrange=">10lambda",\
                refant=refant,calmode='ap')
        time2=timeit.default_timer()
        logging.debug('Solving for selfcal gain solutions took {0:.1f} s'.format(time2-time1))
        utils.put_keyword(imagename + ".gcal", ms_keyword, utils.get_keyword(msfile, ms_keyword))
        if logging_level == 'debug' or logging_level == 'DEBUG':
            utils.get_flagged_solution_num(imagename + ".gcal")
        logging.debug("Applying solutions")
        time1=timeit.default_timer()
        applycal(vis=msfile, gaintable=imagename + ".gcal", calwt=[False], applymode=applymode)
        time2=timeit.default_timer()
        logging.debug('Apply selfcal gain solutions took {0:.1f} s'.format(time2-time1))

    logging.debug("Phase self-calibration finished successfully")

    if num_phase_cal > 0:
        final_phase_caltable = imagename + ".gcal"
    else:
        final_phase_caltable = ''
    for i in range(num_phase_cal, num_phase_cal + num_apcal):
        imagename = msfile[:-3] + "_self" + str(i)
        deconvolve.run_wsclean(msfile, imagename=imagename, niter=np.max(niters) + (i+1) * niter_incr,
                                mgain=0.9, auto_mask=False, auto_threshold=False, pol=pol, auto_pix_fov=auto_pix_fov)
        
        good = utils.check_image_quality(imagename, max1, min1)
        
        logging.debug('Maximum pixel values are: ' + str(max1[0]) + "," + str(max1[1]))
        logging.debug('Minimum pixel values around peaks are: ' + str(min1[0]) + "," + str(min1[1]))
        if not good:
            logging.debug('Dynamic range has reduced. Doing a round of flagging')
            flagdata(vis=msfile, mode='rflag', datacolumn='corrected')
            deconvolve.run_wsclean(msfile, imagename=imagename, niter=np.max(niters),
                                mgain=0.9, auto_mask=False, auto_threshold=False, pol=pol, auto_pix_fov=auto_pix_fov)
            
            good = utils.check_image_quality(imagename, max1, min1, reorder=False)
            
            print(good)
            if not good:
                logging.debug('Flagging could not solve the issue. Restoring flags, applying last good solutions.')
                utils.restore_flag(msfile)
                os.system("rm -rf " + imagename + "-*.fits")
                caltable = msfile[:-3] + "_self" + str(i - 1) + "_ap_over_p.gcal"
                os.system("rm -rf " + caltable)
                imagename = msfile[:-3] + "_self" + str(i - 2)
                caltable = imagename + "_ap_over_p.gcal"
                if os.path.isdir(caltable):
                    logging.debug("Applying " + caltable + " and " + final_phase_caltable)
                    if num_phase_cal > 0:
                        applycal(vis=msfile, gaintable=[caltable, final_phase_caltable], calwt=[False, False],
                                 applymode=applymode)
                        os.system("cp -r " + final_phase_caltable + " " + caltable_folder)
                    else:
                        applycal(vis=msfile, gaintable=[caltable], calwt=[False, False], applymode=applymode)
                    os.system("cp -r " + caltable + " " + caltable_folder)

                else:
                    logging.warning("No good aplitude-phase selfcal solution found.")
                    if num_phase_cal > 0:
                        logging.debug("Applying " + final_phase_caltable)
                        applycal(vis=msfile, gaintable=[final_phase_caltable], calwt=[False], applymode=applymode)
                        os.system("cp -r " + final_phase_caltable + " " + caltable_folder)
                    else:
                        logging.warning("No caltable found. Setting corrected data to DATA")
                        clearcal(msfile)
                return good
        caltable = imagename + "_ap_over_p.gcal"
        
        if not bandpass_selfcal:
            gaincal(vis=msfile, caltable=caltable, uvrange=">10lambda",
                    calmode='ap', solnorm=True, normtype='median', solmode='L1R',
                    rmsthresh=[10, 8, 6], gaintable=final_phase_caltable, refant=refant)
        else:
            calibration.find_bandpass_sol(msfile,caltable=imagename + ".gcal", uvrange=">10lambda",\
                refant=refant,calmode='ap')
        utils.put_keyword(caltable, ms_keyword, utils.get_keyword(msfile, ms_keyword))
        if logging_level == 'debug' or logging_level == 'DEBUG':
            utils.get_flagged_solution_num(imagename + "_ap_over_p.gcal")
        applycal(vis=msfile, gaintable=[caltable, final_phase_caltable], calwt=[False, False], applymode=applymode)
        if i == num_phase_cal:
            flagdata(vis=msfile, mode='rflag', datacolumn='corrected')
    logging.debug('Flagging on the residual')
    flagdata(vis=msfile, mode='rflag', datacolumn='residual')
    if num_apcal>0:
        os.system("cp -r " + caltable + " " + caltable_folder)

    if len(final_phase_caltable)!=0:    
        os.system("cp -r " + final_phase_caltable + " " + caltable_folder)
    time2=timeit.default_timer()
    logging.debug("Time taken for selfcal: "+str(time2-time1)+"seconds")
    return True


def do_fresh_selfcal(solar_ms, num_phase_cal=3, num_apcal=5, logging_level='info',pol='I', refant='202', niter0=600, niter_incr=200):
    """
    Do fresh self-calibration if no self-calibration tables are found

    :param solar_ms: input solar visibility
    :param num_phase_cal: (maximum) rounds of phase-only selfcalibration. Default to 3
    :param num_apcal: (maximum) rounds of ampitude and phase selfcalibration. Default to 5
    :param logging_level: type of logging, default to "info"
    :return: N/A
    """
    logging.debug('Starting to do direction independent Stokes I selfcal')
    success = do_selfcal(solar_ms, num_phase_cal=num_phase_cal, num_apcal=num_apcal, logging_level=logging_level, pol=pol, 
            refant=refant, niter0=niter0, niter_incr=niter_incr)
    if not success:
#TODO Understand why this step is needed
        logging.debug('Starting fresh selfcal as DR decreased significantly')
        clearcal(solar_ms)
        success = do_selfcal(solar_ms, num_phase_cal=num_phase_cal, num_apcal=num_apcal, logging_level=logging_level, pol=pol, 
                refant=refant, niter0=niter0, niter_incr=niter_incr)
    return

def convert_caltables_for_fast_vis(solar_ms,calib_ms,caltables):
    fast_caltables=[]
    for caltb in caltables:
        fast_caltables.append(calibration.make_fast_caltb_from_slow(calib_ms, solar_ms, caltb))
    return fast_caltables
    

def DI_selfcal(solar_ms, solint_full_selfcal=14400, solint_partial_selfcal=3600, caltable_folder = 'caltables/', calib_ms=None,
               full_di_selfcal_rounds=[1,1], partial_di_selfcal_rounds=[1, 1], logging_level='info', pol='I', refant='202',
               fast_vis=False, niter0=1000, niter_incr=500, do_fluxscaling=False):
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
    
    if fast_vis==True:
        applymode='calonly'
        solint_full_selfcal=1e8   ### putting a insanely high value so that
        solint_partial_selfcal=1e8 ### selfcal is not done for fast visibilities
    else:
        applymode='calflag'
        
    sep = 100000000
    prior_selfcal = False
    caltables = []

    mstime = utils.get_time_from_name(solar_ms)
    mstime_str = utils.get_timestr_from_name(solar_ms)
    msfreq_str = utils.get_freqstr_from_name(solar_ms)

    caltables = glob.glob(caltable_folder + "/*" + msfreq_str + "*.gcal")
    if len(caltables) != 0:
        prior_selfcal = True

    if prior_selfcal:
        dd_cal = glob.glob(caltable_folder + "/*" + msfreq_str + "*sun_only*.gcal")
        di_cal = [cal for cal in caltables if cal not in dd_cal]
        print(di_cal)
        selfcal_time = utils.get_selfcal_time_to_apply(solar_ms, di_cal)
        print(selfcal_time)

        caltables = glob.glob(caltable_folder + "/" + selfcal_time + "*" + msfreq_str + "*.gcal")
        dd_cal = glob.glob(caltable_folder + "/" + selfcal_time +  "*" + msfreq_str + "*sun_only*.gcal")
        di_cal = [cal for cal in caltables if cal not in dd_cal]

        if len(di_cal) != 0:
            di_selfcal_time_str, success = utils.get_keyword(di_cal[0], 'di_selfcal_time', return_status=True)
            print(di_selfcal_time_str, success)
            if success:
                di_selfcal_time = utils.get_time_from_name(di_selfcal_time_str)

                sep = abs((di_selfcal_time - mstime).value * 86400)  ### in seconds
                
                if fast_vis==True:
                    if calib_ms:
                        di_cal=convert_caltables_for_fast_vis(solar_ms,calib_ms,di_cal)
                    else:
                        raise RuntimeError("Supplying a calibration MS is mandatory for imaging fast visibilities")

                applycal(solar_ms, gaintable=di_cal, calwt=[False] * len(di_cal), applymode=applymode)
                flagdata(vis=solar_ms, mode='rflag', datacolumn='corrected')

                if sep < solint_partial_selfcal:
                    logging.debug('Seperation is shorter than the partial solint, skipping all direction independent selfcal')
                    logging.debug('Applying gaintables from ' + di_selfcal_time_str)
                    applycal(solar_ms, gaintable=di_cal, calwt=[False] * len(di_cal))
                    flagdata(vis=solar_ms, mode='rflag', datacolumn='corrected')
                    success = utils.put_keyword(solar_ms, 'di_selfcal_time', di_selfcal_time_str, return_status=True)

                elif sep > solint_partial_selfcal and sep < solint_full_selfcal:
                    # Partical selfcal does one additional round of ap self-calibration
                    logging.debug('Seperation is shorter than the full solint, skipping phase-only selfcal')
                    logging.debug(
                        'Starting to do direction independent Stokes I selfcal after applying ' + di_selfcal_time_str)
                    applycal(solar_ms, gaintable=di_cal, calwt=[False] * len(di_cal))
                    success = utils.put_keyword(solar_ms, 'di_selfcal_time', mstime_str, return_status=True)
                    flagdata(vis=solar_ms, mode='rflag', datacolumn='corrected')
                    success = do_selfcal(solar_ms, num_phase_cal=0,
                                         num_apcal=partial_di_selfcal_rounds[1], logging_level=logging_level, pol=pol, refant=refant, 
                                         niter0=niter0, niter_incr=niter_incr)
                    datacolumn = 'corrected'
                else:
                    success = utils.put_keyword(solar_ms, 'di_selfcal_time', mstime_str, return_status=True)
                    logging.debug('Seperation is longer than the full solint, doing fresh direction independent selfcal')
                    logging.debug(
                        'Starting to do direction independent Stokes I selfcal after applying ' + di_selfcal_time_str)
                    do_fresh_selfcal(solar_ms, num_phase_cal=full_di_selfcal_rounds[0],
                                     num_apcal=full_di_selfcal_rounds[1], logging_level=logging_level, pol=pol, refant=refant)
                    if success == False:
                        clearcal(solar_ms)
                        success = do_selfcal(solar_ms, logging_level=logging_level,pol=pol, refant=refant, niter0=niter0, niter_incr=niter_incr)
            else:
                success = utils.put_keyword(solar_ms, 'di_selfcal_time', mstime_str, return_status=True)
                logging.debug(
                    'Starting to do direction independent Stokes I selfcal as I failed to retrieve the keyword for DI selfcal')
                do_fresh_selfcal(solar_ms, num_phase_cal=full_di_selfcal_rounds[0],
                                 num_apcal=full_di_selfcal_rounds[1], logging_level=logging_level, pol=pol, refant=refant)
        else:
            success = utils.put_keyword(solar_ms, 'di_selfcal_time', mstime_str, return_status=True)
            logging.debug(
                'Starting to do direction independent Stokes I selfcal as mysteriously I did not find a suitable caltable')
            do_fresh_selfcal(solar_ms, num_phase_cal=full_di_selfcal_rounds[0],
                             num_apcal=full_di_selfcal_rounds[1], logging_level=logging_level, pol=pol, refant=refant)
    else:
        if not fast_vis:
            success = utils.put_keyword(solar_ms, 'di_selfcal_time', mstime_str, return_status=True)
            logging.debug('I do not find any existing selfcal tables. Starting to do fresh direction independent Stokes I selfcal')
            do_fresh_selfcal(solar_ms, num_phase_cal=full_di_selfcal_rounds[0],
                             num_apcal=full_di_selfcal_rounds[1], logging_level=logging_level, pol=pol, refant=refant)
        else:
            logging.warning("DI selfcal caltable not found. Proceed with caution.")
            solar_ms_slfcaled = solar_ms[:-3] + "_selfcalibrated.ms"
            gencal(vis=solar_ms, caltable="dummy.gencal", caltype='amp', parameter=1.0)
            applycal(solar_ms,gaintable="dummy.gencal") ### this is needed to ensure flux scaling works fine.
            os.system("rm -rf dummy.gencal")
    
    solar_ms_slfcaled = solar_ms[:-3] + "_selfcalibrated.ms"
    if do_fluxscaling:
        logging.debug('Doing a flux scaling using background strong sources')
        fc=flux_scaling.flux_scaling(vis=solar_ms, min_beam_val=0.1, pol=pol, fast_vis=fast_vis, calib_ms=calib_ms)
        fc.correct_flux_scaling()
        logging.debug('Splitted the selfcalibrated MS into a file named ' + solar_ms[:-3] + "_selfcalibrated.ms")
        split(vis=solar_ms, outputvis=solar_ms_slfcaled, datacolumn='data')
    else:
        logging.debug('Splitted the selfcalibrated MS into a file named ' + solar_ms[:-3] + "_selfcalibrated.ms")
        corrected_data_present=utils.check_corrected_data_present(solar_ms)
        if corrected_data_present:
            datacolumn='corrected'
        else:
            datacolumn='data'
        split(vis=solar_ms, outputvis=solar_ms_slfcaled, datacolumn=datacolumn)
            

    return solar_ms_slfcaled


def DD_selfcal(solar_ms, solint_full_selfcal=1800, solint_partial_selfcal=600, caltable_folder='caltables/', calib_ms=None,
               full_dd_selfcal_rounds=[3, 5], partial_dd_selfcal_rounds=[1, 1],
               logging_level='info', pol='I', refant='202', niter0=1000, niter_incr=500, fast_vis=False):
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

    if fast_vis==True:
        solint_full_selfcal=1e8   ### putting a insanely high value so that
        solint_partial_selfcal=1e8 ### selfcal is not done for fast visibilities
        
    mstime = utils.get_time_from_name(solar_ms)
    mstime_str = utils.get_timestr_from_name(solar_ms)
    msfreq_str = utils.get_freqstr_from_name(solar_ms)
    
    selfcal_time = utils.get_selfcal_time_to_apply(solar_ms, glob.glob(caltable_folder + "/*" + msfreq_str + "*.gcal"))
    
    sep = 100000000
    prior_selfcal = False

    caltables = glob.glob(caltable_folder + selfcal_time + "*" + msfreq_str + "*sun_only*.gcal")

    if len(caltables) != 0:
        prior_selfcal = True

    if prior_selfcal:
        dd_selfcal_time_str, success = utils.get_keyword(caltables[0], 'dd_selfcal_time', return_status=True)

        if success:
            dd_selfcal_time = utils.get_time_from_name(dd_selfcal_time_str)

            sep = abs((dd_selfcal_time - mstime).value * 86400)  ### in seconds
            
            if fast_vis==True:
                    if calib_ms:
                        caltables=convert_caltables_for_fast_vis(solar_ms,calib_ms,caltables)
                    else:
                        raise RuntimeError("Supplying a calibration MS is mandatory for imaging fast visibilities")

            applycal(solar_ms, gaintable=caltables, calwt=[False] * len(caltables), applymode='calonly')
            flagdata(vis=solar_ms, mode='rflag', datacolumn='corrected')

            if sep < solint_partial_selfcal:
                logging.debug('No direction dependent Stokes I selfcal after applying ' + dd_selfcal_time_str)
                success = utils.put_keyword(solar_ms, 'dd_selfcal_time', dd_selfcal_time_str, return_status=True)

            elif sep > solint_partial_selfcal and sep < solint_full_selfcal:
                success = utils.put_keyword(solar_ms, 'dd_selfcal_time', mstime_str, return_status=True)
                logging.debug(
                    'Starting to do direction dependent Stokes I selfcal after applying ' + dd_selfcal_time_str)
                success = do_selfcal(solar_ms, num_phase_cal=partial_dd_selfcal_rounds[0],
                                     num_apcal=partial_dd_selfcal_rounds[1], applymode='calonly',
                                     logging_level=logging_level, ms_keyword='dd_selfcal_time', pol=pol, refant=refant, niter0=niter0, niter_incr=niter_incr)
                datacolumn = 'corrected'


            else:
                success = utils.put_keyword(solar_ms, 'dd_selfcal_time', mstime_str, return_status=True)
                logging.debug(
                    'Starting to do direction dependent Stokes I selfcal after applying ' + dd_selfcal_time_str)
                success = do_selfcal(solar_ms, num_phase_cal=full_dd_selfcal_rounds[0],
                                     num_apcal=full_dd_selfcal_rounds[1], applymode='calonly',
                                     logging_level=logging_level, ms_keyword='dd_selfcal_time', pol=pol, refant=refant, niter0=niter0, niter_incr=niter_incr)
                datacolumn = 'corrected'
        else:
            success = utils.put_keyword(solar_ms, 'dd_selfcal_time', mstime_str, return_status=True)
            logging.debug(
                'Starting to do direction dependent Stokes I selfcal as I failed to retrieve the keyword for DD selfcal')
            success = do_selfcal(solar_ms, num_phase_cal=full_dd_selfcal_rounds[0],
                                 num_apcal=full_dd_selfcal_rounds[1], applymode='calonly',
                                 logging_level=logging_level, ms_keyword='dd_selfcal_time', pol=pol, refant=refant, niter0=niter0, niter_incr=niter_incr)



    else:
        if not fast_vis:
            success = utils.put_keyword(solar_ms, 'dd_selfcal_time', mstime_str, return_status=True)
            logging.debug('Starting to do direction dependent Stokes I selfcal')
            success = do_selfcal(solar_ms, num_phase_cal=full_dd_selfcal_rounds[0], num_apcal=full_dd_selfcal_rounds[1],
                                 applymode='calonly', logging_level=logging_level,
                                 ms_keyword='dd_selfcal_time', pol=pol, refant=refant, niter0=niter0, niter_incr=niter_incr)
        else:
            logging.warning("DD selfcal caltable not found. Proceed with caution.")
            os.system("cp -r "+solar_ms+" "+solar_ms[:-3] + "_sun_selfcalibrated.ms")
    if not fast_vis or prior_selfcal:
        logging.debug('Splitted the selfcalibrated MS into a file named ' + solar_ms[:-3] + "_sun_selfcalibrated.ms")
        split(vis=solar_ms, outputvis=solar_ms[:-3] + "_sun_selfcalibrated.ms")

    solar_ms = solar_ms[:-3] + "_sun_selfcalibrated.ms"
    return solar_ms
