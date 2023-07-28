from casatasks import clearcal, ft, bandpass, applycal, flagdata, tclean, flagmanager, uvsub, gaincal, split, imstat, \
    gencal
from casatools import table, measures, componentlist, msmetadata
import math
import sys, os, time
import numpy as np
import utils,flagging
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


def gen_calibration(msfile, modelcl=None, uvrange='', bcaltb=None, logging_level='info', caltable_fold='caltables'):
    """
    This function is for doing initial self-calibrations using strong sources that are above the horizon
    It is recommended to use a dataset observed at night when the Sun is not in the field of view
    :param uvrange: uv range to consider for calibration. Following CASA's syntax, e.g., '>1lambda'
    :param msfile: input CASA ms visibility for calibration
    :param modelcl: input model of strong sources as a component list, produced from gen_model_cl()
    """
	
    time1=timeit.default_timer()
    if not modelcl or not (os.path.exists(modelcl)):
        print('Model component list does not exist. Generating one from scratch.')
        logging.info('Model component list does not exist. Generating one from scratch.')
       
       
        md=model_generation(vis=msfile,separate_pol=True) 	    
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
    time2=timeit.default_timer()
    logging.info("Time for producing bandpass table: "+str(time2-time1)+"seconds")
    return bcaltb


def apply_calibration(msfile, gaintable=None, doantflag=False, doflag=False, antflagfile=None, do_solar_imaging=True,
                      imagename='test'):
    if doantflag:
        logging.info("Flagging using auro-correlation")
        flagging.flag_bad_ants(msfile, antflagfile=antflagfile)
    if not gaintable:
        logging.error("No bandpass table found. Proceed with extreme caution")
        print('No calibration table is provided. Abort... ')
    else:
        if type(gaintable) == str:
            gaintable = [gaintable]
    # Apply the calibration
    time1=timeit.default_timer()
    clearcal(msfile)
    applycal(msfile, gaintable=gaintable, flagbackup=True, applymode='calflag')
    time2=timeit.default_timer()
    logging.info("Time for applying bandpass table: "+str(time2-time1)+"seconds")
    if doflag == True:
        logging.debug("Running rflag on corrected data")
        flagdata(vis=msfile, mode='rflag', datacolumn='corrected')
    sunpos = utils.get_sun_pos(msfile)
    if do_solar_imaging:
        tclean(msfile, imagename=imagename, imsize=[512], cell=['1arcmin'],
               weighting='uniform', phasecenter=sunpos, niter=500)
        print('Solar image made {0:s}.image'.format(imagename))
        
        
        
def do_bandpass_correction(solar_ms, calib_ms=None, bcal=None, caltable_fold='caltables', logging_level='info'):
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
            flagging.flag_bad_ants(calib_ms)
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
