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

def make_fast_caltb_from_slow(calib_ms, solar_ms, caltb, \
                            caltable_fold='caltables', overwrite=False):
    '''
    calib_ms: This is a essentially a MS which has slow visibilties
    solar_ms: The fast visibility MS
    caltb: the caltable generated from slow visibility data
           which will be converted to fast visibility format
    delayfile: This file is probably unnecessary.
    caltable_fold: location of the caltb and the output of this function
    overwrite: If True, overwrite
    
    Writes a caltable in the caltable_fold with name ".fast" appended to caltb
    '''                         
                            
    use_ant_names = True

    NSTAND = 366

    
    caltb_fast = caltb+'.fast'
    if os.path.isdir(caltb_fast) and overwrite==False:
        return caltb_fast  #### No need to generate again if exists
        
    os.system('cp -r '+ caltb + ' ' + caltb_fast)


    # The following is to overwrite the antenna list from the ms visibility (if necessary)
    # fast_ants_inp = 'LWA-015, LWA-018, LWA-038, LWA-067, LWA-085, LWA-096, LWA-131, LWA-151, LWA-180, LWA-191, LWA-260, LWA-263, LWA-264, LWA-272, LWA-274, LWA-275, LWA-277, LWA-278, LWA-281, LWA-284, LWA-285, LWA-285, LWA-286, LWA-287, LWA-289, LWA-290, LWA-295, LWA-301, LWA-304, LWA-305, LWA-307, LWA-313, LWA-314, LWA-320, LWA-321, LWA-326, LWA-327, LWA-328, LWA-329, LWA-330, LWA-334, LWA-335, LWA-336, LWA-337, LWA-338, LWA-339, LWA-340, LWA-347'.split(', ')

    # fast_ants_inp  = [i.replace('-','') for i in fast_ants_inp]
    
    tb.open(caltb+"/SPECTRAL_WINDOW")
    chan_freqs=tb.getcol("CHAN_FREQ")
    if chan_freqs.size==1:
        gcaltb=True
    else:
        gcaltb=False
    tb.close()
    
    

    msmd.open(solar_ms)
    fast_antids = msmd.antennaids()
    fast_ants = msmd.antennanames()
    nant_fast = len(fast_antids)
    nchan_fast = msmd.nchan(0)
    msmd.done()

    msmd.open(calib_ms)
    slow_antids = msmd.antennaids()
    slow_ants = msmd.antennanames()
    nchan_slow = msmd.nchan(0)
    nant_slow = len(slow_antids)

    print('Channel binning factor is ', nchan_slow/nchan_fast)
    nch_bin = nchan_slow // nchan_fast

    tb0 = table()
    tb0.open(caltb)
    bcal_antids = tb0.getcol('ANTENNA1')
    cols = tb0.colnames()

    tb1 = table()
    tb1.open(caltb_fast, nomodify=False)
    
    for i in range(nant_slow-nant_fast):
        tb1.removerows(0)

    if use_ant_names:
        antids = []
        for i, fast_ant in enumerate(fast_ants):
        #for i, fast_ant in enumerate(fast_ants_inp):
            for slow_ant in slow_ants:
                if fast_ant == slow_ant:
                    antid = msmd.antennaids(fast_ant)
                    print('found antid ', antid[0], 'for antenna', fast_ant)
                    antids.append(antid[0])


    

    fast_ants_name_wolwa = [int(i.replace('LWA','')) for i in fast_ants]
    slow_ants_name_wolwa = [int(i.replace('LWA','')) for i in slow_ants][:48]


    
    

    for col in cols:
        if col != 'WEIGHT':
            data = tb0.getcol(col)
            ndim = data.ndim
            
            print('This column {0:s} has {1:d} dimensions'.format(col, ndim))
            if ndim == 1:
                tb1.putcol(col, data[antids])
            else:
                npol, nchan, nant= data.shape
                if gcaltb:
                    data_new = data
                else:
                    data_new=np.nanmean(data.reshape(npol, nchan_fast, nch_bin, nant), axis=2)
                print('shape of data_new', data_new.shape)
                tb1.putcol(col, data_new[:, :, antids])

    # now reset the antenna ids to 0 to 47
    tb1.putcol('ANTENNA1', np.arange(nant_fast))

    tb0.close()
    tb1.close()

    if gcaltb==False:
        # Finally, overwrite the SPECTRAL_WINDOW table in the calibration table using that from the fast visibility ms
        os.system('cp -r '+ solar_ms + '/SPECTRAL_WINDOW ' + caltb_fast + '/')

    msmd.done()
    return caltb_fast
    
def gen_calibration(msfile, modelcl=None, uvrange='', bcaltb=None, logging_level='info', caltable_fold='caltables',refant='202'):
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
    bandpass(msfile, caltable=bcaltb, uvrange=uvrange, combine='scan,field,obs', fillgaps=0,refant=refant)
    logging.debug("Applying the bandpass solutions")
    applycal(vis=msfile, gaintable=bcaltb)
    logging.debug("Doing a rflag run on corrected data")
    flagdata(vis=msfile, mode='rflag', datacolumn='corrected')
    logging.debug("Finding updated and final bandpass table")
    bandpass(msfile, caltable=bcaltb, uvrange=uvrange, combine='scan,field,obs', fillgaps=0,refant=refant)

    if logging_level == 'debug':
        utils.get_flagged_solution_num(bcaltb)
    time2=timeit.default_timer()
    logging.info("Time for producing bandpass table: "+str(time2-time1)+"seconds")
    return bcaltb


def apply_calibration(msfile, gaintable=None, doantflag=False, doflag=False, antflagfile=None, do_solar_imaging=True,
                      imagename='test'):
    '''
    doantflag: If True, flag antennas based on antflagfile
    antflagfile: filenames containing list of antennas to be flagged.
    doflag: If true, run rflag on corrected data after applying the gaintable
    do_solar_imaging: use tclean to do solar imaging.
    '''
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
    #time1=timeit.default_timer()
    clearcal(msfile)
    applycal(msfile, gaintable=gaintable, flagbackup=True, applymode='calflag')
    #time2=timeit.default_timer()
    #logging.info("Time for applying bandpass table: "+str(time2-time1)+"seconds")
    if doflag == True:
        logging.debug("Running rflag on corrected data")
        flagdata(vis=msfile, mode='rflag', datacolumn='corrected')
    sunpos = utils.get_sun_pos(msfile)
    if do_solar_imaging:
        tclean(msfile, imagename=imagename, imsize=[512], cell=['1arcmin'],
               weighting='uniform', phasecenter=sunpos, niter=500)
        print('Solar image made {0:s}.image'.format(imagename))
        
        
        

def do_bandpass_correction(solar_ms, calib_ms=None, bcal=None, caltable_folder='caltables/', 
                           freqbin=1, logging_level='info', fast_vis=False, overwrite=False):
    '''
    solar_ms: Name of the MS which needs to be calibrated
    calib_ms: Name of calibration table from which calibration table is to be generated
    bcal: Name of bandpass table
    caltable_fold: Name of folder where to keep/search for the caltable. This folder should exist
    fast_vis: If True, solar_ms is a fast visibility dataset.
    bcal: Can be relative/absolute paths
    
    IMPORTANT: For slow visibility, if bcal is provided, then calib_ms need not be provided. But if fast_vis is set to True,
               then calib_ms must be a slow visibility dataset.
    '''
    
    solar_ms1 = solar_ms[:-3] + "_calibrated.ms"
    if os.path.isdir(solar_ms1):
        logging.info('Found existing calibrated dataset on disk.')
        if not overwrite:
            logging.info('I am told to not overwrite it. No bandpass correction is done.')
            return solar_ms1
        else:
            logging.info('I am told to overwrite it. Proceed with bandpass correction.')
            os.system('rm -rf '+solar_ms1)
    if calib_ms:
        if os.path.exists(calib_ms):
            if not bcal:
                bcal = caltable_folder + "/" + os.path.basename(calib_ms).replace('.ms', '.bcal')
            # check if bandpass table already exists
            if overwrite or not os.path.isdir(os.path.join(caltable_folder,os.path.basename(bcal))):
                logging.debug('Flagging antennas before calibration.')
                flagging.flag_bad_ants(calib_ms)
                bcal = gen_calibration(calib_ms, logging_level=logging_level, caltable_fold=caltable_folder)
                logging.info('Bandpass calibration table generated using ' + calib_ms)
            else:
                logging.info('I found an existing bandpass table. Will reuse it to calibrate.')
    else:
        if not bcal or not os.path.isdir(os.path.join(caltable_folder,os.path.basename(bcal))):
            print('Neither calib_ms nor bcal exists. Need to provide calibrations to continue. Abort..')
            logging.error('Neither calib_ms nor bcal exists. Need to provide calibrations to continue. Abort..')

    # correct_ms_bug(solar_ms)
    
    doantflag=True
    if bcal and fast_vis==True:
        bcal=make_fast_caltb_from_slow(calib_ms, solar_ms, os.path.join(caltable_folder,os.path.basename(bcal)))
        doantflag=False  ### The antenna flag calculation will be wrong for the fast ms.
    apply_calibration(solar_ms, gaintable=bcal, doantflag=doantflag, doflag=True, do_solar_imaging=False)
    split(vis=solar_ms, outputvis=solar_ms[:-3] + "_calibrated.ms", width=int(freqbin))
    logging.info('Splitted the input solar MS into a file named ' + solar_ms[:-3] + "_calibrated.ms")
    solar_ms = solar_ms[:-3] + "_calibrated.ms"
    return solar_ms
