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
import utils,flagging,calibration,selfcal,source_subtraction
import logging, glob
from file_handler import File_Handler
from primary_beam import analytic_beam as beam 
import primary_beam
from generate_calibrator_model import model_generation
import generate_calibrator_model
tb = table()
me = measures()
cl = componentlist()
msmd = msmetadata()

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

def get_flux_scaling_factor(msfile, src_area=100, min_beam_val=0.1,pol='I'):
    if pol=='I':
        md=model_generation(vis=msfile,pol=pol,separate_pol=False)
    else:
        md=model_generation(vis=msfile)
    md.predict=False
    md.model=False
    md.min_beam_val=min_beam_val
    modelcl, ft_needed = md.gen_model_cl()
    if pol=='I':
        images = glob.glob(msfile[:-3] + "_self*-image.fits")
    else:
        images= glob.glob(msfile[:-3] + "_self*XX-image.fits")
    num_image = len(images)
    
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
    
        srcs = source_subtraction.get_nonsolar_sources_loc_pix(msfile, final_image, min_beam_val=min_beam_val)
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
        break
        
    msmd.open(msfile)
    ref_freqmhz = msmd.meanfreq(0) * 1e-6
    msmd.done()
    
    mean_factor=[]
    pol_prefix=[]
    srcs_with_scaling = []
    scaling_factor = []
    for s in srcs:
        src_x = s['xpix']
        src_y = s['ypix']
        bbox = [[src_y - src_area_ypix // 2, src_y + src_area_ypix // 2],
                [src_x - src_area_xpix // 2, src_x + src_area_xpix // 2]]
        for pola in ['I','XX','YY']:
            if pol=='I' and pola=='I':
                prefix=''
            elif pola=='XX' and pol!='I':
                prefix='-XX'
            elif pola=='YY' and pol!='I':
                prefix='-YY'
            else:
                continue
           
        
        

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
            logging.info('Image flux of ' + s['label'] + ' is  ' + str(image_flux))
            # print (image_flux)
            print(s['label'], image_flux, model_flux)
            s['model_flux'] = model_flux
            s['image_flux'] = image_flux
            s['ref_freqmhz'] = ref_freqmhz
            
            if (model_flux > 0 and image_flux > 0):
                try:
                    s['scaling_factor'][pola] = model_flux / image_flux
                except KeyError:
                    s['scaling_factor']={}
                    s['scaling_factor'][pola] = model_flux / image_flux
                logging.info('Scaling factor obtained from ' + s['label'] + ' is ' + str(model_flux / image_flux))
            else:
                logging.warning('Scaling factor is not calculated for ' + s[
                    'label'] + ' as either/both model and image flux is negative')
                    
        srcs_with_scaling.append(s)
    return srcs_with_scaling

def correct_flux_scaling(msfile, src_area=100, min_beam_val=0.1, caltable_suffix='fluxscale',pol='I'):
    import glob
    
    os.system("rm -rf calibrator-model.fits calibrator-XX-model.fits calibrator-YY-model.fits")
    mstime_str = utils.get_timestr_from_name(msfile)
    di_selfcal_str, success = utils.get_keyword(msfile, 'di_selfcal_time', return_status=True)

    if di_selfcal_str == mstime_str and success:
        '''
        if pol=='I':
            images = glob.glob(msfile[:-3] + "_self*-image.fits")
        else:
            images= glob.glob(msfile[:-3] + "_self*XX-image.fits")
            
        num_image = len(images)
        
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
       # if os.path.isfile(final_image)==False:
       #     continue
        '''
        srcs_with_scaling = get_flux_scaling_factor(msfile, src_area=src_area, min_beam_val=min_beam_val,pol=pol)
            
        scaling_factor=[]
        for s in srcs_with_scaling:
            for pola in ['I','XX','YY']:
                try:
                    scaling_factor.append(s['scaling_factor'][pola])
                except KeyError:
                    pass
        
        pol_prefix=[]
        mean_factor=[]        
        if len(scaling_factor) > 0:
            mean_factor.append(np.mean(np.array(scaling_factor)))
            print(scaling_factor)
            print(mean_factor)
            for pola in ['I','XX','YY']:
                if pol=='I' and pola=='I':
                    prefix=''
                elif pola=='XX' and pol!='I':
                    prefix='-XX'
                elif pola=='YY' and pol!='I':
                    prefix='-YY'
                else:
                    continue
                images = msfile[:-3] + "_self*" + prefix+"-image.fits"
                if len(glob.glob(images))==0:
                    continue
       
            if prefix=='-XX':
                pol_prefix.append('X')
            elif prefix=='-YY':
                pol_prefix.append('Y')
            else:
                pol_prefix.append('')
        logging.info('Scaling factor is for :' + str(mean_factor))

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
