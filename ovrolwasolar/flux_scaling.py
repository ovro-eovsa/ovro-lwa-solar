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
from .  import utils,flagging,calibration,selfcal,source_subtraction
import logging, glob
from .file_handler import File_Handler
from .primary_beam import analytic_beam as beam 
from . import primary_beam
from .generate_calibrator_model import model_generation
from . import generate_calibrator_model
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
    
class flux_scaling():
    def __init__(self,vis,src_area=100, min_beam_val=0.1, caltable_suffix='fluxscale',pol='I', fast_vis=False,calib_ms=None):
        self.msfile=vis
        self.src_area=src_area
        self.min_beam_val=min_beam_val
        self.caltable_suffix=caltable_suffix
        self.pol=pol
        self.fast_vis=fast_vis
        self.calib_ms=calib_ms
        
        if self.pol=='I':
            self.num_images=len(glob.glob(self.msfile[:-3] + "_self*-image.fits"))
            
        elif self.pol=='XX,YY':
            self.num_images=len(glob.glob(self.msfile[:-3] + "_self*-XX-image.fits"))
        else:
            raise RuntimeError("Only \'I\' and \'XX,YY\' are supported") 
            
        msmd.open(self.msfile)
        self.ref_freqmhz = msmd.meanfreq(0) * 1e-6
        msmd.done()
            
    def correct_flux_scaling(self):
        os.system("rm -rf calibrator-model.fits calibrator-XX-model.fits calibrator-YY-model.fits")
        mstime_str = utils.get_timestr_from_name(self.msfile)
        di_selfcal_str, success = utils.get_keyword(self.msfile, 'di_selfcal_time', return_status=True)
        
        
        if di_selfcal_str == mstime_str and success and self.fast_vis==False:
            self.get_flux_scaling_factor()
        
            if self.pol=='I':
                scaling_factor=[]
                pol_prefix=''
            else:
                scaling_factor=[[],[]]
                pol_prefix='X,Y'
            
            
            source_present=False
                
            for s in self.srcs_with_scaling:
                if self.pol=='I':
                    if 'I' in s['scaling_factor']:
                        scaling_factor.append(s['scaling_factor']['I'])
                        source_present=True
                else:
                    for j,pol in enumerate(['XX','YY']):
                        if pol in s['scaling_factor']:
                            scaling_factor[j].append(s['scaling_factor'][pol])
                            source_present=True
                    
                        
            caltable = self.msfile[:-3] + "." + self.caltable_suffix
            
           
            if source_present==True:
                
                if type(scaling_factor[0])==list:
                    mean_factor=np.zeros(2)
                    if len(scaling_factor[0])>0:
                        mean_factor[0]=np.mean(scaling_factor[0])
                        logging.debug('Scaling factor for X pol is for :' + str(mean_factor[0]))
                    else:
                        logging.warning("Flux scaling factor could not be found for X pol")
                        mean_factor[0]=1
                    if len(scaling_factor[1])>0:
                        mean_factor[1]=np.mean(scaling_factor[1])
                        logging.debug('Scaling factor for Y pol is for :' + str(mean_factor[1]))
                    else:
                        logging.warning("Flux scaling factor could not be found for Y pol")
                        mean_factor[1]=1
                else:
                    mean_factor=np.mean(scaling_factor)                
                    logging.debug('Scaling factor is for :' + str(mean_factor))
                
                print(mean_factor)
                

                logging.debug("Generating caltable for fluxscaling. Filename is " + self.msfile[:-3] + "." + self.caltable_suffix)

                
                gencal(vis=self.msfile, caltable=caltable, caltype='amp', parameter=np.sqrt(1. /mean_factor ),pol=pol_prefix)
                
                os.system("cp -r " + caltable + " caltables/")
            else:
                gencal(vis=self.msfile, caltable=caltable, caltype='amp', parameter=1.0)
                logging.warning("Could not find appropriate flux scaling factor. No correction will be done.")
                os.system("cp -r " + caltable + " caltables/")
        
        elif success == True:
            caltable = glob.glob("caltables/" + di_selfcal_str + "*.fluxscale")[0]
            if self.fast_vis==True:
            	caltable=calibration.make_fast_caltb_from_slow(self.calib_ms, self.msfile, caltable)
            logging.debug("Applying {0:s} for doing fluxscaling".format(caltable))
        else:
            caltable = self.msfile[:-3] + "." + self.caltable_suffix
            gencal(vis=self.msfile, caltable=caltable, caltype='amp', parameter=1)
            logging.warning("Could not find appropriate flux scaling factor. No correction will be done.")

        
        DI_val = utils.get_keyword(self.msfile, 'di_selfcal_time')

        logging.debug('Correcting the DATA with the scaling factor')
        temp_file = 'temp_' + os.path.basename(self.msfile)

        split(vis=self.msfile, outputvis=temp_file)

        applycal(vis=temp_file, gaintable=caltable, calwt=False)

        os.system("rm -rf " + self.msfile + "*")

        split(vis=temp_file, outputvis=self.msfile)
        os.system("rm -rf " + temp_file + "*")
        utils.put_keyword(self.msfile, 'di_selfcal_time', DI_val)
        
        return    
            
  
    def get_image_props(self,final_image):
    
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
        self.src_area_xpix = self.src_area / dx
        self.src_area_ypix = self.src_area / dy          
            
    def get_flux_scaling_factor(self, use_cascyg_only=True):
        if self.pol=='I':
            md=model_generation(vis=self.msfile,pol=self.pol,separate_pol=False)
        else:
            md=model_generation(vis=self.msfile)
            
        md.predict=False
        md.model=False
        md.min_beam_val=self.min_beam_val
        modelcl, ft_needed = md.gen_model_cl()
        
        if self.pol=='I':
            final_image = self.msfile[:-3] + "_self" + str(self.num_images - 1) +"-image.fits"
        else:
            final_image = self.msfile[:-3] + "_self" + str(self.num_images - 1) +"-XX-image.fits"
            
        srcs = source_subtraction.get_nonsolar_sources_loc_pix(self.msfile, final_image, min_beam_val=self.min_beam_val)

        if use_cascyg_only:
            srcs_name = [s['label'] for s in srcs]
            if 'CasA' in srcs_name or 'CygA' in srcs_name:
                for i, s in enumerate(srcs):
                    if s['label'] != 'CasA' and s['label'] != 'CygA':
                        del(srcs[i])
            else:
                print('I did not find either CasA or CygA in the image. Use all sources available')
                print(srcs_name)


        self.get_image_props(final_image)

        if self.pol=='I':
            pols=['']
        else:
            pols=['-XX','-YY']
        
        
        self.srcs_with_scaling = []

        
        for s in srcs:
            s['scaling_factor']={}
        
        for s in srcs:
            src_x = s['xpix']
            src_y = s['ypix']
            bbox = [[src_y - self.src_area_ypix // 2, src_y + self.src_area_ypix // 2],
                    [src_x - self.src_area_xpix // 2, src_x + self.src_area_xpix // 2]]    
                    
            for prefix in pols:
                final_image = self.msfile[:-3] + "_self" + str(self.num_images - 1) + prefix+"-image.fits"
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
                logging.debug('Model flux of ' + s['label'] + ' is  ' + str(model_flux))
                image_flux = imstat(imagename=final_image, box=str(src_x - self.src_area_xpix // 2) + "," + \
                                                               str(src_y - self.src_area_ypix // 2) + "," + \
                                                               str(src_x + self.src_area_xpix // 2) + "," + \
                                                               str(src_y + self.src_area_ypix // 2))['flux'][0]
                logging.debug('Image flux of ' + s['label'] + ' is  ' + str(image_flux))
                # print (image_flux)
                print(s['label'], image_flux, model_flux)
                s['model_flux'] = model_flux
                s['image_flux'] = image_flux
                s['ref_freqmhz'] = self.ref_freqmhz
                
                if (model_flux > 0 and image_flux > 0):
                    
                    if prefix=='':
                        s['scaling_factor']['I'] = model_flux / image_flux
                    else:
                        s['scaling_factor'][prefix[1:]] = model_flux / image_flux
                    logging.debug('Scaling factor obtained from ' + s['label'] + ' is ' + str(model_flux / image_flux))
                    
                else:
                    logging.warning('Scaling factor is not calculated for ' + s[
                        'label'] + ' as either/both model and image flux is negative')
            self.srcs_with_scaling.append(s)
        

