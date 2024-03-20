from casatasks import clearcal, ft, bandpass, applycal, flagdata, tclean, flagmanager, uvsub, gaincal, split, imstat, \
    gencal
from casatools import table, measures, componentlist, msmetadata
import math
import sys, os, time
import numpy as np
from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.wcs import WCS
from astropy.io import fits
import matplotlib.pyplot as plt
from . import utils,flagging,calibration,selfcal,source_subtraction
import logging, glob
from . import utils,flagging
from .file_handler import File_Handler
from .primary_beam import analytic_beam as beam 
from . import primary_beam
from .generate_calibrator_model import model_generation
from . import generate_calibrator_model
from . import polcalib

import timeit
tb = table()
me = measures()
cl = componentlist()
msmd = msmetadata()

def run_wsclean(msfile, imagename, size:int =4096, scale='2arcmin', fast_vis=False, field=None,
            predict=True, auto_pix_fov = False, telescope_size = 3200, im_fov=182*3600, pix_scale_factor=1.5,
            **kwargs):  ### uvrange is in lambda units
    """
    Wrapper for imaging using wsclean, use the parameter of wsclean in args. 
    
    Additional features: For polarized clean, if join_polarizations is not in kwargs,
                        join_polarizations will be added by default to the WSClean call.
                        
                        If pol == either of ['I','XX','YY','XX,YY'] and no_negative not in 
                        kwargs, no_negative keyword will be added to WSClean call
    
    To be noted:

    * replace '-' with '_' in the argument name), 
    * The args without value is set to True or False, for example.
    * add False to a key to remove it from the args list.

    ``run_wsclean('OVRO-60MHz.MS', 'IMG-60MHz',  size=2048, niter=1000, mgain=0.9, no_reorder=True, predict=True)``
    
    :param msfile: input CASA measurement set
    :param imagename: output image name
    :param size: (int) image size, default 4096
    :param scale: pixel scale, default 2arcmin
    :param fast_vis: if True, split the measurement set into multiple measurement sets, each containing one field
    :param field: field ID to image, if fast_vis is True
    :param predict: if True, predict the model visibilities from the image
    :param auto_pix_fov: if True, automatically set the pixel scale to match the field of view
    :param telescope_size: size of the telescope in meters, default 3200 (OVRO-LWA)
    :param im_fov: field of view of the image in arcseconds, default 182*3600asec (full sky+ 2deg)
    :param j: number of threads, default 4
    :param mem: fraction of max memory usage, default 2 
    :param weight: weighting scheme, default uniform
    :param no_dirty: don't save dirty image, default True
    :param niter: number of iterations, default 10000
    :param mgain: maximum gain in each cycle, default 0.8
    :param auto_threshold: auto threshold, default 3
    :param auto_mask: auto mask, default 8
    :param pol: polarization, default I
    :param minuv_l: minimum uv distance in lambda, default 10
    :param intervals_out: number of output images, default 1
    :param no_reorder: don't reorder the channels, default True
    """


    logging.debug("Running WSCLEAN")
    
    default_kwargs={
        'j':'1',                    # number of threads
        'mem': '2',		            # fraction of memory usage	
        'weight':'briggs 0',         # weighting scheme
        'no_dirty':'',              # don't save dirty image
        'no_update_model_required':'', # don't update model required
        'niter':'10000',            # number of iterations
        'mgain':'0.8',              # maximum gain in each cycle
        'auto_threshold':'5',       # auto threshold
        'auto_mask':'8',            # auto mask
        'pol':'I',                  # polarization
        'minuv_l':'10',             # minimum uv distance in lambda
        'intervals_out':'1',        # number of output images
        'no-reorder':'',            # don't reorder the channels
        'beam_fitting_size':'2',    # beam fitting size
        'horizon_mask':"2deg"    # horizon mask distance (to mask horizon direction RFI)
    }
    
    
    if auto_pix_fov:
        msmd = msmetadata()
        msmd.open(msfile)
        freqcenter = msmd.chanfreqs(0)
        msmd.close()
        freq = np.median(freqcenter)

        scale_num = 1.22*(3e8/freq)/telescope_size * 180/np.pi*3600 / pix_scale_factor
        scale = str(scale_num/60)+'arcmin'
        size = find_smallest_fftw_sz_number(im_fov/scale_num)
        logging.debug("Auto pixel scale: " + scale+ ", size: " + str(size)+ "pix, at freq:" + str(freq/1e6) + "MHz")

    default_kwargs['size']=str(size)+' '+str(size)
    default_kwargs['scale']=scale

    # remove the key if val is False from kwargs
    for key, value in kwargs.items():
        if value is False:
            default_kwargs.pop(key, None)
        elif value is True:
            # Add the key with an empty string as value if True
            default_kwargs[key] = ''
        else:
            default_kwargs[key] = str(value)


    if fast_vis==True:
        if field is None:
            default_kwargs['intervals_out']='1'
            default_kwargs['field']='all'
        else:
            default_kwargs["intervals_out"] =str(len(field.split(',')))
    else:
        default_kwargs['intervals_out']='1'
        default_kwargs['field']='all'
    if default_kwargs['intervals_out']!='1' and predict:
        raise RuntimeError("Prediction cannot be done with multiple images.")
    
    time1 = timeit.default_timer()

    cmd_clean = "wsclean "
    # Add additional arguments from default_params
    for key, value in default_kwargs.items():
        # Convert Python-style arguments to command line format
        cli_arg = key.replace('_', '-')
        cmd_clean += f" -{cli_arg} {value} " if value != '' else f" -{cli_arg} "
        
    if ('I' in default_kwargs['pol']) and ('join_polarizations' not in default_kwargs.keys()) and \
            ('Q' in default_kwargs['pol'] or 'U' in default_kwargs['pol'] \
            or 'V' in default_kwargs['pol']):
        cmd_clean+= " -join-polarizations "
    
    elif (default_kwargs['pol']=='I' or default_kwargs['pol']=='XX' or default_kwargs['pol']=='YY' \
            or default_kwargs['pol']=='XX,YY') and ('no_negative' not in default_kwargs.keys()):
        cmd_clean+= " -no-negative " 
        
    cmd_clean += " -name " + imagename + " " + msfile
    
    

    logging.debug(cmd_clean)
    os.system(cmd_clean)
    
    for str1 in ['residual','psf']:
       os.system("rm -rf "+imagename+"*"+str1+"*.fits") 
    time2 = timeit.default_timer()
    logging.debug('Time taken for all sky imaging is {0:.1f} s'.format(time2-time1))

    if default_kwargs['intervals_out']!='1':
        image_names=utils.get_fast_vis_imagenames(imagename,pol=pol,msfile=msfile)
        for name in image_names:
            wsclean_imagename=name[0]
            final_imagename=name[1]
            os.system("mv "+wsclean_imagename+" "+final_imagename)

    if predict:
        enforce_threshold_on_model(msfile,imagename,pol=default_kwargs['pol'])
        logging.debug("Predicting model visibilities from " + imagename + " in " + msfile)
        time1 = timeit.default_timer()
        os.system("wsclean -predict -pol "+default_kwargs['pol']+" "+ "-name " + imagename + \
                " -j "+default_kwargs['j']+" -mem "+default_kwargs['mem']+" " + msfile)
        time2 = timeit.default_timer()
        logging.debug('Time taken for predicting the model column is {0:.1f} s'.format(time2-time1))

def enforce_threshold_on_model(imagename,thresh=7,pol='I',src_area=100, msfile=None, \
                                sol_area=400., neg_thresh=1.5, enforce_polarised_beam_thresholding=False):
    '''
    imagename is the prefix of the image and is same as that supplied to the WSClean call
    This function will first determine the pixels for which the Stokes I/XX/YY/RR/LL image is less than 
    than thresh x rms . Then it will go to the polarised models and put all such pixels to
    be zeros. If these images are not found, this the function will give a warning to log file
    and exit.
    
    :param imagename: Imagename prefix. This is same as that passed to WSclean call
    :type imagename: str
    :param thresh: Threshold used to determine low SNR pixels in Stokes I/XX/YY image. This
                    is in units of rms. Default: 7
    :type thresh: float
    :param pol: This is the list of polarisations on which the thresholding is done. Either
                I,XX,YY,RR,LL is necessary to do this thresholding. Format='I,Q,U,V'
    :type pol: str
    :param src_area: Diameter of region around the sources other than source considered for
                    estimating rms and negatives. Value in arcminutes. Default:100
    :type src_area: float
    :param msfile: Name of MS. Is used to determine az-el of sun at the time of image
    :type msfile: str
    :param sol_area: Diameter of region around Sun considered for
                    estimating rms and negatives. Value in arcminutes. Default:400
    :type sol_area: float
    :param neg_thresh: Stokes I values which are smaller than neg_thresh x abs(min) are flagged.
                        Default: 1.5
    :type neg_thresh: float
    '''
    pols=pol.split(',')
    num_pol=len(pols)
    print (pol)
    
    #### check if either of I,XX,YY,RR,LL is in pols or not
    if set(['I','XX','YY',"RR","LL"]).isdisjoint(pols):
        logging.warning("Intensity or pseudo-intensity image not in pols. Threshold could not be done.")
        return
    
    if num_pol==1:
        Iimage=imagename+"-image.fits"
    else:
        for pol1 in ['I','XX','YY','RR','LL']:
            Iimage=imagename+"-"+pol1+"-image.fits"
            if os.path.isfile(Iimage):
                break
    if not os.path.isfile(Iimage):
        logging.warning("Intensity or pseudo-intensity image not found. Threshold could not be done.")
        return
        
    Idata=fits.getdata(Iimage)
    head=fits.getheader(Iimage)
    
    freq=head['CRVAL3']*1e-6
    
    if msfile is not None:
        solx, soly = utils.get_solar_loc_pix(msfile, Iimage)
    else:
        solx,soly=None, None
    
    beam_factors=None
    
    
    
    if head['cunit1'] == 'deg':
        dx = np.abs(head['cdelt1'] * 60.)
    else:
        print(head['cunit1'] + ' not recognized as "deg". Model could be wrong.')
    if head['cunit2'] == 'deg':
        dy = np.abs(head['cdelt2'] * 60.)
    else:
        print(head['cunit2'] + ' not recognized as "deg". Model could be wrong.')
        
    if solx is not None:
        sol_area_xpix = int(sol_area / dx)
        sol_area_ypix = int(sol_area / dy)
        solar_data=Idata[0, 0, soly - sol_area_ypix // 2:soly + sol_area_ypix // 2 + 1,
                        solx - sol_area_xpix // 2:solx + sol_area_xpix // 2 + 1]
        if enforce_polarised_beam_thresholding:
            beam_factors=np.array([[1.0 , 0.30,-0.0844,-0.0175]])#polcalib.get_beam_factors([solar_el],[solar_az],freq=freq)
        
    rms=np.nanstd(Idata)
    pos=np.where(Idata>10*rms)
    ### removed outliers, which are actual sources. This will stop them from biasing the rms calculation.
    Idata[pos]=np.nan
    
    rms=np.nanstd(Idata)
    Idata_copy=np.zeros_like(Idata)
    Idata_copy[...]=Idata[...]
    pos2=np.where(Idata>=thresh*rms)
    
    for src_y,src_x in zip(pos2[2],pos2[3]):
        src_area_xpix = src_area / dx
        src_area_ypix = src_area / dy
        
        new_data = Idata[0,0,int(src_y - src_area_ypix // 2): int(src_y + src_area_ypix // 2),\
                    int(src_x - src_area_xpix // 2): int(src_x + src_area_xpix // 2)]
        max_data=np.nanmax(new_data)
        min_data=np.nanmin(new_data)
        if max_data<neg_thresh*abs(min_data):          
            Idata_copy[0,0,int(src_y - src_area_ypix // 2): int(src_y + src_area_ypix // 2),\
                    int(src_x - src_area_xpix // 2): int(src_x + src_area_xpix // 2)]=0.0
                    
    if solx is not None:
        Idata_copy[0, 0, soly - sol_area_ypix // 2:soly + sol_area_ypix // 2 + 1,
                solx - sol_area_xpix // 2:solx + sol_area_xpix // 2 + 1]=solar_data      
            
    pos=np.where(Idata_copy<neg_thresh*rms)
    

    
    for pol1 in pols:
        if num_pol==1:
            modelname=imagename+"-model.fits"
        else:
            modelname=imagename+"-"+pol1+"-model.fits"
        
        if not os.path.isfile(modelname):
            continue
        
        hdu=fits.open(modelname,mode='update')
        try:
            hdu[0].data[pos]=0.00000000
            if enforce_polarised_beam_thresholding and beam_factors is not None \
                    and pol1 not in ['XX','YY','RR','LL','I']:
                enforce_polarised_beam_threshold(hdu[0].data,Idata_copy,beam_factors, pol1)
                
            hdu.flush()
        finally:
            hdu.close()
    return
    
def enforce_polarised_beam_threshold(stokes_data, Idata, beam_factors, stokes,thresh=1.5):
    '''
    :param_stokes_data: Stokes data. Modified inplace
    :type stokes_data: numpy ndarray
    :param Idata: Stokes I data
    :type Idata: numpy ndarray
    :param beam_factors: Primary beam values for different stokes parameters, normalised to I value
    :type beam_factors: numpy ndarray
    :param stokes: Stokes parameter of stokes data
    :type stokes: str
    :param thresh: Values above thresh x primary_beam_value x I model is not zeroed.
    :type thresh: float 
    '''
    if stokes=='Q':
        factor=beam_factors[1]
    elif stokes=='U':
        factor=beam_factors[2]
    elif stokes=='V':
        factor=beam_factors[3]
    elif stokes in ['I','XX','YY','RR','LL']:
        factor=0.0  ### do nothing for Stokes I data
    else:
        logging.warning("Only Q,U,V supported.")
        return
    
    threshold=Idata*factor
    
    stokes_data[abs(stokes_data)<abs(threshold*thresh)]=0.0
    stokes_data[abs(stokes_data)>abs(threshold*thresh)]-=threshold
    return
    
    

def find_smallest_fftw_sz_number(n):
    """
    Find the smallest number that can be decomposed into 2,3,5,7
    
    :param n: input number
    :return: the smallest number that can be decomposed into 2,3,5,7
    """

    max_a = int(np.ceil(np.log(n) / np.log(2)))
    max_b = int(np.ceil(np.log(n) / np.log(3)))
    max_c = int(np.ceil(np.log(n) / np.log(5)))
    max_d = int(np.ceil(np.log(n) / np.log(7)))

    smallest_fftw_sz = float('inf')
    for a in range(max_a + 1):
        for b in range(max_b + 1):
            for c in range(max_c + 1):
                for d in range(max_d + 1):
                    fftw_sz = (2 ** a) * (3 ** b) * (5 ** c) * (7 ** d)
                    if fftw_sz > n and fftw_sz < smallest_fftw_sz:
                        smallest_fftw_sz = int(fftw_sz)
    return smallest_fftw_sz


def predict_model(msfile, outms, image="_no_sun",pol='I'):
    """
    Predict a model measurement set from an image. In the pipeline, it is
    used for transforming a model all sky image without the Sun to the output ms, and write it into the model column
    :param msfile: input CASA measurement set
    :param outms: output CASA measurement set
    :param image: input all sky image with non-solar sources, generated by gen_nonsolar_source_model()
    :return: N/A, but with an output CASA measurement set written into the same area as in the input ms
    """
    enforce_threshold_on_model(image,pol=pol,msfile=msfile)
    os.system("cp -r " + msfile + " " + outms)
    clearcal(outms, addmodel=True)
    os.system("wsclean -predict -pol "+pol+" -name " + image + " -j 1 -mem 2 " + outms)





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
    sunpos = utils.get_sun_pos(msfile)
    tclean(msfile, imagename=imagename, uvrange=uvrange, imsize=imsize, cell=cell,
           weighting='uniform', phasecenter=sunpos, niter=niter, psfcutoff=psfcutoff)
           
                  
