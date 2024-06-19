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
from . import utils
import logging, glob, shlex, subprocess
import timeit
tb = table()
me = measures()
cl = componentlist()
msmd = msmetadata()

def run_wsclean(msfile, imagename, size:int =4096, scale='2arcmin', fast_vis=False, field=None, dry_run=False, rm_misc=True,
            predict=True, auto_pix_fov = False, telescope_size = 3200, im_fov=182*3600, pix_scale_factor=1.5,
            **kwargs):  ### uvrange is in lambda units
    """
    Wrapper for imaging using wsclean, use the parameter of wsclean in args. 
    
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
        'mem':'2',                 # fraction of memory usage
        'weight':'uniform',         # weighting scheme
        'no_dirty':'',              # don't save dirty image
        'no_update_model_required':'', # don't update model required
        'no_negative':'',           # no negative gain for CLEAN
        'niter':'10000',            # number of iterations
        'mgain':'0.8',              # maximum gain in each cycle
        'auto_threshold':'3',       # auto threshold
        'auto_mask':'8',            # auto mask
        'pol':'I',                  # polarization
        'minuv_l':'10',             # minimum uv distance in lambda
        'intervals_out':'1',        # number of output images
        'no-reorder':'',            # don't reorder the channels
        'beam_fitting_size':'2',    # beam fitting size
        'horizon_mask':"2deg",      # horizon mask distance (to mask horizon direction RFI)
        'quiet':'',                 # stop printing to stdout, save time
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
            default_kwargs['field']='all' # magic, has to be 'all', otherwise only 1st time slot has image
    else:
        default_kwargs['intervals_out']='1'
        default_kwargs['field']='all'
    if default_kwargs['intervals_out']!='1' and predict:
        raise RuntimeError("Prediction cannot be done with multiple images.")
    
    cmd_clean = "wsclean "
    # Add additional arguments from default_params
    for key, value in default_kwargs.items():
        # Convert Python-style arguments to command line format
        cli_arg = key.replace('_', '-')
        cmd_clean += f" -{cli_arg} {value} " if value != '' else f" -{cli_arg} "

    cmd_clean += " -name " + imagename + " " + msfile
    
    if not dry_run:
        time1 = timeit.default_timer()
        logging.debug(cmd_clean)
        try:
            proc=subprocess.run(shlex.split(cmd_clean))
        except Exception as e:
            proc.terminate()
            raise e

        if rm_misc:            
            for str1 in ['residual','psf']:
                os.system("rm -rf "+imagename+"*"+str1+"*.fits") 
        time2 = timeit.default_timer()
        logging.debug('Time taken for all sky imaging is {0:.1f} s'.format(time2-time1))

        if predict:
            logging.debug("Predicting model visibilities from " + imagename + " in " + msfile)
            time1 = timeit.default_timer()
            os.system("wsclean -j 1 -mem 2 -no-reorder -predict -pol "+default_kwargs['pol']+" "+ "-field all -name " + imagename + " " + msfile)
            ### if field is not all, model visibilities are predicted only for first field. Does not work with fast vis
            time2 = timeit.default_timer()
            logging.debug('Time taken for predicting the model column is {0:.1f} s'.format(time2-time1))

    return cmd_clean


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
    os.system("cp -r " + msfile + " " + outms)
    clearcal(outms, addmodel=True)
    os.system("wsclean -j 1 -mem 2 -no-reorder -predict -pol "+pol+" -field all -name " + image + " " + outms)
    #### ### if field is not all, model visibilities are predicted only for first field. Does not work with fast vis





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
           
                  
