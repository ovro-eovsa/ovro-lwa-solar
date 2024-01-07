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

import timeit
tb = table()
me = measures()
cl = componentlist()
msmd = msmetadata()

def run_wsclean(msfile, imagename,  fast_vis=False, field=None,
            **kwargs):  ### uvrange is in lambda units
    """
    Wrapper for imaging using wsclean, with the following default parameters:
    ```
        default_kwargs={
        'j':'4',                    # number of threads
        'size':'4096 4096',         # image size
        'scale':'2arcmin',          # pixel scale
        'weight':'uniform',         # weighting scheme
        'no_dirty':'',              # don't save dirty image
        'niter':'10000',            # number of iterations
        'mgain':'0.8',              # maximum gain in each cycle
        'auto_threshold':'3',       # auto threshold
        'auto_mask':'8',            # auto mask
        'pol':'I',                  # polarization
        'minuv_l':'10',             # minimum uv distance in lambda
        'intervals_out':'1',        # number of output images
        'no_reorder':'',            # don't reorder the channels
    }
    ```
    use the parameter of wsclean in args (need to replace '-' with '_' in the argument name), for example:
    ```
    run_wsclean('OVRO-60MHz.MS', 'IMG-60MHz',  size='2048 2048', niter=1000, mgain=0.9)
    ```

    :param msfile: input CASA measurement set
    :param imagename: output image name
    :param fast_vis: if True, split the measurement set into multiple measurement sets, each containing one field
    :param field: field ID to image, if fast_vis is True
    :param kwargs: additional arguments to wsclean, need to replace '-' with '_' in the argument name
    """


    logging.debug("Running WSCLEAN")
    
    default_kwargs={
        'j':'4',                    # number of threads
        'size':'4096 4096',         # image size
        'scale':'2arcmin',          # pixel scale
        'weight':'uniform',         # weighting scheme
        'no_dirty':'',              # don't save dirty image
        'niter':'10000',            # number of iterations
        'mgain':'0.8',              # maximum gain in each cycle
        'auto_threshold':'3',       # auto threshold
        'auto_mask':'8',            # auto mask
        'pol':'I',                  # polarization
        'minuv_l':'10',             # minimum uv distance in lambda
        'intervals_out':'1',        # number of output images
        'no_reorder':'',            # don't reorder the channels
    }

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
    if default_kwargs['intervals_out']!='1' and 'predict' in default_kwargs:
        raise RuntimeError("Prediction cannot be done with multiple images.")
    
    time1 = timeit.default_timer()

    cmd_clean = "wsclean "
    # Add additional arguments from default_params
    for key, value in default_kwargs.items():
        # Convert Python-style arguments to command line format
        cli_arg = key.replace('_', '-')
        cmd_clean += f" -{cli_arg} {value}" if value != '' else f" -{cli_arg}"

    cmd_clean += " -name " + imagename + " " + msfile
    
    #TODO: put -weighting in free param

    logging.debug(cmd_clean)
    os.system(cmd_clean)
    
    for str1 in ['residual','psf']:
        os.system("rm -rf "+imagename+"*"+str1+"*.fits") 
    time2 = timeit.default_timer()
    logging.debug('Time taken for all sky imaging is {0:.1f} s'.format(time2-time1))

    if default_kwargs['intervals_out']!='1':
        image_names=utils.get_fast_vis_imagenames(msfile,imagename,pol)
        for name in image_names:
            wsclean_imagename=name[0]
            final_imagename=name[1]
            os.system("mv "+wsclean_imagename+" "+final_imagename)

    if 'predict' in default_kwargs:
        logging.debug("Predicting model visibilities from " + imagename + " in " + msfile)
        time1 = timeit.default_timer()
        os.system("wsclean -predict -pol "+default_kwargs['pol']+" "+ "-name " + imagename + " " + msfile)
        time2 = timeit.default_timer()
        logging.debug('Time taken for predicting the model column is {0:.1f} s'.format(time2-time1))


def cook_wsclean_cmd(fname, mode="default", multiscale=True,
                     weight="briggs 0", mgain=0.8,
                     thresholding="-auto-mask 3 -auto-threshold 0.3",
                     len_baseline_eff=3200, FOV=14000, scale_factor=9,
                     circbeam=True, niter=5000, pol='I', data_col="DATA",
                     misc="", name=""):

    mgain_var = "-mgain {}".format(mgain)
    weight_var = "-weight "+weight
    thresholding_var = thresholding
    multiscale_var = "-multiscale" if multiscale else ""
    circbeam_var = "-circularbeam" if circbeam else ""
    pol_var = "-pol "+pol
    data_col_var = "-data-column "+data_col

    msmd = msmetadata()
    msmd.open(fname)
    freqcenter = msmd.chanfreqs(0)
    msmd.close()

    freq = np.median(freqcenter)

    scale = 1.22*(3e8/freq)/len_baseline_eff * 180/np.pi*3600 / scale_factor
    scale_var = "-scale {}asec".format(scale)
    size_var = "-size {} {}".format(int(FOV/scale), int(FOV/scale))

    clean_cmd = ("wsclean -no-reorder -no-update-model-required  " + mgain_var + 
                 " " + weight_var + " " + multiscale_var + " " + thresholding_var + " " + 
                 size_var + " " + scale_var + " " + pol_var + " " + data_col_var + " "
                 + " " + circbeam_var + " " + misc +
                 " -niter {} -name "+ name).format(niter)

    return clean_cmd


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
    os.system("wsclean -predict -pol "+pol+" -name " + image + " " + outms)





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
           
                  
