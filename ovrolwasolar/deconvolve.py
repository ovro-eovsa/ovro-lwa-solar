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

def run_wsclean(msfile, imagename, imsize=4096, cell='2arcmin', uvrange='10', niter=10000,
                mgain=0.8, do_automask=True, automask_thresh=8, do_autothresh=False, autothreshold_rms=3, 
                predict=True, pol='I', fast_vis=False, intervals_out=None, field=None,**kwargs):  ### uvrange is in lambda units
    """
    Wrapper for imaging using wsclean
    :param msfile: input CASA measurement set
    :param imagename: output image name
    :param imsize: size of the image in pixels
    :param cell: pixel scale
    :param niter: number of iterations
    :param uvrange: uvrange following tclean's syntax
    :param do_automask: whether or not to do automask
    :param do_autothresh: whether or not to use local RMS thresholding
    :param mgain: maximum gain in each major cycle (during every major iteration, the peak is reduced by the given factor)
    
    Example of using kwargs
    
    kwargs={'channels-out':'10','spws':'5,6'}
    deconvolve.run_wsclean(msfile,imagename=imagename,**kwargs)
    """
    logging.debug("Running WSCLEAN")
    if fast_vis==True:
        if field is None:
           intervals_out=1
           field='all'
        else: 
            intervals_out=len(field.split(','))
    else:
        intervals_out=1
        field='all'
    if intervals_out!=1 and predict:
        raise RuntimeError("Prediction cannot be done with multiple images.")
        
    if do_automask:
        automask_handler = " -auto-mask " + str(automask_thresh)
    else:
        automask_handler = ""

    if do_autothresh:
        autothresh_handler = " -local-rms -auto-threshold " + str(autothreshold_rms)
    else:
        autothresh_handler = ""

    time1 = timeit.default_timer()

    cmd_str1=''
    for cmd,val in zip(kwargs,kwargs.values()):
        cmd_str1+='-'+cmd+" "+val+" "    
    
    os.system("wsclean -j 4 -no-dirty -no-update-model-required -no-negative -size " + str(imsize) + " " + \
              str(imsize) + " -scale " + cell + " -weight uniform -minuv-l " + str(uvrange) + " -name " + imagename + \
              " -niter " + str(niter) + " -mgain " + str(mgain) + \
              automask_handler + autothresh_handler + \
              " -beam-fitting-size 2 -pol " + pol + ' ' + "-intervals-out "+ \
              str(intervals_out) + " -field " + field + " " + cmd_str1+msfile)
    
    for str1 in ['residual','psf']:
        os.system("rm -rf "+imagename+"*"+str1+"*.fits") 
    time2 = timeit.default_timer()
    logging.debug('Time taken for all sky imaging is {0:.1f} s'.format(time2-time1))

    
    if intervals_out!=1:
        image_names=utils.get_fast_vis_imagenames(msfile,imagename,pol)
        for name in image_names:
            wsclean_imagename=name[0]
            final_imagename=name[1]
            os.system("mv "+wsclean_imagename+" "+final_imagename)

    if predict:
        logging.debug("Predicting model visibilities from " + imagename + " in " + msfile)
        time1 = timeit.default_timer()
        os.system("wsclean -predict -pol "+pol+" "+ "-name " + imagename + " " + msfile)
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
           
                  