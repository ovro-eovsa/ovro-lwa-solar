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
from . import utils,flagging,calibration,selfcal,deconvolve
import logging, glob
from .generate_calibrator_model import model_generation
from .  import generate_calibrator_model
tb = table()
me = measures()
cl = componentlist()
msmd = msmetadata()

def get_nonsolar_sources_loc_pix(msfile, image="allsky", verbose=False, min_beam_val=1e-6):
    """
    Converting the RA & DEC coordinates of nonsolar sources to image coordinates in X and Y

    :param image: input CASA image
    :return: an updated directionary of strong sources with 'xpix' and 'ypix' added
    """
    from astropy.wcs.utils import skycoord_to_pixel
    srcs = utils.get_strong_source_list()
    tb.open(msfile)
    t0 = tb.getcell('TIME', 0)
    tb.close()
    # me.set_data_path('/opt/astro/casa-data')
    ovro = me.observatory('OVRO_MMA')
    time = me.epoch('UTC', '%fs' % t0)
    me.doframe(ovro)
    me.doframe(time)
    

    for i in range(len(srcs) - 1, -1, -1):
        src = srcs[i]
        coord = src['position'].split()
        d0 = None
        if len(coord) == 1:
            d0 = me.direction(coord[0])
            d0_j2000 = me.measure(d0, 'J2000')
            src['position'] = 'J2000 %frad %frad' % (d0_j2000['m0']['value'], d0_j2000['m1']['value'])
        elif len(coord) == 3:
            coord[2] = generate_calibrator_model.conv_deg(coord[2])
            d0 = me.direction(coord[0], coord[1], coord[2])
            d0_j2000 = me.measure(d0, 'J2000')
        else:
            raise Exception("Unknown direction")
        d = me.measure(d0, 'AZEL')
        elev = d['m1']['value']*180/np.pi
        az=d['m0']['value']*180/np.pi
        scale=np.sin(elev*np.pi/180)**1.6  ### sufficient for doing this check
        if elev > 0 and scale > min_beam_val:
            ra = d0_j2000['m0']['value']
            dec = d0_j2000['m1']['value']
            coord = SkyCoord(ra * u.rad, dec * u.rad, frame='icrs')
            
            head = fits.getheader(image)
            
            w = WCS(head)
            pix = skycoord_to_pixel(coord, w)
            x = int(pix[0])
            y = int(pix[1])
            srcs[i]['xpix'] = x
            srcs[i]['ypix'] = y
            logging.debug('Found source {0:s} at pix x {1:d}, y {2:d}'.format(srcs[i]['label'], x, y))
            if verbose:
                print('Found source {0:s} at pix x {1:d}, y {2:d}'.format(srcs[i]['label'], x, y))
        else:
            logging.debug('Source {0:s} has a <0 elevation or very low gain'.format(srcs[i]['label']))
            if verbose:
                print('Source {0:s} has a <0 elevation or very low gain'.format(srcs[i]['label']))
            del srcs[i]
    return srcs


def gen_nonsolar_source_model(msfile, imagename="allsky", outimage=None, sol_area=400., src_area=200.,
                              remove_strong_sources_only=True, verbose=True, pol='I'):
    """
    Take the full sky image, remove non-solar sources from the image

    :param msfile: path to CASA measurement set
    :param imagename: input all sky image
    :param outimage: output all sky image without other sources
    :param sol_area: size around the Sun in arcmin to be left alone
    :param src_area: size around the source to be taken away
    :param remove_strong_sources_only: If True, remove only known strong sources.
        If False, remove everything other than Sun.
    :param verbose: Toggle to print out more information
    :param pol: specifies which pol it is. should be 'I,Q,U,V,XX,YY'. It is 
                important that the entries are comma separated.
    :return: FITS image with non-solar sources removed
    """
    imagename1=imagename 
    pols=pol.split(',')
    num_pol=len(pols)
    if num_pol==1:
        imagename=imagename+"-image.fits"
    else:
        imagename=imagename+"-"+pols[0]+"-image.fits"
    if os.path.isfile(imagename)==False:
        logging.error("Image does not exist.")
        raise IOError("Image does not exist")
        
    solx, soly = utils.get_solar_loc_pix(msfile, imagename)
    srcs = get_nonsolar_sources_loc_pix(msfile, imagename)
    
    head = fits.getheader(imagename)
    if head['cunit1'] == 'deg':
        dx = np.abs(head['cdelt1'] * 60.)
    else:
        print(head['cunit1'] + ' not recognized as "deg". Model could be wrong.')
    if head['cunit2'] == 'deg':
        dy = np.abs(head['cdelt2'] * 60.)
    else:
        print(head['cunit2'] + ' not recognized as "deg". Model could be wrong.')
   
    imagename=imagename1
    for pola in pols:
        if num_pol==1:
            prefix=''
        else:
            prefix="-"+pola
        data = fits.getdata(imagename + prefix+"-model.fits")
        head=fits.getheader(imagename + prefix+"-model.fits")
        if remove_strong_sources_only:
            new_data = np.zeros_like(data)
            src_area_xpix = src_area / dx
            src_area_ypix = src_area / dy
            for s in srcs:
                src_x = s['xpix']
                src_y = s['ypix']
                bbox = [[src_y - src_area_ypix // 2, src_y + src_area_ypix // 2],
                        [src_x - src_area_xpix // 2, src_x + src_area_xpix // 2]]
                slicey, slicex = slice(int(bbox[0][0]), int(bbox[0][1]) + 1), slice(int(bbox[1][0]), int(bbox[1][1]) + 1)
                new_data[0, 0, slicey, slicex] = data[0, 0, slicey, slicex]
                if verbose:
                    print('Adding source {0:s} to model at x {1:d}, y {2:d} '
                          'with flux {3:.1f} Jy'.format(s['label'], src_x, src_y, np.max(data[0, 0, slicey, slicex])))
        else:
            new_data = np.copy(data)
            sol_area_xpix = int(sol_area / dx)
            sol_area_ypix = int(sol_area / dy)
            new_data[0, 0, soly - sol_area_ypix // 2:soly + sol_area_ypix // 2 + 1,
            solx - sol_area_xpix // 2:solx + sol_area_xpix // 2 + 1] = 0.0000

        if not outimage:
            outimage = imagename + "_no_sun"
        print (outimage+prefix+'-model.fits')
        fits.writeto(outimage + prefix+'-model.fits', new_data, header=head, overwrite=True)
    return outimage
    
    
def remove_nonsolar_sources(msfile, imsize=4096, cell='2arcmin', minuv=0,
                            remove_strong_sources_only=True, pol='I', niter=50000, fast_vis=False, 
                            fast_vis_image_model_subtraction=False, delete_tmp_files=True, auto_pix_fov=False):
    """
    Wrapping for removing the nonsolar sources from the solar measurement set

    :param msfile: input CASA measurement set
    :param imsize: size of the image in pixels
    :param cell: pixel scale
    :param minuv: minimum uv to consider for imaging (in # of wavelengths)

    :return: a CASA measurement set with non-solar sources removed. Default name is "*_sun_only.ms"
    """
    outms = msfile.replace('.ms', "_sun_only.ms")
    if os.path.isdir(outms):
        return outms
    
    if fast_vis:
        remove_strong_sources_only=False
        
    tmpimg = msfile[:-3] + "_allsky"

    tmpms = msfile[:-3] + "_nonsolar_subtracted.ms"
    if not fast_vis or (fast_vis and fast_vis_image_model_subtraction):
        deconvolve.run_wsclean(msfile=msfile, imagename=tmpimg, size=imsize,
                            scale=cell, minuv_l=minuv, predict=False,
                            pol=pol, niter=niter, auto_pix_fov=auto_pix_fov)
        image_nosun = gen_nonsolar_source_model(msfile, imagename=tmpimg,
                                                remove_strong_sources_only=remove_strong_sources_only, pol=pol)
        deconvolve.predict_model(msfile, outms=tmpms, image=image_nosun, pol=pol)
        
    elif fast_vis and not fast_vis_image_model_subtraction:
        md = model_generation(vis=msfile, separate_pol=True) 	    
        modelcl, ft_needed = md.gen_model_cl()
        if ft_needed:
            os.system("cp -r " + msfile + " " + tmpms)
            clearcal(tmpms, addmodel=True)
            ft(tmpms, complist=modelcl, usescratch=True)
            
    uvsub(tmpms)
    split(vis=tmpms, outputvis=outms, datacolumn='corrected')
    # remove temporary image and ms
    if delete_tmp_files:
        os.system("rm -rf " + tmpms)
        os.system("rm -rf " + tmpimg)
    return outms
    
    
