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
from . import utils,deconvolve
import logging, glob
from .generate_calibrator_model import model_generation
from .  import generate_calibrator_model
from line_profiler import profile
tb = table()
me = measures()
cl = componentlist()
msmd = msmetadata()


def get_solar_loc_pix(msfile, image="allsky"):
    """
    Get the x, y pixel location of the Sun from an all-sky image

    :param msfile: path to CASA measurement set
    :param image: all sky image made from the measurement set
    :return: pixel value in X and Y for solar disk center
    """
    from astropy.wcs.utils import skycoord_to_pixel
    m = utils.get_sun_pos(msfile, str_output=False)
    ra = m['m0']['value']
    dec = m['m1']['value']
    coord = SkyCoord(ra * u.rad, dec * u.rad, frame='icrs')
    head=fits.getheader(image)
    w = WCS(head)
    pix = skycoord_to_pixel(coord, w)
    if np.isnan(pix[0]):
        logging.warning('Sun is not in the image')
        return None, None
    x = int(pix[0])
    y = int(pix[1])
    logging.debug( 'RA, Dec of Sun: ra={}, dec={}, Pix Sun: x={}, y={}, imagename={}'.format(ra, dec, x, y, image))
    return x, y


@profile
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


def mask_source_for_subtraction(imgdata, source,subtraction_area,cell,solar_pix,
                                        no_subtraction_region, min_subtraction_region=60):
    '''
    :param source: This has the source coordinates in pixel units
    :param subtraction area: Rectangular region centered on the source 
                                which should be subtracted. In arcminutes
    :param cell: cell size along X in arcminutes
    :param solar_pix: (solar x, solar y) coords
    :param no_subtraction_region: This is the region around sun, which will never
                                   be subtracted. We use a square of size this number.
                                   In arcminutes
    :param min_subtraction_region: This is the minimum size of the subtraction region.
                                    In arcminutes.
    '''
    srcx = source['xpix']
    srcy = source['ypix']
    solx,soly=solar_pix
    dx,dy=cell
    
    src_area_xpix = subtraction_area / dx
    src_area_ypix = subtraction_area / dy
    bbox = [[srcy - src_area_ypix // 2, srcy + src_area_ypix // 2],
                    [srcx - src_area_xpix // 2, srcx + src_area_xpix // 2]]
    
    if isinstance(solx,int) or isinstance(solx,float):                    
        dist=np.sqrt(((solx-srcx)**2+(soly-srcy)**2)*dx*dy)
    else:
        dist=1e8
    
    if dist>subtraction_area:
        logging.debug("Distance larger than the default subtraction region. Going ahead with default.")
        print ("Distance larger than the default subtraction region. Going ahead with default.")
    else:
        peak=np.nanmax(imgdata[0,0,int(bbox[0][0]):int(bbox[0][1]),int(bbox[1][0]):int(bbox[1][1])])
        pos=np.where(abs(imgdata[0,0,int(bbox[0][0]):int(bbox[0][1]),int(bbox[1][0]):int(bbox[1][1])]-peak)<1e-3)
        print (pos)
        srcx=pos[1][0]+int(bbox[1][0])
        srcy=pos[0][0]+int(bbox[0][0])
        print (srcx,srcy)
        logging.debug("Distance smaller than the default subtraction region. Will shrink.")
        while dist<subtraction_area and subtraction_area>min_subtraction_region:
            subtraction_area=max(min_subtraction_region, subtraction_area/2)
            
            src_area_xpix = subtraction_area / dx
            src_area_ypix = subtraction_area / dy
            
            bbox = [[srcy - src_area_ypix // 2, srcy + src_area_ypix // 2],
                        [srcx - src_area_xpix // 2, srcx + src_area_xpix // 2]]
            distances=np.zeros(4)
            distances[0]=np.sqrt(((solx-bbox[1][0])**2+(soly-bbox[0][0])**2)*dx*dy)
            distances[1]=np.sqrt(((solx-bbox[1][1])**2+(soly-bbox[0][1])**2)*dx*dy)
            distances[2]=np.sqrt(((solx-bbox[1][0])**2+(soly-bbox[0][1])**2)*dx*dy)
            distances[3]=np.sqrt(((solx-bbox[1][1])**2+(soly-bbox[0][0])**2)*dx*dy)
            dist=np.min(distances)
        logging.debug("Final subtraction region is "+str(subtraction_area))
        print ("Final subtraction region is "+str(subtraction_area))
    slicey, slicex = slice(int(bbox[0][0]), int(bbox[0][1]) + 1), slice(int(bbox[1][0]), int(bbox[1][1]) + 1)
    
    print('Adding source {0:s} to model at x {1:d}, y {2:d} '
                          'with flux {3:.1f} Jy'.format(source['label'], srcx, srcy, np.max(imgdata[0, 0, slicey, slicex])))
    return bbox


def mask_all_non_sun(imgdata, solar_pix, cell, shape="rect", blur_border=0,
                     include_edge_source = -1, mask_size=200):
    """
    Mask all non-solar sources from the image

    :param imgdata: input image data
    :param solar_pix: pixel location of the Sun
    :param shape ["rect", "circ",]
    :param mask_size: size of the mask in arcminutes
    :param include_edge_source: 1 to include sources at the edge (to the final image, (meaning not flagging the edge source)),
      -1 to exclude them
    :return: image with non-solar sources masked
    """

    from scipy import ndimage
    from skimage import filters
    from skimage.morphology import binary_erosion, binary_dilation, disk
    from skimage.measure import label, regionprops

    data_copy = np.copy(imgdata) # should not be modified during the process
    new_data = np.zeros_like(imgdata)
    mask = np.zeros_like(imgdata)
    (solx,soly)=solar_pix
    dx,dy=cell

    # check if the method is valid
    if shape not in ["rect", "circ"]:
        logging.warning("Method not recognized. Using default rect method")
        shape = "rect"

    if shape == "rect":
        sol_area_xpix = int(mask_size / dx)
        sol_area_ypix = int(mask_size / dy)
        new_data[0, 0, soly - sol_area_ypix // 2:soly + sol_area_ypix // 2 + 1,
            solx - sol_area_xpix // 2:solx + sol_area_xpix // 2 + 1] = 0
        mask[0, 0, soly - sol_area_ypix // 2:soly + sol_area_ypix // 2 + 1,
            solx - sol_area_xpix // 2:solx + sol_area_xpix // 2 + 1] = 1.0000

    elif shape == "circ":
        xx, yy = np.meshgrid(np.arange(imgdata.shape[3]), np.arange(imgdata.shape[2]))
        r = np.sqrt((xx - solx) ** 2 + (yy - soly) ** 2)
        r_in_pix = mask_size / dx
        mask[0,0,r < r_in_pix] = 1.0000
        new_data[0, 0, r < r_in_pix] = 0


    if include_edge_source != 0:
        mask_tmp = mask.squeeze()
        masked_data_tmp = new_data.squeeze()
        data_copy_tmp = data_copy.squeeze()
        # edge 
        mask_edge = filters.sobel(mask_tmp)
        thresholded  = data_copy_tmp> ((data_copy_tmp*mask_tmp)*1e-3)

        # expand the mask by 10 pixels disk
        masked_source = ndimage.morphology.grey_dilation(thresholded, size=(12,12))>1e-3
            
        # for the regions intersect with the edge, added to the mask
        regions_sources = label(masked_source)
        props = regionprops(regions_sources)
        for prop in props:
            mask_one_source = (regions_sources == prop.label)
            if np.sum(mask_one_source * mask_edge) > 0.5:
                if include_edge_source == 1: # include the edge source in the model to subtract
                    mask_tmp[ mask_one_source ] = 0
                    masked_data_tmp[ mask_one_source ] = data_copy_tmp[ mask_one_source ]
                elif include_edge_source == -1: # exclude the edge source from the model to subtract
                    mask_tmp[ mask_one_source ] = 1
                    masked_data_tmp[ mask_one_source ] = 0
        
        new_data = masked_data_tmp.reshape(new_data.shape)
        mask = mask_tmp.reshape(mask.shape)

    if blur_border > 0:
        from scipy.ndimage import gaussian_filter
        mask_tmp_blurred = gaussian_filter(mask_tmp, sigma=blur_border)
        mask = mask_tmp_blurred.reshape(mask.shape)
        new_data = data_copy * (1 - mask)
    
    
    return new_data, mask


    
    
@profile
def gen_nonsolar_source_model(msfile, imagename="allsky", outimage=None, 
            sol_area=400., src_area=200., overwrite_exist=False,
            remove_strong_sources_only=True, verbose=True, pol='I', no_subtraction_region=120,
            shape_sun_mask="circ", include_edge_source = -1, mask_blur_border_pix=3):
    """
    Take the full sky image, remove non-solar sources from the image

    :param msfile: path to CASA measurement set
    :param imagename: input all sky image
    :param outimage: output all sky image without other sources
    :param sol_area: size around the Sun in arcmin to be left alone
    :param src_area: size around the source to be taken away, in arcminutes
    :param remove_strong_sources_only: If True, remove only known strong sources.
        If False, remove everything other than Sun.
    :param verbose: Toggle to print out more information
    :param no_subtraction_region: This is the region around sun, which will never
                                   be subtracted. We use a square of size this number.
                                   This is in arcminutes.
    :param shape_sun_mask: shape of the mask around the Sun. Default is "circ"
    :param include_edge_source: 1 to flag the edge source, -1 to keep them in final image
    :return: FITS image with non-solar sources removed, the masked data and the mask
    """
    if not outimage:
        outimage = imagename + "_no_sun"
    present=utils.check_for_file_presence(outimage,pol=pol, suffix='model')
    if present and not overwrite_exist:
        logging.debug("I will use existing model for source subtraction "+outimage)
        return outimage, np.nan, np.nan # placeholder for data and mask
    
    imagename1=imagename
    pols=pol.split(',')
    names=[]
    for pol in pols:
        pol_prefix="-"+pol if len(pols)!=1 else '' 
        imagename=imagename+pol_prefix+"-image.fits"
        
        if os.path.isfile(imagename):
            solx, soly = get_solar_loc_pix(msfile, imagename)
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
            break
   
    imagename=imagename1
    for pol in pols:
        pol_prefix="-"+pol if len(pols)!=1 else '' 
        data = fits.getdata(imagename + pol_prefix+"-model.fits")
        head=fits.getheader(imagename + pol_prefix+"-model.fits")
        if remove_strong_sources_only:
            new_data = np.zeros_like(data)
            mask = np.ones_like(data)
            for s in srcs:
                bbox = mask_source_for_subtraction(data,s,src_area,(dx,dy),(solx,soly),no_subtraction_region)
                slicey, slicex = slice(int(bbox[0][0]), int(bbox[0][1]) + 1), slice(int(bbox[1][0]), int(bbox[1][1]) + 1)
                new_data[0, 0, slicey, slicex] = data[0, 0, slicey, slicex]
                mask[0, 0, slicey, slicex] = 0.0000
            
            if isinstance(solx,int) or isinstance(solx,float):        
                no_subtraction_region_ypix=no_subtraction_region/dy
                no_subtraction_region_xpix=no_subtraction_region/dx
                
                bbox = [[soly - no_subtraction_region_ypix // 2, soly + no_subtraction_region_ypix // 2],
                            [solx - no_subtraction_region_xpix // 2, solx + no_subtraction_region_xpix // 2]]
                slicey, slicex = slice(int(bbox[0][0]), int(bbox[0][1]) + 1), slice(int(bbox[1][0]), int(bbox[1][1]) + 1)
                new_data[0, 0, slicey, slicex] = 0.0
            #print (bbox)
        else:
            new_data, mask = mask_all_non_sun(data, (solx, soly), (dx, dy),
                    mask_size=sol_area, blur_border=mask_blur_border_pix,
                    shape=shape_sun_mask, include_edge_source = include_edge_source)
        
        fits.writeto(outimage + pol_prefix+'-model.fits', new_data, header=head, overwrite=True)
    return outimage, new_data, mask

@profile
def remove_nonsolar_sources(msfile, imsize=4096, cell='2arcmin', minuv=0, auto_pix_fov=False,
        remove_strong_sources_only=True, pol='I', niter=50000, fast_vis=False, 
        fast_vis_image_model_subtraction=False, sol_area=400., src_area=200.,
        no_subtraction_region=120, shape_sun_mask="circ", include_edge_source = -1, mask_blur_border_pix=3,
        delete_tmp_files=True, delete_allsky=True, skyimage=None,):

    """
    Wrapping for removing the nonsolar sources from the solar measurement set

    :param msfile: input CASA measurement set
    :param imsize: size of the image in pixels
    :param cell: pixel scale
    :param minuv: minimum uv to consider for imaging (in # of wavelengths)

    :return: a CASA measurement set with non-solar sources removed. Default name is "*_sun_only.ms"
    """
    outms = msfile[:-3] + "_sun_only.ms"
    if os.path.isdir(outms):
        return outms
    
    if fast_vis:
        remove_strong_sources_only=False
    
    if skyimage is None:
        tmpimg = msfile[:-3] + "_allsky"
    else:
        tmpimg = skyimage

    tmpms = msfile[:-3] + "_nonsolar_subtracted.ms"
    present=False
    if fast_vis and not fast_vis_image_model_subtraction:
        md = model_generation(vis=msfile, separate_pol=True) 	    
        modelcl, ft_needed = md.gen_model_cl()
        if ft_needed:
            os.system("cp -r " + msfile + " " + tmpms)
            clearcal(tmpms, addmodel=True)
            ft(tmpms, complist=modelcl, usescratch=True)
    
    #elif not fast_vis or (fast_vis and fast_vis_image_model_subtraction):
    else:
        present=utils.check_for_file_presence(tmpimg,pol=pol)
        
        if not present:
            print("Tried to find "+tmpimg+" but it does not exist. Will create it.")    
            deconvolve.run_wsclean(msfile=msfile, imagename=tmpimg, size=imsize,
                            scale=cell, minuv_l=minuv, predict=False,
                            auto_mask=5, pol=pol, niter=niter, auto_pix_fov=auto_pix_fov)
        else:
            logging.debug("I will use existing image "+tmpimg)
        image_nosun = gen_nonsolar_source_model(msfile, imagename=tmpimg,
                remove_strong_sources_only=remove_strong_sources_only, pol=pol,
                shape_sun_mask= shape_sun_mask, include_edge_source= include_edge_source,
                no_subtraction_region=no_subtraction_region,mask_blur_border_pix = mask_blur_border_pix,
                sol_area=sol_area, src_area=src_area)[0]
        deconvolve.predict_model(msfile, outms=tmpms, image=image_nosun, pol=pol)
            
    uvsub(tmpms)
    utils.manual_split_corrected_ms(vis=tmpms, outputvis=outms)# datacolumn='corrected'
    # remove temporary image and ms
    
    if delete_tmp_files:
        os.system("rm -rf " + tmpms)
        
        if delete_allsky and not present:
            os.system("rm -rf " + tmpimg+"*")
    
    return outms
    
    
