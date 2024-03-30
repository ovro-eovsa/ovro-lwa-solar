import os, sys
from astropy.time import Time
from astropy.io import fits
from casatools import image, table, msmetadata, quanta, measures
import numpy as np
import logging, glob
from . import primary_beam
from casatasks import split
from .primary_beam import analytic_beam as beam 
from .generate_calibrator_model import model_generation
from . import generate_calibrator_model

def get_sun_pos(msfile, str_output=True):
    """
    Return J2000 RA and DEC coordinates of the solar disk center

    :param msfile: input CASA measurement set
    :param str_output: if True, return coordinate in string form acceptable by CASA tclean
        if False, return a dictionary in CASA measures format: https://casa.nrao.edu/docs/casaref/measures.measure.html
    :return: solar disk center coordinate in string or dictionary format
    """
    tb=table()
    me=measures()
    tb.open(msfile)
    t0 = tb.getcell('TIME', 0)
    tb.close()
    ovro = me.observatory('OVRO_MMA')
    timeutc = me.epoch('UTC', '%fs' % t0)
    me.doframe(ovro)
    me.doframe(timeutc)
    d0 = me.direction('SUN')
    d0_j2000 = me.measure(d0, 'J2000')
    if str_output:
        d0_j2000_str = 'J2000 %frad %frad' % (d0_j2000['m0']['value'], d0_j2000['m1']['value'])
        return d0_j2000_str
    return d0_j2000

def get_solar_azel(msfile):
    '''
    Returns az ,el of sun in degrees.
    
    :param msfile: Name of MS 
    :type msfile: str
    
    :return: az,el in degrees
    '''
    me=measures()
    m = get_sun_pos(msfile, str_output=False)
    logging.debug('Solar ra: ' + str(m['m0']['value']))
    logging.debug('Solar dec: ' + str(m['m1']['value']))
    tb=table()
    tb.open(msfile)
    t0 = tb.getcell('TIME', 0)
    tb.close()
    ovro = me.observatory('OVRO_MMA')
    timeutc = me.epoch('UTC', '%fs' % t0)
    me.doframe(ovro)
    me.doframe(timeutc)
    d = me.measure(m, 'AZEL')
    logging.debug('Solar azimuth: ' + str(d['m0']['value']))
    logging.debug('Solar elevation: ' + str(d['m1']['value']))
    elev = d['m1']['value']*180/np.pi
    az=d['m0']['value']*180/np.pi
    return az,elev

def get_msinfo(msfile):
    """
    Return some basic information of an OVRO-LWA measurement set
    
    :param msfile: path to CASA measurement set
    :return: number of antennas, number of spectral windows, number of channels
    """
    msmd = msmetadata()
    msmd.open(msfile)
    nant = msmd.nantennas()  # number of antennas
    nspw = msmd.nspw()  # number of spectral windows
    nchan = msmd.nchan(0)  # number of channels of the first spw
    msmd.close()
    return nant, nspw, nchan

def get_image_data(imagename):
    if os.path.isfile(imagename):
        data = np.squeeze(fits.getdata(imagename))
    elif os.path.isdir(imagename):
        ia = image()
        ia.open(imagename)
        data = ia.getchunk()
        ia.close()
        data = np.squeeze(data)
    else:
        raise RuntimeError("Image does not exist")
    return data


def get_image_maxmin(imagename, local=True):
    data = get_image_data(imagename)
    maxval = np.nanmax(data)
    if local == True:
        maxpos = np.where(abs(data - maxval) < 1e-5)
        max1 = data[maxpos][0]
        min1 = np.nanmin(data[maxpos[0][0] - 100:maxpos[0][0] + 100, \
                         maxpos[1][0] - 100:maxpos[1][0] + 100])

        return max1, min1
    else:
        minval = np.nanmin(data)
    return maxval, minval


def check_image_quality(imagename, max1, min1, reorder=True):
        
    if max1[0] == 0:
        if len(max1)==2:
            max1[0], min1[0] = get_image_maxmin(imagename+"-image.fits")
        else:
            max1[0], min1[0] = get_image_maxmin(imagename+"-XX-image.fits")
            max1[1], min1[1] = get_image_maxmin(imagename+"-YY-image.fits")
        print(max1, min1)
    else:
        if reorder:
            if len(max1)==2 and max1[1]>0.00001:
                max1[0], min1[0] = max1[1], min1[1]
            elif len(max1)==4 and max1[2]>0.00001:
                max1[0], min1[0] = max1[2], min1[2]
                max1[1], min1[1] = max1[3], min1[3]
        if len(max1)==2:
            max1[1], min1[1] = get_image_maxmin(imagename+"-image.fits")
        else:
            max1[2], min1[2] = get_image_maxmin(imagename+"-XX-image.fits")
            max1[3], min1[3] = get_image_maxmin(imagename+"-YY-image.fits")

        if len(max1)==2:
            DR1 = max1[0] / abs(min1[0])
            DR2 = max1[1] / abs(min1[1])
            print(DR1, DR2)
            if (DR1 - DR2) / DR2 > 0.2:
                ### if max decreases by more than 20 percent
                ## absolute value of minimum increases by more than 20 percent
                if min1[1] < 0:
                    return False
        else:
            DR1 = max1[0] / abs(min1[0])
            DR2 = max1[2] / abs(min1[2])
            DR3 = max1[1] / abs(min1[1])
            DR4 = max1[3] / abs(min1[3])
            print (DR1,DR2,DR3,DR4)
            if (DR1 - DR2) / DR2 > 0.2 :
                ### if max decreases by more than 20 percent
                ## absolute value of minimum increases by more than 20 percent
                if min1[2] < 0:
                    return False
            if (DR3-DR4)/DR4>0.2:
                if min1[3]<0:
                    return False 
    return True


def restore_flag(msfile):
    from casatasks import flagmanager
    flag_tables = flagmanager(msfile)
    keys = flag_tables.keys()
    last_flagtable = flag_tables[len(keys) - 2]['name']  #### last key is MS.
    flagmanager(vis=msfile, mode='restore', versionname=last_flagtable)
    flagmanager(vis=msfile, mode='delete', versionname=last_flagtable)
    return


def get_flagged_solution_num(caltable):
    tb = table()
    tb.open(caltable)
    flag = tb.getcol('FLAG')
    tb.close()
    shape = flag.shape
    for i in range(shape[1]):
        num_solutions_flagged = np.where(flag[:, i, :] == True)
        if shape[1] == 1:
            logging.debug(str(len(num_solutions_flagged[0])) + " flagged out of " + str(shape[0] * shape[2]))
        else:
            logging.debug(str(len(num_solutions_flagged[0])) + " flagged out of " + str(
                shape[0] * shape[2]) + " in channel " + str(i))
    return


def get_strong_source_list():
    srcs = [{'label': 'CasA', 'position': 'J2000 23h23m24s +58d48m54s', 'flux': '16530', 'alpha': -0.72},
            {'label': 'CygA', 'position': 'J2000 19h59m28.35663s +40d44m02.0970s', 'flux': '16300', 'alpha': -0.58},
            {'label': 'TauA', 'position': 'J2000 05h34m31.94s +22d00m52.2s', 'flux': '1770', 'alpha': -0.27},
            {'label': 'VirA', 'position': 'J2000 12h30m49.42338s +12d23m28.0439s', 'flux': '2400', 'alpha': -0.86}]
    return srcs


def get_time_from_name(msname):
    pieces = os.path.basename(msname).split('_')
    ymd = pieces[0]
    hms = pieces[1]
    mstime = Time(ymd[0:4] + "-" + ymd[4:6] + "-" + ymd[6:] +
                  'T' + hms[0:2] + ":" + hms[2:4] + ":" + hms[4:],
                  scale='utc', format='isot')
    return mstime


def get_timestr_from_name(msname):
    pieces = os.path.basename(msname).split('_')
    return '_'.join(pieces[0:2])


def get_freqstr_from_name(msname):
    return os.path.basename(msname)[16:21]


def get_selfcal_time_to_apply(msname, caltables):
    mstime = get_time_from_name(msname)
    times = np.unique(np.array(['_'.join(i.split('/')[1].split('_')[0:2]) for i in caltables]))

    if len(times) > 0:
        sep = np.zeros(len(times))
        for n, t1 in enumerate(times):
            caltime = get_time_from_name(t1)
            sep[n] = abs((caltime - mstime).value * 86400)

        time_to_apply = times[np.argsort(sep)[0]]
        return time_to_apply
    return 'none'


def get_keyword(caltable, keyword, return_status=False):
    tb = table()
    success = False
    try:
        tb.open(caltable)
        val = tb.getkeyword(keyword)
        success = True
    except:
        pass
    finally:
        tb.close()
    if not return_status:
        return val
    return val, success


def put_keyword(caltable, keyword, val, return_status=False):
    tb = table()
    success = False
    try:
        tb.open(caltable, nomodify=False)
        tb.putkeyword(keyword, val)
        tb.flush()
        success = True
    except:
        pass
    finally:
        tb.close()
    if return_status == False:
        return
    return success

def get_obs_time_interval(msfile):
    msmd = msmetadata()
    msmd.open(msfile)
    trange = msmd.timerangeforobs(0)
    btime = Time(trange['begin']['m0']['value'],format='mjd').isot
    etime = Time(trange['end']['m0']['value'],format='mjd').isot
    msmd.close()
    return btime+'~'+etime
    
def convert_to_heliocentric_coords(msname, imagename, helio_imagename=None, reftime=''):
    '''
    The imagename, helio_imagename and reftime all can be a list.
    If reftime is not provided, it is assumed to the the center
    of the observation time given in the MS
    '''
    import datetime as dt
    from suncasa.utils import helioimage2fits as hf
    from casatasks import importfits
    msmd = msmetadata()
    qa = quanta()
    
    if type(imagename) is str:
        imagename=[imagename]
    elif type(imagename) is not list or type(imagename[0]) is not str:
        logging.error("Imagename provided should either be a string or a list of strings")
        raise RuntimeError("Imagename provided should either be a string or a list of strings")
    
    if helio_imagename is None:
        helio_imagename = [img.replace('.fits', '.helio.fits') for img in imagename]
    else:
        if type(helio_imagename) is str:
            helio_imagename=[helio_imagename]
        elif type(helio_imagename) is not list or type(helio_imagename[0]) is not str:
            logging.warning("Helio imagename provided should either be a string or a list of strings. Ignoring")
            helio_imagename = [img.replace('.fits', '.helio.fits') for img in imagename]
        elif len(helio_imagename)!=len(imagename):
            logging.warning("Number of helio imagenames provided does not match with the number of images provided. Ignoring")
            helio_imagename = [img.replace('.fits', '.helio.fits') for img in imagename]

    if reftime == '':
        reftime = [get_obs_time_interval(msname)]*len(imagename)
    elif type(reftime) is str:
        reftime = [reftime]*len(imagename)
    elif type(reftime) is not list or type(reftime[0]) is not str:
        logging.warning("Reftime provided should either be a string or a list of strings. Ignoring")
        reftime = [get_obs_time_interval(msname)]*len(imagename)
    elif len(reftime)!=len(imagename):
        logging.warning("Number of reftimes provided does not match with number of images. Ignoring")
        reftime = [get_obs_time_interval(msname)]*len(imagename)
        
    print('Use this reference time for registration: ', reftime)
    logging.debug('Use this reference time for registration: ', reftime[0])
    
    temp_image_list=[None]*len(imagename)
    for j,img in enumerate(imagename):
        temp_image_list[j]=img

    try:
        hf.imreg(vis=msname, imagefile=temp_image_list, timerange=reftime,
                 fitsfile=helio_imagename, usephacenter=True, verbose=True, toTb=True)
        return helio_imagename
    except:
        logging.warning("Could not convert to helicentric coordinates")
        return None
        
def make_wsclean_compatible(msname):
    tb=table()
    tb.open(msname+"/DATA_DESCRIPTION",nomodify=False)   
    nrows=tb.nrows()
    if nrows>1:
        tb.removerows([i for i in range(1,nrows)])
    tb.close()      

def get_total_fields(msname):
    msmd=msmetadata()
    msmd.open(msname)
    num_field=msmd.nfields()
    msmd.done()
    return num_field
    
def get_fast_vis_imagenames(msfile,imagename,pol):
    pols=pol.split(',')
    num_pols=len(pols)
    num_field=get_total_fields(msfile)
    msmd=msmetadata()
    msmd.open(msfile)
    names=[]
    for i in range(num_field):
        time1 = msmd.timesforfield(i)
        t=Time(time1/86400,format='mjd')
        t.format='isot'
        time_str=t.value[0].split('T')[1].replace(':','')
        for pol1 in pols:
            if pol1=='I' or num_pols==1:
                pol1=''
            else:
                pol1='-'+pol1
            
            wsclean_imagename=imagename+'-t'+str(i).zfill(4)+pol1+"-image.fits"
            final_imagename=imagename+"_"+time_str+pol1+"-image.fits"
            names.append([wsclean_imagename,final_imagename])
    return names
    
def check_corrected_data_present(msname):
    tb=table()
    try:
        tb.open(msname)
        colnames=tb.colnames()
        if 'CORRECTED_DATA' in colnames:
            tb.close()
            return True
    finally:
        tb.close()
    return False
    
    
def correct_primary_beam(msfile, imagename, pol='I', fast_vis=False):
    '''
    Can handle multiple images in a list. However if providing multiple images
    provide full name of files. No addition to filename is done.
    If single file is provided, we can add '.image.fits' to it. 
    '''
    
    az,elev=get_solar_azel(msfile)
    pb=beam(msfile=msfile)
    pb.srcjones(az=[az],el=[elev])
    jones_matrices=pb.get_source_pol_factors(pb.jones_matrices[0,:,:])

    md=generate_calibrator_model.model_generation(vis=msfile)
    if fast_vis==False:
        if pol=='I':
            scale=md.primary_beam_value(0,jones_matrices)
            logging.info('The Stokes I beam correction factor is ' + str(round(scale, 4)))
            if type(imagename) is str:
                if os.path.isfile(imagename+"-image.fits"):
                    imagename=[imagename+"-image.fits"]
                elif os.path.isfile(imagename):
                    imagename=[imagename]
                else:
                    raise RuntimeError("Image supplied is not found")
                    
            for img in imagename:
                hdu = fits.open(img, mode='update')
                hdu[0].data /= scale
                hdu.flush()
                hdu.close()
        else:
            for pola in ['I','Q','U','V','XX','YY']:
                if pola=='I' or pola=='XX' or pola=='YY':
                    n=0
                elif pola=='Q':
                    n=1
                elif pola=='U':
                    n=2
                else:
                    n==3
                scale=md.primary_beam_value(n,jones_matrices)
                logging.info('The Stokes '+pola+' beam correction factor is ' + str(round(scale, 4)))
                if os.path.isfile(imagename+ "-"+pola+"-image.fits"):
                    hdu = fits.open(imagename+ "-"+pola+"-image.fits", mode='update')
                    hdu[0].data /= scale
                    hdu.flush()
                    hdu.close()
                elif pola=='I' and os.path.isfile(imagename+"-image.fits"):
                    hdu = fits.open(imagename+"-image.fits", mode='update')
                    hdu[0].data /= scale
                    hdu.flush()
                    hdu.close()
    else:
        image_names=get_fast_vis_imagenames(msfile,imagename,pol)
        for name in image_names:
            if os.path.isfile(name[1]):
                if pol=='I':
                    scale=md.primary_beam_value(0,jones_matrices)
                else:
                    pola=name[1].split('-')[-1]
                    if pola=='I' or pola=='XX' or pola=='YY':
                        n=0
                    elif pola=='Q':
                        n=1
                    elif pola=='U':
                        n=2
                    else:
                        n==3
                    scale=md.primary_beam_value(n,jones_matrices)
                hdu = fits.open(name[1], mode='update')
                hdu[0].data /= scale
                hdu.flush()
                hdu.close()
    return
    
def get_solar_loc_pix(msfile, image="allsky"):
    """
    Get the x, y pixel location of the Sun from an all-sky image

    :param msfile: path to CASA measurement set
    :param image: all sky image made from the measurement set
    :return: pixel value in X and Y for solar disk center
    """
    from astropy.wcs.utils import skycoord_to_pixel
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from astropy.wcs import WCS
    
    m = get_sun_pos(msfile, str_output=False)
    ra = m['m0']['value']
    dec = m['m1']['value']
    coord = SkyCoord(ra * u.rad, dec * u.rad, frame='icrs')
    logging.debug('RA, Dec of Sun is ' + str(ra) + ", " + str(dec) + ' rad')
    head=fits.getheader(image)
    w = WCS(head)
    pix = skycoord_to_pixel(coord, w)
    if np.isnan(pix[0]):
        logging.warning('Sun is not in the image')
        return None, None
    x = int(pix[0])
    y = int(pix[1])
    logging.debug('Pixel location of Sun is ' + str(x) + " " + str(y) + " in imagename " + image)
    return x, y
    
def get_rms(data,thresh=7):
    '''
    This function returns the rms of the data. As a first cut, it calculates
    the std. Then it calculates the std again, by only considering pixels which
    are lower than the thresh*rms
    
    :param data: image data
    :type data: numpy ndarray
    :param thresh: threshold above rms to remove true sources in rms caluclation.
                    Default: 7
    :type thresh: float
    :return: rms
    :rtype: float
    '''
    rms=np.nanstd(data)
    pos=np.where(abs(data)<thresh*rms)
    rms=np.nanstd(data[pos])
    return rms

def blank_all_pixels(imagename):
    hdu=fits.open(imagename,mode='update')
    try:
        hdu[0].data*=0.0
        hdu.flush()
    finally:
        hdu.close()
    return

def get_uvlambda_from_uvdist(u,v,msname=None, freqs=None):
    msmd = msmetadata()
    msmd.open(msname)
    chan_freqs = msmd.chanfreqs(0)
    msmd.done()
    
    wavelengths=299792458/chan_freqs
    
    u=np.expand_dims(u,1)
    u=np.repeat(u,len(chan_freqs),axis=1)
    ulambda=(u/wavelengths).T
    
    v=np.expand_dims(v,1)
    v=np.repeat(u,len(chan_freqs),axis=1)
    vlambda=(v/wavelengths).T
    
