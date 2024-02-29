import os, sys
from astropy.time import Time
from astropy.io import fits
from casatools import image, table, msmetadata, quanta, measures
import numpy as np
import logging, glob
from . import primary_beam
from casatasks import split
from .primary_beam import woody_beam as beam 
from . import utils
from . import generate_calibrator_model
from . import selfcal
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.convolution import Gaussian2DKernel, convolve_fft
from ovrolwasolar import deconvolve
from scipy.optimize import minimize_scalar

def get_corr_type(msname):
    '''
    This function reads the msname/POLARIZATION table and determines
    the polarisation type.
    5,6,7,8: RR,RL,LR,LL
    9,10,11,12: XX,XY,YX,YY
 
    :param msname: Name of MS
    :return: RL/XY
    :rtype: str
    '''
    tb=table()
    tb.open(msname+"/POLARIZATION")
    try:
        corr_type=tb.getcol('CORR_TYPE')
    finally:
        tb.close()
    
    corr_type=np.squeeze(corr_type)
    if corr_type[0]==9 and corr_type[3]==12:
        return "XY"
    if corr_type[0]==5 and corr_type[3]==8:
        return "RL"
    
def change_corr_type(msname):
    '''
    This function reads the msname/POLARIZATION table and determines
    the polarisation type.
    5,6,7,8: RR,RL,LR,LL
    9,10,11,12: XX,XY,YX,YY
    Next it changes the basis type from linear to circular and vice versa
 
    :param msname: Name of MS
    :return: If successful returns True, else False
    :rtype: boolean
    '''
    success=False
    tb=table()
    tb.open(msname+"/POLARIZATION",nomodify=False)
    try:
        corr_type=tb.getcol('CORR_TYPE')
        if corr_type[0][0]==9 and corr_type[3][0]==12:
            corr_type=np.array([[5],[6],[7],[8]],dtype=int)
        elif corr_type[0][0]==5 and corr_type[3][0]==8:
            corr_type=np.array([[9],[10],[11],[12]])
        
        tb.putcol('CORR_TYPE',corr_type)
        tb.flush()
        success=True
    finally:
        tb.close()
    return success
    
def change_visibility_basis(msname):
    '''
    This function will convert the MS from a linear basis to circular basis
    or vice versa. The basis type is determined from the MS itself. Note
    that the original MS will never be rewritten. If corrected data column 
    is present, the corrected data column will be splitted and then the splitted
    MS will be changed. In absence of corrected data column, data column will 
    be splitted and then the change will be done
    
    :param msname: Name of the input MS
    :type msname: str
    
    :return: outms
    :rtype: str
    '''
    present=utils.check_corrected_data_present(msname)
    if present:
        datacolumn='CORRECTED'
    else:
        datacolumn='DATA'
    
    outms=msname.replace('.ms','_basis_changed.ms')
    
    split(vis=msname,outputvis=outms,datacolumn=datacolumn)
    
    corr_type=get_corr_type(msname)

    
    tb=table()
    success1=False
    tb.open(outms,nomodify=False)
    try:
        data=tb.getcol('DATA')
        flag=tb.getcol('FLAG')
        pos=np.where(flag)
        data[pos]=np.nan
        data1=np.zeros_like(data)
        flag1=np.zeros_like(flag,dtype=bool)
        if corr_type=='XY':
            data1[0,...]=0.5*(data[0,...]+data[3,...]-1j*(data[1,...]-data[2,...]))
            data1[3,...]=0.5*(data[0,...]+data[3,...]+1j*(data[1,...]-data[2,...]))
            data1[1,...]=0.5*(data[0,...]-data[3,...]+1j*(data[1,...]+data[2,...]))
            data1[2,...]=0.5*(data[0,...]-data[3,...]-1j*(data[1,...]+data[2,...]))
            logging.debug("Changing to RL basis")
        elif corr_type=='RL':
            data1[0,...]=0.5*(data[0,...]+data[3,...]+data[1,...]+data[2,...])
            data1[3,...]=0.5*(data[0,...]+data[3,...]-data[1,...]-data[2,...])
            data1[1,...]=1j*0.5*(data[0,...]-data[3,...]-data[1,...]+data[2,...])
            data1[2,...]=1j*0.5*(-data[0,...]+data[3,...]-data[1,...]+data[2,...])
            logging.debug("Changing to XY basis")
        pos=np.where(np.isnan(data1))
        data1[pos]=1.0
        flag1[pos]=True
        tb.putcol('DATA',data1)
        tb.putcol('FLAG',flag1)
        tb.flush()
        success1=True
    finally:
        tb.close()
    success=change_corr_type(outms)

    if (not success) or (not success1):
        os.system("rm -rf "+outms)
        raise RuntimeError("Failed to change polarisation basis")
    return outms

def squint_correction(msname,caltable_folder='squint_correction',output_ms=None):
    '''
    This function corrects for squint. This will produce an I image and then
    do a round of phase selfcal. The caltable will be copied in a folder named
    fold. The function will verify if the MS is in RL basis. If not it will raise
    an error. The caltable will be applied in a calonly mode.
    :param msname: Name of the measurementset
    :type msname: str
    :param caltable_folder: Folder where the phase caltable will be copied
    :type caltable_folder: str
    :param output_ms: Name of output MS. If not provided, squint_corrected will be appended
    :type output_ms: str,optional
    
    :return: output_ms
    :rtype: str
    '''
    corr_type=get_corr_type(msname)
    if corr_type!='RL':
        outms=change_visibility_basis(msname)
    else:
        logging.debug("Visibilities are in RL basis. Proceeding")
        outms=msname    
    try:
        os.mkdir(caltable_folder)
    except OSError:
        pass
    
    mstime_str = utils.get_timestr_from_name(outms)
    success = utils.put_keyword(outms, 'di_selfcal_time', mstime_str, return_status=True)
    ### The above two lines are needed to keep the code consistent with DI and DD selfcal
    success=selfcal.do_selfcal(outms,num_phase_cal=1,num_apcal=0,applymode='calonly',\
                                caltable_folder=caltable_folder,bandpass_selfcal=True)
    if not success:
        logging.error("Squint correction was not successfull")
        raise RuntimeError("Squint correction was not successfull")
    
    if corr_type!='RL':
        ms1=change_visibility_basis(outms)
    else:
        ms1=outms
    print (ms1)    
    if output_ms is None:
        output_ms=msname.replace('.ms','_squint_corrected.ms')
        
    corrected_present=utils.check_corrected_data_present(ms1)
    if corrected_present:
        split(vis=ms1,outputvis=output_ms)
    else:
        os.system("mv "+ms1+" "+output_ms)
    
    if os.path.isdir(outms):
        os.system("rm -rf "+outms)
        os.system("rm -rf "+outms+".flagversions")
        
    if os.path.isdir(ms1):
        os.system("rm -rf "+ms1)
        os.system("rm -rf "+ms1+".flagversions")
    
    return output_ms 

def detect_sources(imagename,adaptive_rms=True,thresh_pix=10,thresh_isl=8,imgcat=None, overwrite=False):
    '''
    This function uses PyBDSF to detect and write a CSV file containing the information of the detected
    sources. This function will only work as intended for Stokes I images. All parameters except provided
    below are same as those used by PyBDSF.
    :param imagename: Name of the image. This should be the full name and not the image prefix provided
                        to WSClean
    :type imagename: str
    :param overwrite: The image catalogue will be overwritten if True. Default: False
    :type overwrite: Boolean
    :param imgcat: Name of the image catalogue. Default is None. If not provided, the imgcat will
                    replace "-image.fits" by ".pybdsf"
    :type imgcat: str
    
    :return: imgcat
    :rtype: str
    '''
    import bdsf
    
    if imgcat is None:
        imgcat=imagename.replace("-image.fits",".pybdsf")

    if not os.path.isfile(imgcat) or overwrite:
        outputs=bdsf.process_image(imagename,adaptive_rms=adaptive_rms,\
                                thresh_pix=thresh_pix,thresh_isl=thresh_isl)
    
        outputs.write_catalog(outfile=imgcat,format='csv',catalog_type='srl',clobber=overwrite)
        logging.debug("Catalogue of sources found by PyBDSF is written to ",imgcat)
    else:
        logging.debug("Catalogue file not written because the file existed and user does not want to overwrite")   
    return imgcat
    
def read_pybdsf_output(imgcat):
    '''
    This just reads the source catalogue produced by PyBDSF
    
    :param imgcat: source catalogue produced by PyBDSF
    :type imgcat: str
    
    :return: img_ra,img_dec,source_code, integrated_flux
    :rtype: float,float, str, float
    '''
    img_data=pd.read_csv(imgcat,skiprows=5,sep=', ',engine='python')
    img_ra=img_data['RA']
    img_dec=img_data['DEC']
    s_code=img_data['S_Code']
    img_flux=img_data['Total_flux']
    return img_ra,img_dec,s_code,img_flux
    
def convert_to_altaz(img_coord,obstime,observer='ovro'):
    '''
    This function will convert a sky coordinate to alt-az coordinates
    
    :param img_coord: Sky coordinate of the source
    :type img_coord: Astropy SkyCoord
    :param obstime: Observative time
    :type obstime: Astropy Time object
    :param observer: observer will be passed to EarthLocation.of_site.
                     If it fails we assume that observer is already a 
                     Astropy Earth location.
    :type observer: str or Astropy Earth location
    
    :return: alt,az in degrees
    :rtype: float, float
    '''
    try:
        ovro_loc = EarthLocation.of_site(observer)
    except:
        ovro_loc=observer
    
    
    aa = AltAz(location=ovro_loc, obstime=obstime)
    coord1=img_coord.transform_to(aa)
    return coord1.alt.value, coord1.az.value
    
def get_beam_factors(alt,az,freq,normalise=True):
    '''
    This function will take alt,az and freq and return the primary beam values
    
    :param alt: Altitude in degrees
    :type alt: float
    :param az: Azimuth in degrees
    :type az: float
    :param freq: Frequency in MHz
    :type freq: float
    :param normalise: If True, we will normalise all values by the calculated 
                        I beam value. Default: True
    :type normalise: Boolean
    
    :return: array containing [I,Q,U,V]
    :rtype: float
    
    '''
    beamfac=beam(freq=freq)    
    beamfac.srcjones(az=az,el=alt)
    num_source=len(alt)
    factors=np.zeros((num_source,4))
    for i in range(num_source):
        pol_fac=beamfac.get_source_pol_factors(beamfac.jones_matrices[i,:,:])
        factors[i,0]=np.real(0.5*(pol_fac[0,0]+pol_fac[1,1]))
        factors[i,1]=np.real(0.5*(pol_fac[0,0]-pol_fac[1,1]))
        factors[i,2]=np.real(0.5*(pol_fac[0,1]+pol_fac[1,0]))
        factors[i,3]=np.real(1j*0.5*(pol_fac[0,1]-pol_fac[1,0]))
    
    if normalise:
        return factors/np.expand_dims(factors[:,0],axis=1) ## all factors are in respect to I value
    else:
        return factors

def get_good_sources(imagename, imgcat, min_alt=5):
    '''
    This function will choose good sources from the source catalog.
    
    :param imagename: Name of image. Should not be the prefix of the imagename
                        supplied to WSClean
    :type imagename: str
    :param imgcat: Name of source catalog
    :type imgcat: str
    :param min_alt: Minimum altitude at which the sources should be to be considered
    :type min_alt: float
    
    :return: img_ra,img_dec, integrated_flux
    :rtype: float,float, float 
    
    '''

    img_ra,img_dec,s_code,img_flux=read_pybdsf_output(imgcat)
    
    pos=np.where((np.isnan(img_ra)==False) & (np.isnan(img_dec)==False) & (s_code=='S'))[0]

    img_ra=np.array(img_ra[pos])
    img_dec=np.array(img_dec[pos])
    img_flux=np.array(img_flux[pos]) 
    
    
    
    img_coord=SkyCoord(img_ra*u.degree,img_dec*u.degree,frame='icrs')   
    
    head=fits.getheader(imagename)
    obstime=Time(head['DATE-OBS'])
    
    alt,az=convert_to_altaz(img_coord,obstime)
    pos=np.where(alt>min_alt)
    
    img_ra=np.array(img_ra[pos])
    img_dec=np.array(img_dec[pos])
    img_flux=np.array(img_flux[pos]) 
    return img_ra,img_dec,img_flux
    
def generate_polarised_skymodel(imagename,imgcat=None, min_alt=5): 
    '''
    This will generate the polarised sky model using the model image.
    We assume that the sky in entirely unpolarised, and the only polarisation
    arises because of the primary beam
    
    :param imagename: imagename should be standard WSclean format. Ensure that 
                        imagename-{I,Q,U,V}-image.fits and -model.fits exist
    :type imagename: str
    :param imgcat: Name of source catalog. Default: None
    :type imgcat: str or None
    :param min_alt: Minimum altitude
    :type min_alt: float
    '''
    
    if imgcat is None:
        imgcat= detect_sources(imagename+"-I-image.fits")
        
    img_ra,img_dec,img_flux=get_good_sources(imagename+"-I-image.fits",imgcat)
    
    num_sources=len(img_ra)
    
    img_coord=SkyCoord(img_ra*u.degree,img_dec*u.degree,frame='icrs')   
    
    head=fits.getheader(imagename+"-I-image.fits")
    obstime=Time(head['DATE-OBS'])
    
    alt,az=convert_to_altaz(img_coord,obstime)
    
    
    
    freq=head['CRVAL3']*1e-6
    
    factors=get_beam_factors(alt,az,freq=freq)
    
    
    imwcs=WCS(head)
    img_xy = imwcs.all_world2pix(list(zip(img_ra, img_dec,[head['CRVAL3']]*num_sources,[head['CRVAL4']]*num_sources)), 1)
    

    for j,stokes in enumerate(['Q','U','V']):
        image=imagename+'-'+stokes+"-model.fits"
        hdu=fits.open(image,mode='update')
        try:
            data=hdu[0].data
            data*=0.0
            for i in range(num_sources):
                x=int(img_xy[i,0])
                y=int(img_xy[i,1])
                data[0,0,y,x]=img_flux[i]*factors[i,j+1]
            hdu[0].data=data
            hdu.flush()
        finally:
            hdu.close()

        
def add_solar_model(imagename,msfile,sol_area=400.):  
    '''
    This will add the polarised solar model.
    :param imagename:  imagename should be standard WSclean format. Ensure that 
                        imagename-{I,Q,U,V}-image.fits and -model.fits exist
    :type imagename: str
    :param msfile: Name of MS. This is used to get the solar ra-dec
    :type msfile: str
    :param sol_area: This is area of the sky around the sun, inside which
                        sun is guaranteed to be present. This is the diameter
                        of the region
    :type sol_area: float
    '''          
    me=measures()
    m = utils.get_sun_pos(msfile, str_output=False)
    solar_ra=m['m0']['value']*180/np.pi
    solar_dec=m['m1']['value']*180/np.pi
    solar_az,solar_el=utils.get_solar_azel(msfile)
    
    head=fits.getheader(imagename+"-I-image.fits")
    freq=head['CRVAL3']*1e-6
    imwcs=WCS(head) 
    solar_xy = imwcs.all_world2pix(list(zip([solar_ra], [solar_dec],[head['CRVAL3']],[head['CRVAL4']])), 1)      
    
    if head['cunit1'] == 'deg':
        dx = np.abs(head['cdelt1'] * 60.)
    else:
        print(head['cunit1'] + ' not recognized as "deg". Model could be wrong.')
    if head['cunit2'] == 'deg':
        dy = np.abs(head['cdelt2'] * 60.)
    else:
        print(head['cunit2'] + ' not recognized as "deg". Model could be wrong.')                   
    sol_area_xpix = int(sol_area / dx)
    sol_area_ypix = int(sol_area / dy)
    
    factors=get_beam_factors([solar_el],[solar_az],freq=freq)

    
    solx=int(solar_xy[0][0])
    soly=int(solar_xy[0][1])
    
    img_data=fits.getdata(imagename+'-I-model.fits')
    
    for j,stokes in enumerate(['Q','U','V']):
        image=imagename+'-'+stokes+"-model.fits"
        hdu=fits.open(image,mode='update')
        try:
            data=hdu[0].data
           
            data[0, 0, soly - sol_area_ypix // 2:soly + sol_area_ypix // 2 + 1,
                    solx - sol_area_xpix // 2:solx + sol_area_xpix // 2 + 1] = \
                                            img_data[0, 0, soly - sol_area_ypix // 2:soly + sol_area_ypix // 2 + 1,
                                            solx - sol_area_xpix // 2:solx + sol_area_xpix // 2 + 1]*factors[0,j+1]
            hdu[0].data=data
            hdu.flush()
        finally:
            hdu.close()
            
def get_conv_kernel(head):
    
    bmaj=head['BMAJ']
    bmin=head['BMIN']
    bpa=head['BPA']
    dx=abs(head['CDELT1'])
    
    
    sigma_major=bmaj/(2*np.sqrt(2*np.log(2)))
    sigma_minor=bmin/(2*np.sqrt(2*np.log(2)))
    
    sigma_major_pix=sigma_major/dx
    sigma_minor_pix=sigma_minor/dx
    
    theta=90-bpa  ### CASA angle increases clockwise
    
    kernel=Gaussian2DKernel(sigma_major_pix,sigma_minor_pix,theta=theta)
    return kernel
    
def get_img_correction_factor(imagename,stokes,msfile,imgcat,sun_only=False, sol_area=400.,src_area=200,thresh=5,limit_frac=0.05):
    '''
    This function determines the correction factor using an image based method. It calculates the difference between ratio of 
    predicted Q/I,U/I,V/I values and their observed values for each source detected. For all sources other than Sun,
    the model is essentially a point. To get the I flux, we search for the peak within a 200arcmin region around
    the detected source. For the Sun, we have put in a scaled copy of the I model. Hence we do a average of the
    ratios between a 400arcmin region around the Sun. Then it does a weighted average of
    the differences. The weighting is done by the Stokes I fluxes. Thus the difference of the brightest source gets the
    highest weight. If the resultant mean is small, then wo do no bother to do an image based correction. The way the
    correction fraction is determined implies that the corrected image should be
    Q_corrected=Q_obs+correction_factor*Iobs
    U_corrected=U_obs+correction_factor*Iobs
    V_corrected=V_obs+correction_factor*Iobs
    
    Please also ensure that the Stokes I and other parameters are correlated before running 
    this function for that Stokes parameter.
    
    :param imagename: Prefix of the image. This means imagename-I-image.fits, imagename-Q-image.fits etc should exist
    :type imagename: str
    :param stokes: Stokes of the image. Can be either Q,U,V
    :type stokes: str
    :param msfile: MS name, which is used to determine the solar loc
    :type msfile: str
    :param imgcat: Catalog of the source list returned by PyBDSF
    :type imgcat: str
    :param sol_area: Diameter of region around Sun which will be used to calculate correction factor
    :type sol_area: integer
    :param src_area: Diameter of region around other sources which will be used to calculate correction factor
    :type src_area: integer
    :param thresh: Threshold in terms of rms which will be used to determine a source detection in Stokes images
    :type thresh: float
    :param limit_frac: If absolute of the correction factor is less than this, the image plane correction will
                        not be done. This function will return 0 in that case
    :type limit_frac: float
    :param sun_only: Other sources will be ignored when finding the leakage correction factor.
    :type sun_only: Boolean. Default: False
    '''
    me=measures()
    m = utils.get_sun_pos(msfile, str_output=False)
    solar_ra=m['m0']['value']*180/np.pi
    solar_dec=m['m1']['value']*180/np.pi
    
    head=fits.getheader(imagename+"-I-image.fits")
    Idata=fits.getdata(imagename+"-I-image.fits")
    freq=head['CRVAL3']*1e-6
    imwcs=WCS(head) 
    solar_xy = imwcs.all_world2pix(list(zip([solar_ra], [solar_dec],[head['CRVAL3']],[head['CRVAL4']])), 1)   
    
    if head['cunit1'] == 'deg':
        dx = np.abs(head['cdelt1'] * 60.)
    else:
        print(head['cunit1'] + ' not recognized as "deg". Model could be wrong.')
    if head['cunit2'] == 'deg':
        dy = np.abs(head['cdelt2'] * 60.)
    else:
        print(head['cunit2'] + ' not recognized as "deg". Model could be wrong.')                   
    sol_area_xpix = int(sol_area / dx)
    sol_area_ypix = int(sol_area / dy)
    
    src_area_xpix = int(src_area / dx)
    src_area_ypix = int(src_area / dy)
    
    solx=int(solar_xy[0][0])
    soly=int(solar_xy[0][1])
    
    model=imagename+"-"+stokes+"-model.fits"
    img=imagename+"-"+stokes+"-image.fits"
    
    model_data=fits.getdata(model)
    img_data=fits.getdata(img)
    
    
    head=fits.getheader(imagename+"-I-image.fits")
    obstime=Time(head['DATE-OBS'])
    
    diff_frac=[]
    weights=[]
    
    rms=np.nanstd(img_data)
    pos=np.where(abs(img_data)<10*rms)
    rms=np.nanstd(img_data[pos])
    
    if not sun_only:
    
        img_ra,img_dec,img_flux=get_good_sources(imagename+"-I-image.fits",imgcat)
        
        num_sources=len(img_ra)
        
        img_coord=SkyCoord(img_ra*u.degree,img_dec*u.degree,frame='icrs')   
        
        
        
        alt,az=convert_to_altaz(img_coord,obstime)
        
        
        
        
        freq=head['CRVAL3']*1e-6
        
        factors=get_beam_factors(alt,az,freq=freq)
        
        if stokes=='Q':
            factors=factors[:,1]
        elif stokes=='U':
            factors=factors[:,2]
        elif stokes=='V':   
            factors=factors[:,3]
        
        img_coord=SkyCoord(img_ra*u.degree,img_dec*u.degree,frame='icrs')  
        
        img_xy = imwcs.all_world2pix(list(zip(img_ra, img_dec,[head['CRVAL3']]*num_sources,[head['CRVAL4']]*num_sources)), 1)
        
        
                
        for i in range(num_sources):
            x=int(img_xy[i,0])
            y=int(img_xy[i,1])


            if x>sol_area_xpix//2-solx and x<sol_area_xpix//2+solx and \
                y>sol_area_ypix//2-solx and y<sol_area_xpix//2+soly:
                continue
            Ival=np.nanmax(Idata[0,0,y-src_area_ypix//2:y+src_area_ypix//2,\
                                    x-src_area_xpix//2:x+src_area_xpix//2])

            model_val=model_data[0,0,y,x]
            

            
            img_val1=np.nanmax(img_data[0,0,y-src_area_ypix//2:y+src_area_ypix//2,\
                                    x-src_area_xpix//2:x+src_area_xpix//2])
            img_val2=np.nanmin(img_data[0,0,y-src_area_ypix//2:y+src_area_ypix//2,\
                                    x-src_area_xpix//2:x+src_area_xpix//2])
            
            
            if img_val1>abs(img_val2):
                img_val=img_val1
            else:
                img_val=img_val2
            

            if abs(img_val)>thresh*rms and (not (abs(weights-Ival)<1e-3).any()):
                diff_frac.append(factors[i]-img_val/Ival)
                weights.append(Ival)
            
   

    pos=np.where(abs(img_data)<thresh*rms)
    img_data[pos]=np.nan
   
    
    rms=np.nanstd(Idata)
    pos=np.where(abs(Idata)<thresh*rms)
    Idata[pos]=np.nan
    
    
    
    solar_az,solar_el=utils.get_solar_azel(msfile)        
    factors=get_beam_factors([solar_el],[solar_az],freq=freq)
    
    if stokes=='Q':
        solar_model_frac=factors[0,1]
    elif stokes=='U':
        solar_model_frac=factors[0,2]
    elif stokes=='V':   
        solar_model_frac=factors[0,3]
    
    
    solar_obs_frac=np.nanmedian(img_data[0, 0, soly - sol_area_ypix // 2:soly + sol_area_ypix // 2 + 1,
                                            solx - sol_area_xpix // 2:solx + sol_area_xpix // 2 + 1]/
                                            Idata[0, 0, soly - sol_area_ypix // 2:soly + sol_area_ypix // 2 + 1,
                                            solx - sol_area_xpix // 2:solx + sol_area_xpix // 2 + 1])

    if not np.isnan(solar_obs_frac):
        diff_frac.append(solar_model_frac-solar_obs_frac)
        weights.append(np.nanmax(Idata[0, 0, soly - sol_area_ypix // 2:soly + sol_area_ypix // 2 + 1,
                                            solx - sol_area_xpix // 2:solx + sol_area_xpix // 2 + 1]))
    
    diff_frac=np.array(diff_frac)
    weight=np.array(weights)
    print (diff_frac)
    print (weights)
    frac=np.average(diff_frac,weights=weights)
    if abs(frac)<limit_frac:
        frac=0.0
    return frac
    
            
            
    
    

def correct_image_leakage(msname,factor, inplace=False, outms=None):
    '''
    This function implements a constant leakage correction for
    visibilities. If corrected data column is present, the corrected
    column is modified. If not, datacolumn is modified. However if
    inplace is set to False, then a MS is split and then the correction
    is done. The correction factor should be determined using the function
    get_img_correction_factor
    
    :param msname: MS to be corrected
    :type param: str
    :param factor: Correction factor
    :type factor: array of floats of size 3 [Qfac,Ufac,Vfac]
    :param inplace: Determines whether correction will be done inplace; optional
                    default: False, which means, by default the input MS will not be
                    modified.
    :type inplace: Boolean
    :param outms: If inplace is False, then outms is created and the data is modified.
    :type outms: str
    '''  
    
      
    if outms is None:
        outms=msname.replace(".ms","_img_leak_corrected.ms")
    
    present=utils.check_corrected_data_present(msname)
    if present:
        datacolumn='CORRECTED'
    else:
        datacolumn='DATA'
        
    if not inplace:
        split(vis=msname,outputvis=outms,datacolumn=datacolumn)
        datacolumn='DATA'
        msname=outms
    
    if datacolumn=='CORRECTED':
        datacolumn='CORRECTED_DATA'
    
    
    success=False
    
    Qfac=factor[0]
    Ufac=factor[1]
    Vfac=factor[2]
    
    tb=table()
    tb.open(msname,nomodify=False)
    try:
        data=tb.getcol(datacolumn)
        Isum=0.5*(data[0,...]+data[3,...])
        data[0,...]+=Qfac*Isum
        data[3,...]+=-Qfac*Isum
        data[1,...]+=(Ufac+1j*Vfac)*Isum
        data[2,...]+=(Ufac-1j*Vfac)*Isum
        tb.putcol(datacolumn,data)
        tb.flush()
        success=True
    except:
        pass
    finally:
        tb.close()
    
    if not success:
        logging.warning("Image based leakage correction "+\
                        "was not successfull. Please proceed with caution.")
    return msname



def correlation_based_leakage_correction(msname,imagename, slope=None,outms=None,sol_area=400,thresh=7,correlation_thresh=10):
    '''
    This function calculates the correlation of Q,U and V with I, after taking into account the leakage of the beam. Any
    residual correlation should come from leakage. We determine the correlation factor and then use the image based
    leakage correction method to apply this to the data.
    
    :param msname: Name of MS
    :param imagename: Prefix of the image which shall be used to calculate the correlation parameters
    :param slope: The linear correlation between I and various Stokes parameters. Should be in order of [Q,U,V]
    :param outms: Name of the MS which will be output after correction.
    :param sol_area: The area around Sun in arcminutes which will be considered for determining the correction factor
    :param thresh: thresh x rms is the threshold which will be used to ignore low SNR points during correlation 
                    calculation
    :param correlation_thresh: The correlation should be determined with at least this SNR for it to be considered
                                as real.
    :return outms: The MS after correction
    :rtype : str
    '''
    if slope is None:
        me=measures()
        m = utils.get_sun_pos(msname, str_output=False)
        solar_ra=m['m0']['value']*180/np.pi
        solar_dec=m['m1']['value']*180/np.pi
        
        head=fits.getheader(imagename+"-I-image.fits")

        freq=head['CRVAL3']*1e-6
        imwcs=WCS(head) 
        solar_xy = imwcs.all_world2pix(list(zip([solar_ra], [solar_dec],[head['CRVAL3']],[head['CRVAL4']])), 1)   
        
        if head['cunit1'] == 'deg':
            dx = np.abs(head['cdelt1'] * 60.)
        else:
            print(head['cunit1'] + ' not recognized as "deg". Model could be wrong.')
        if head['cunit2'] == 'deg':
            dy = np.abs(head['cdelt2'] * 60.)
        else:
            print(head['cunit2'] + ' not recognized as "deg". Model could be wrong.')                   
        sol_area_xpix = int(sol_area / dx)
        sol_area_ypix = int(sol_area / dy)
        
        solx=int(solar_xy[0][0])
        soly=int(solar_xy[0][1])
        
        ymin=soly - sol_area_ypix // 2
        ymax=soly + sol_area_ypix // 2 + 1
        xmin=solx - sol_area_xpix // 2
        xmax=solx + sol_area_xpix // 2 + 1
        

        
        Idata=fits.getdata(imagename+"-I-image.fits")
        Qdata=fits.getdata(imagename+"-Q-image.fits")[0,0,ymin:ymax,xmin:xmax]
        Udata=fits.getdata(imagename+"-U-image.fits")[0,0,ymin:ymax,xmin:xmax]
        Vdata=fits.getdata(imagename+"-V-image.fits")[0,0,ymin:ymax,xmin:xmax]
        
        rms=np.nanstd(Idata)
        pos=np.where(Idata<thresh*rms)
        Idata[pos]=np.nan
        
        Idata=Idata[0,0,ymin:ymax,xmin:xmax]
        
        solar_az,solar_el=utils.get_solar_azel(msname)        
        factors=get_beam_factors([solar_el],[solar_az],freq=freq)
        

        
        Qdata_true=Qdata-factors[0,1]*Idata
        Udata_true=Udata-factors[0,2]*Idata
        Vdata_true=Vdata-factors[0,3]*Idata
        
        slope=np.zeros(3)
        error=np.zeros(3)
        
        pos=np.where(np.isnan(Idata)==False)
        p,cov=np.polyfit(Idata[pos],Qdata_true[pos],deg=1,cov=True)
        
        if abs(p[0])>correlation_thresh*np.sqrt(cov[0,0]):
            slope[0]=p[0]
            error[0]=cov[0,0]
        
        p,cov=np.polyfit(Idata[pos],Udata_true[pos],deg=1,cov=True)
        if abs(p[0])>correlation_thresh*np.sqrt(cov[0,0]):
            slope[1]=p[0]
            error[1]=cov[0,0]
        
        
        p,cov=np.polyfit(Idata[pos],Vdata_true[pos],deg=1,cov=True)
        if abs(p[0])>correlation_thresh*np.sqrt(cov[0,0]):
            slope[2]=p[0]
            error[2]=cov[0,0]
        
        slope=np.array(slope)*(-1)
    
    if outms is None:
        outms=msname.replace(".ms","_correlation_leak_corrected.ms")

    outms=correct_image_leakage(msname,slope,outms=outms)
    return outms
    
def remove_flags(caltable):
    tb=table()
    tb.open(caltable,nomodify=False)
    
    try:
        flag=tb.getcol('FLAG')
        flag[...]=False
        tb.putcol('FLAG',flag)
        tb.flush()
    finally:
        tb.close()
    return

def update_caltable(caltable,crosshand_phase):
    tb=table()
    tb.open(caltable,nomodify=False)
    try:
        data=tb.getcol('CPARAM')
        data[...]=np.cos(crosshand_phase)+1j*np.sin(crosshand_phase)
        tb.putcol('CPARAM',data)
        tb.flush()
    finally:
        tb.close()
    return

def apply_correction(msname,caltable):
    from casatasks import applycal
    applycal(vis=msname,gaintable=caltable,applymode='calonly')
    return
    
def crosshand_phase_optimising_func(crosshand_phase,msname=None,caltable=None,\
                                    thresh=7,correlation_thresh=10,sol_area=400,):
    update_caltable(caltable,crosshand_phase)
    apply_correction(msname,caltable)
    #imagename=msname.replace(".ms","_temp")
    imagename='/home/surajit/Downloads/lwa_polarisation_tests/solar_data/fresh_start/img_leak_crosshand_phase_variation/temp'+str(round(crosshand_phase,2))
    deconvolve.run_wsclean(msname,imagename,size=2048,scale='4arcmin',weight='briggs 0',niter=1000,pol='I,Q,U,V',predict=False,field='0',fast_vis=True)
    return
    me=measures()
    m = utils.get_sun_pos(msname, str_output=False)
    solar_ra=m['m0']['value']*180/np.pi
    solar_dec=m['m1']['value']*180/np.pi
    
    head=fits.getheader(imagename+"-I-image.fits")

    freq=head['CRVAL3']*1e-6
    imwcs=WCS(head) 
    solar_xy = imwcs.all_world2pix(list(zip([solar_ra], [solar_dec],[head['CRVAL3']],[head['CRVAL4']])), 1)   
    
    if head['cunit1'] == 'deg':
        dx = np.abs(head['cdelt1'] * 60.)
    else:
        print(head['cunit1'] + ' not recognized as "deg". Model could be wrong.')
    if head['cunit2'] == 'deg':
        dy = np.abs(head['cdelt2'] * 60.)
    else:
        print(head['cunit2'] + ' not recognized as "deg". Model could be wrong.')                   
    sol_area_xpix = int(sol_area / dx)
    sol_area_ypix = int(sol_area / dy)
    
    solx=int(solar_xy[0][0])
    soly=int(solar_xy[0][1])
    
    ymin=soly - sol_area_ypix // 2
    ymax=soly + sol_area_ypix // 2 + 1
    xmin=solx - sol_area_xpix // 2
    xmax=solx + sol_area_xpix // 2 + 1
    

    
    Idata=fits.getdata(imagename+"-I-image.fits")
    Udata=fits.getdata(imagename+"-U-image.fits")[0,0,ymin:ymax,xmin:xmax]
    Vdata=fits.getdata(imagename+"-V-image.fits")[0,0,ymin:ymax,xmin:xmax]
    
    rms=np.nanstd(Idata)
    pos=np.where(Idata<thresh*rms)
    Idata[pos]=np.nan
    
    Idata=Idata[0,0,ymin:ymax,xmin:xmax]
    
    solar_az,solar_el=utils.get_solar_azel(msname)        
    #factors=get_beam_factors([solar_el],[solar_az],freq=freq)
    factors=np.array([1, -0.03, 0.09, 0.0016])
    
    Udata_true=Udata-factors[0,2]*Idata
    Vdata_true=Vdata-factors[0,3]*Idata
    
    pos=np.where(np.isnan(Idata)==False)
    
    p,cov=np.polyfit(Udata_true[pos],Vdata_true[pos],deg=1,cov=True)
    print (p[0])
    if abs(p[0])>correlation_thresh*np.sqrt(cov[0,0]):
        return abs(p[0])
    return 0
    
    
def correct_crosshand_phase(msname):
    from casatasks import polcal
    
    caltable=msname.replace(".ms","_dummy.xf")
    if not os.path.isdir(caltable):
        polcal(vis=msname,caltable=caltable,poltype='Xf')
    remove_flags(caltable)
    
    res=minimize_scalar(crosshand_phase_optimising_func,bounds=(-3.14159,3.14159),\
                    method='bounded',args=(msname,caltable))
    print (res)
    
def correct_crosshand_phase_self(msname,crosshand_phase=0.0, inplace=False, outms=None):
    '''
    Here I apply the crosshand phase by hand. The crosshand phase suppliedto this function
    should be exactly equal to the crosshand_phase supplied to crosshand_phase_optimising_func.
    The way caltable is modified by that function is same as what is applied here.
    '''
    if outms is None:
        outms=msname.replace(".ms","_img_leak_corrected.ms")
    
    present=utils.check_corrected_data_present(msname)
    if present:
        datacolumn='CORRECTED'
    else:
        datacolumn='DATA'
        
    if not inplace:
        split(vis=msname,outputvis=outms,datacolumn=datacolumn)
        datacolumn='DATA'
        msname=outms
    
    if datacolumn=='CORRECTED':
        datacolumn='CORRECTED_DATA'
    
    tb=table()
    tb.open(msname,nomodify=False)
    try:
        data=tb.getcol(datacolumn)
        data[1,...]*=(np.cos(crosshand_phase)-1j*np.sin(crosshand_phase))
        data[2,...]*=(np.cos(crosshand_phase)+1j*np.sin(crosshand_phase))
        tb.putcol(datacolumn,data)
        tb.flush()
        success=True
    except:
        pass
    finally:
        tb.close()
    
    if not success:
        logging.warning("Crosshand phase correction "+\
                        "was not successfull. Please proceed with caution.")
    return msname
    


#TODO This function needs rewritting. 
def image_based_leakage_correction(msname,stokes='Q,U,V',factor=None,outms=None):
    '''
    This function implements an image based leakage correction.
    However, note that this function does not determine the factor
    by which the correction is done. It expects that the factor will
    be provided. By default he factor is None, which means nothing will be
    done. The order of the stokes parameters should be same as the order in
    which the correction factors are provided. The factors should be determined
    as follows:
    
    Qfrac_beam-Qfrac_observed
    Ufrac_beam-Ufrac_observed
    Vfrac_beam-Vfrac_observed
    
    :param msname: MS which has the data to be corrected
    :type msname: str
    :param stokes: Stokes parameters for which image based leakage correction
                    will be done; example "Q,U,V"
    :type stokes: str
    :param factor: the correction factor. Should in units of I. Should be a list
    :type factor: float
    :param outms: Name of output MS; optional
    :type outms: str
    '''
    
    if outms is None:
        outms=msname.replace(".ms","_image_leak_corrected.ms")
    
    present=utils.check_corrected_data_present(msname)
    if present:
        datacolumn='CORRECTED'
    else:
        datacolumn='DATA'
    
    split(vis=msname,outputvis=outms,datacolumn=datacolumn)
    
    if factor is None:
        return outms
    
    ind_stokes=stokes.split(',') ### contains the individual stokes parameters
    num_stokes=len(ind_stokes)
    
    if num_stokes!=len(factor):
        logging.warning("The length of correction factor does not "+\
                        "match the number of stokes parameters given."+\
                        " I will assume all reamining correction factors "+\
                        "to be zero. Please proceed with caution")
    
    
    
    for j,pol in enumerate(ind_stokes):
        if pol=='Q':
            correct_Q_leakage(outms,factor[j],inplace=True)
        if pol=='U':
            correct_U_leakage(outms,factor[j])
        if pol=='V':
            correct_V_leakage(outms,factor[j])
    return
    
    
