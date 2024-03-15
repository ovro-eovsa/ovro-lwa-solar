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
from ovrolwasolar import deconvolve
from scipy.optimize import minimize

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
    #return np.array([[ 1.00000000e+00, -1.58626752e-01,  3.69086757e-03,  3.89909857e-04]])
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

        
def add_solar_model(imagename,msfile,sol_area=400., overwrite_solar_region=True):  
    '''
    This will add the polarised solar model, by scaling the Stokes I model. 
    
    :param imagename:  imagename should be standard WSclean format. Ensure that 
                        imagename-{I,Q,U,V}-image.fits and -model.fits exist
    :type imagename: str
    :param msfile: Name of MS. This is used to get the solar ra-dec
    :type msfile: str
    :param sol_area: This is area of the sky around the sun, inside which
                        sun is guaranteed to be present. This is the diameter
                        of the region
    :type sol_area: float
    :param overwrite_solar_region: If True, the model at this location will be
                                    overwritten . Default: True
                                    Else, the solar model scaled from Stokes I 
                                    will be added to it.
    :type overwrite_solar_region: Bool
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
            
            if overwrite_solar_region:
                data[0, 0, soly - sol_area_ypix // 2:soly + sol_area_ypix // 2 + 1,
                    solx - sol_area_xpix // 2:solx + sol_area_xpix // 2 + 1]=0.0
                    
            data[0, 0, soly - sol_area_ypix // 2:soly + sol_area_ypix // 2 + 1,
                    solx - sol_area_xpix // 2:solx + sol_area_xpix // 2 + 1] += \
                                            img_data[0, 0, soly - sol_area_ypix // 2:soly + sol_area_ypix // 2 + 1,
                                            solx - sol_area_xpix // 2:solx + sol_area_xpix // 2 + 1]*factors[0,j+1]
            hdu[0].data=data
            hdu.flush()
        finally:
            hdu.close()
            

    


def get_high_snr_image_data(imagename,stokes, msfile,sun_only=False, thresh=7, rms_filter=True,\
                            sol_area=400., correct_beam_leakage=True, min_pix=0):
    '''
    This function is used to get high SNR stokes data. 
    
    :param imagename: Prefix of the image. This means imagename-I-image.fits, imagename-Q-image.fits etc should exist
    :type imagename: str
    :param stokes: Stokes of the image. Can be either Q,U,V
    :type stokes: str
    :param msfile: MS name, which is used to determine the solar loc
    :type msfile: str
    :param sol_area: Diameter of region around Sun which will be used to calculate correction factor
    :type sol_area: integer
    :param thresh: Threshold in terms of rms which will be used to determine a source detection in Stokes images.
                    
    :type thresh: float
    :param sun_only: Other sources will be ignored when finding the leakage correction factor.
    :type sun_only: Boolean. Default: False
    
    :param rms_filter: Only pass points which have high SNR in both Stokes I and Stokes image. Default: True
                       If False, pass the full data back, with no filtering
    :type rms_filter: True
    
    :param sol_area: The diameter around solar ra-dec which should be considered as solar area. It is in arcminutes
    :type sol_area: float
    
    :param correct_beam_leakage: If True, correct beam leakage and then return the Stokes data. Default: True
    :type correct_beam_leakage: Boolean
    
    :return: Stokes I and Stokes parameter supplied in input stokes. Will return only the high SNR points
            returns None, None if no high SNR is present
    :rtype: numpy.ndarray, numpy.ndarray
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
    
    
    
    solx=int(solar_xy[0][0])
    soly=int(solar_xy[0][1])
    
    img=imagename+"-"+stokes+"-image.fits"
    
    img_data=fits.getdata(img)
    
    
    
    
    Irms=utils.get_rms(Idata)
    
    
    if sun_only:
        Idata=Idata[0, 0, soly - sol_area_ypix // 2:soly + sol_area_ypix // 2 + 1,
                    solx - sol_area_xpix // 2:solx + sol_area_xpix // 2 + 1]
        img_data=img_data[0, 0, soly - sol_area_ypix // 2:soly + sol_area_ypix // 2 + 1,
                    solx - sol_area_xpix // 2:solx + sol_area_xpix // 2 + 1]
    
    
    
    
    solar_az,solar_el=utils.get_solar_azel(msfile)        
    
    if correct_beam_leakage:
        factors=get_beam_factors([solar_el],[solar_az],freq=freq)
    else:
        factors=np.array([[1,0,0,0]])
    
    if stokes=='Q':
        solar_model_frac=factors[0,1]
    elif stokes=='U':
        solar_model_frac=factors[0,2]
    elif stokes=='V':   
        solar_model_frac=factors[0,3]
        
    img_data=img_data-solar_model_frac*Idata
    if rms_filter:
        stokes_rms=utils.get_rms(img_data)
        pos=np.where((Idata>thresh*Irms) & (abs(img_data)>thresh*stokes_rms))
        if pos[0].size<=min_pix:
            return None, None
        return Idata[pos], img_data[pos]
    else:
        return Idata,img_data

def get_img_correction_factor_minimize_correlation(imagename,stokes,msfile,\
                                sun_only=True, sol_area=400.,\
                                thresh=7):
    '''
    This function determines the correction factor by minimising the correlation of I and other Stokes images.
    For the Sun, we choose the high SNR points in a 400arcmin region around the Sun. The way the
    correction fraction is determined implies that the corrected image should be
    Q_corrected=Q_obs+correction_factor*Iobs
    U_corrected=U_obs+correction_factor*Iobs
    V_corrected=V_obs+correction_factor*Iobs
    
    All the paremeters are passed to the function get_high_snr_image_data for getting the image data. Only those
    
    Please also ensure that the Stokes I and other parameters are correlated before running 
    this function for that Stokes parameter.
    
    :param imagename: Prefix of the image. This means imagename-I-image.fits, imagename-Q-image.fits etc should exist
    :type imagename: str
    :param stokes: Stokes of the image. Can be either Q,U,V
    :type stokes: str
    :param msfile: MS name, which is used to determine the solar loc
    :type msfile: str
    :param sol_area: Diameter of region around Sun which will be used to calculate correction factor
    :type sol_area: integer
    :param thresh: Threshold in terms of rms which will be used to determine a source detection in Stokes images
    :type thresh: float
    :param sun_only: Other sources will be ignored when finding the leakage correction factor.
    :type sun_only: Boolean. Default: True
    '''
    
    Idata, img_data=get_high_snr_image_data(imagename,stokes, msfile,sun_only=sun_only, thresh=thresh,\
                                            sol_area=sol_area)
    
    if Idata is None and img_data is None:
        logging.info("No high SNR point detected. There are only 3 possibilities. Either you have already "+\
                        "run the correction step; or you are extremely lucky and the leakages are already "+\
                        "at minimum level, or Stokes I image is junk and no source has been detected. "+\
                        "I suggest that you confirm that the third case is not true")
        return 0.0
    
    func1=lambda a,stokes_data,Idata: np.abs(np.corrcoef(stokes_data+a*Idata,Idata)[0,1])
    
    res=minimize(func1,x0=[0.0],args=(img_data.flatten(),Idata.flatten()),\
                    method='Nelder-Mead',bounds=[[-1,1]])
    
    if res.success:
        return res.x[0]
    logging.warning("Solution was not found. Proceed with care")
    return 0.0
            
            
    
    

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
    :return: outms
    :rtype: str
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
    
def crosshand_phase_optimising_func(crosshand_phase, Udata=None, Vdata=None, return_corrected=False):
    '''
    This function applies a rotation to the Stokes vector by an angle
    equal to the crosshand_phase, and the rotation axis is along the 
    Q vector. The idea is that at the right crosshand phase, the magnitude
    of U will be minimum and also the correlation coefficient between U and V 
    will be minimum. It returns the absolute correlation coefficient
    between the corrected U and V, by default. If return_corrected is set to 
    True, then the function will return the corrected U and V only. 
    
    :param Udata: Stokes U data. This exclude only pixels where either Stokes
                  U or Stokes V is detected. In general we are also following
                  the logic and at these locations Stokes I should be detected.   
                  But strictly speaking, this is not necessary.
    :type Udata : numpy ndarray; Can be slices as well.
    :param Vdata: Stokes V data
    :type Vdata: numpy ndarray
    :param crosshand_phase: This is the rotation angle of the Stokes vector. 
                            unit: radian
    :type crosshand_phase: float
    :param return_corrected: Boolean to choose if correlation coefficient between
                             U and V is needed, or the corrected U and V is needed.
                             Default: False (correlation coeffcient will be returned)
    :type return_corrected: Boolean
    :return : correlation coefficient between corrected U and V 
              
    '''
    Udata_corrected= Udata*np.cos(crosshand_phase) + Vdata*np.sin(crosshand_phase)
    Vdata_corrected= -Udata*np.sin(crosshand_phase) + Vdata*np.cos(crosshand_phase)
    
    if not return_corrected:
        return np.abs(np.corrcoef(Udata_corrected,Vdata_corrected)[0,1])
    else:
        return Udata_corrected, Vdata_corrected
    

def find_image_crosshand_phase(msfile, imagename=None, Ithresh=7, stokes_thresh=3, solar_case=True, sol_area=400.):
    '''
    The function determines the crosshand phase by minimising the correlation between Stokes U and V
    The idea is that at the right crosshand phase not only is the correlation minimum, but the Stokes U
    is minimum (solar case) or Stokes V is minimum (non-solar case). There are 4 possible solutions for
    which the correlation is minimum. It is easy to see from the rotation matrix that if x is a solution,
    pi-x is also a solution. Because pi-x just reflects the Stokes vector across the origin and hence
    will not change the correlation in any way. The other two solutions are pi/2+x, pi/2-x, which will
    basically rotate the vector by +-pi/2 . 
    
    For this calculation, I will only use pixels which satisfy these two conditions:
    1. The pixels should have detectable signal in Stokes I. Detectable means that the SNR should be 
       at least Ithresh.
    2. The pixels should be detected in at least one of U and V. Again detectable means the SNR should
       be at least stokes_thresh
    
    :param msfile: Name of the MS.
    :type msfile: str
    :param imagename: Name of the image. Default: None. If not provided or not present, image will be generated
    :type imagename: str
    :param Ithresh: threshold in I image for source detection
    :type thresh: float
    :param stokes_thresh:  threshold in Stokes image for source detection
    :type stokes_thresh: float
    :param solar_case: If True, Stokes U will be minimised. If False, Stokes V will be minimised. Default: True
    :type solar_case: Bool
    :return :crosshand_phase in radians
    :rtype: float
    '''
    if imagename is None:
        imagename=msfile.replace('.ms','_crosshand')
    
    if not os.path.isfile(imagename+"-I-image.fits") and \
       not os.path.isfile(imagename+"-Q-image.fits") and \
       not os.path.isfile(imagename+"-U-image.fits") and \
       not os.path.isfile(imagename+"-V-image.fits") :
        deconvolve.run_wsclean(msfile,imagename,pol='I,Q,U,V',niter=1000, minuv_l=30, predict=False)
        
    Idata, Udata=get_high_snr_image_data(imagename,'U', msfile, sun_only=True, thresh=Ithresh, rms_filter=False,\
                                        sol_area=sol_area)
    Idata,Vdata=get_high_snr_image_data(imagename,'V', msfile, sun_only=True, thresh=Ithresh, rms_filter=False,\
                                        sol_area=sol_area)
    
    Udata_flat=Udata.flatten()
    Vdata_flat=Vdata.flatten()
    Idata_flat=Idata.flatten()
    
    Irms=utils.get_rms(Idata)
    pos=np.where(Idata_flat>Ithresh*Irms)[0]
    
    if pos[0].size==0:
        logging.info("No source in Stokes I !!! Did you subtract all sources? Proceed with care")
        return 0.0
        
    
    Upos=np.where(np.abs(Udata_flat)>stokes_thresh*utils.get_rms(Udata_flat))[0]
    Vpos=np.where(np.abs(Vdata_flat)>stokes_thresh*utils.get_rms(Vdata_flat))[0]
    
    pos=np.intersect1d(np.union1d(Upos,Vpos),pos)
    
    if len(pos)==0:
        logging.info("No detectable source in U and V.")
        return 0.0
    
    Udata_flat=Udata_flat[pos]
    Vdata_flat=Vdata_flat[pos]
    
    ### subtracting the mean values out, as they are in anycase do not change the
    ### correlation coefficient.
    Udata_flat-=np.nanmedian(Udata_flat)
    Vdata_flat-=np.nanmedian(Vdata_flat)
    
    res=minimize(crosshand_phase_optimising_func,x0=[0.0],\
                    args=(Udata_flat,Vdata_flat),\
                    method='Nelder-Mead',bounds=[[-3.14159,3.14159]])
    
    if not res.success:
        logging.warning("Solution was not found. Proceed with care")
        return 0.0
        
    solutions=[res.x]
    
    Ucor,Vcor= crosshand_phase_optimising_func(res.x,Udata_flat,Vdata_flat, return_corrected=True)
    Udata_corrected=[np.nansum(Ucor**2)]
    Vdata_corrected=[np.nansum(Vcor**2)]
    
    for angle in [3.14159-res.x,res.x+3.14159/2, -res.x+3.14159/2]:  ### 3.14159: pi
        solutions.append(angle)
        Ucor,Vcor= crosshand_phase_optimising_func(angle,Udata_flat,Vdata_flat, return_corrected=True)
        Udata_corrected.append(np.nansum(Ucor**2))
        Vdata_corrected.append(np.nansum(Vcor**2))
            
    solutions=np.array(solutions)
    Udata_corrected=np.array(Udata_corrected)
    Vdata_corrected=np.array(Vdata_corrected)
    
    print (solutions)
    print (Udata_corrected)
    print (Vdata_corrected)
    
    if solar_case:
        return solutions[np.argmin(Udata_corrected)][0]
    else:
        return solutions[np.argmin(Vdata_corrected)][0]
    
    logging.warning("Do not know why I came here. Some issue is there. Please check")
    return 0.0
    
    
def crosshand_phase_minimising_func(crosshand_phase,data,model=None,return_corrected=False):
    xy_data=data[1,...]
    yx_data=data[2,...]
    
    xy_corrected=np.exp(-1j*crosshand_phase)*xy_data
    yx_corrected=np.exp(1j*crosshand_phase)*yx_data
    
    if not return_corrected:
        xy_model=model[1,...]
        yx_model=model[2,...]
        sum1=np.sqrt(np.nansum((np.abs(xy_corrected-xy_model))**2)+ np.nansum((np.abs(yx_corrected-yx_model))**2))
        print (sum1)
        return sum1
    else:
        data[1,...]=xy_corrected
        data[2,...]=yx_corrected
        return
    
    
def solve_crosshand_phase(msname):

    tb=table()
    
    corrected_data_present=utils.check_corrected_data_present(msname)
    
    if corrected_data_present:
        datacolumn='CORRECTED_DATA'
    else:
        datacolumn='DATA'
        
    tb.open(msname)
    try:
        data=tb.getcol(datacolumn)
        flag=tb.getcol('FLAG')
        model=tb.getcol('MODEL_DATA')
        uvw=tb.getcol('UVW')
        u=uvw[0,:]
        v=uvw[1,:]
    finally:
        tb.close()
        
    pos=np.where(flag==True)
    data[pos]=np.nan
    
    
    res=minimize(crosshand_phase_minimising_func,x0=[0.3],args=(data,model),\
                    method='Nelder-Mead',bounds=[[-3.14159,3.14159]])
    if not res.success:
        logging.warning("Crosshand phase visibility based solution was not found."+\
                            " Proceed with care")
        return 0.0
    
    else:
        return res.x[0]
        
def correct_crosshand_phase(msname,crosshand_phase=None, outms=None, inplace=False):

    tb=table()
    if crosshand_phase is None:
        crosshand_phase=solve_crosshand_phase(msname)
    
    corrected_data_present=utils.check_corrected_data_present(msname)
    
    if corrected_data_present:
        datacolumn='CORRECTED'
    else:
        datacolumn='DATA'
    if not inplace:
        if outms is None:
            outms=msname.replace('.ms','_crossphase.ms')
        split(vis=msname,outputvis=outms,datacolumn=datacolumn)
        datacolumn='DATA'
        msname=outms
    
    if datacolumn=='CORRECTED':
        datacolumn='CORRECTED_DATA'
        
    tb.open(msname,nomodify=False)
    try:
        data=tb.getcol(datacolumn)
        flag=tb.getcol('FLAG')
        pos=np.where(flag==True)
        data[pos]=np.nan
        crosshand_phase_minimising_func(crosshand_phase,data, return_corrected=True)
        pos=np.where(np.isnan(data)==True)
        flag[pos]=True
        data[pos]=0.00
        tb.putcol('DATA',data)
        tb.putcol('FLAG',flag)
        tb.flush()
    finally:
        tb.close()
    return outms

def check_image(imagename,msname, Ithresh=7, crosshand_phase=None,stokes_thresh=5, outimage='outfile.png'):
    rms=[None]*4
    for j,stokes in enumerate(['I','Q','U','V']):
        imname=imagename+"-"+stokes+"-image.fits"
        data=fits.getdata(imname)
        rms[j]=utils.get_rms(data)
        
    Idata,Udata=get_high_snr_image_data(imagename,'U', msname, sun_only=True,rms_filter=False, \
                                        thresh=Ithresh)
    Idata,Vdata=get_high_snr_image_data(imagename,'V', msname, sun_only=True,rms_filter=False,\
                                        thresh=Ithresh)
    Idata,Qdata=get_high_snr_image_data(imagename,'Q', msname, sun_only=True,rms_filter=False,\
                                        thresh=Ithresh)                                    
    
    
    if crosshand_phase is not None:
        Udata_corrected,Vdata_corrected=crosshand_phase_optimising_func(crosshand_phase,\
                                        Udata,Vdata,return_corrected=True)
    else:
        Udata_corrected,Vdata_corrected=Udata,Vdata
    
    pos=np.where(Idata<Ithresh*rms[0])
    Idata[pos]=np.nan
    Qdata[pos]=np.nan
    Udata_corrected[pos]=np.nan
    Vdata_corrected[pos]=np.nan
    pos=np.where(Idata>Ithresh*rms[0])
    xmin,xmax=min(pos[1]),max(pos[1])
    ymin,ymax=min(pos[0]),max(pos[0])
    
    Qdata[Qdata<stokes_thresh*rms[1]]=np.nan
    Udata_corrected[Udata_corrected<stokes_thresh*rms[2]]=np.nan
    Vdata_corrected[Vdata_corrected<stokes_thresh*rms[3]]=np.nan
    
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('agg')
    fig,ax=plt.subplots(nrows=2,ncols=2,sharex=True,sharey=True)
    ax=ax.flatten()
    ax[0].imshow(Idata,origin='lower')
    ax[1].imshow(Qdata,origin='lower')
    ax[2].imshow(Udata,origin='lower')
    ax[3].imshow(Vdata,origin='lower')
    
    ax[0].set_xlim([xmin-30,xmax+30])
    ax[1].set_ylim([ymin-30,ymax+30])
    
    
    plt.savefig(outimage)
    plt.close()
    return

    
    
    
def do_polarisation_calibration(msname):
    if msname[:-1]=='/':
        msname=msname[:-1]
    imagename=msname[:-3]
    orig_image=imagename
    orig_ms=msname
    
    deconvolve.run_wsclean(msname,imagename=imagename,niter=1000,size=4096,\
                            scale='2arcmin',predict=False,pol='I,Q,U,V',\
                            weight='briggs 0',minuv_l=30)
    factor=[None]*3
    for j,stokes in enumerate(['Q','U','V']):
        factor[j]=get_img_correction_factor_minimize_correlation(imagename,stokes,msname)
    
    outms=msname.replace('.ms','_imgleak.ms')
    outms=correct_image_leakage(msname,factor)
    
    deconvolve.run_wsclean(outms,imagename=outms[:-3],niter=1000,size=4096,\
                            scale='2arcmin',predict=False,pol='I,Q,U,V',\
                            weight='briggs 0',minuv_l=30)
                      
    crosshand_phase=find_image_crosshand_phase(msname,outms[:-3])
    
    polarised_model_needed=check_image(outms[:-3],msname, \
                            crosshand_phase=crosshand_phase,\
                            outimage="image_after_crossphase.png")
                            
    print ("Check image_after_crossphase.png in your current directory. If you think "+\
            "that there is a significantly polarised source, then contact the developer "+\
            " for how to proceed. Handling this is under test and not yet supported. If "+\
            "significantly polarised source is not present, type n/no as answer to next "+\
            "question and the code will proceed\n")
    
    answer=input("Polarised source?")
    if answer!='n' and answer!='no':
        raise RuntimeError("User thinks that source is polarised. This is not supported yet")
    
    for stokes in ['Q','U','V']:
        imname=orig_image+"-"+stokes+"-model.fits"
        utils.blank_all_pixels(imname)
    
    add_solar_model(orig_image,msname)
    crosshand_phase=solve_crosshand_phase(msname)
    outms=correct_crosshand_phase(orig_ms,crosshand_phase=crosshand_phase)
    
    deconvolve.run_wsclean(outms,imagename=outms[:-3],niter=1000,size=4096,\
                            scale='2arcmin',predict=False,pol='I,Q,U,V',\
                            weight='briggs 0',minuv_l=30)
    
    factor=[None]*3
    for j,stokes in enumerate(['Q','U','V']):
        factor[j]=get_img_correction_factor_minimize_correlation(outms[:-3],stokes,msname)
    
    
    outms=correct_image_leakage(outms,factor)
    
    deconvolve.run_wsclean(outms,imagename=outms[:-3],niter=1000,size=4096,\
                            scale='2arcmin',predict=False,pol='I,Q,U,V',\
                            weight='briggs 0',minuv_l=30)
    
    check_image(outms[:-3],msname, outimage='final_image.png')
    return outms
    
    
    
    
    
    
        
    
    
    
    
    
    
    
    

    



    
    
