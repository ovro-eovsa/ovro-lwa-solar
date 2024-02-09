import os, sys
from astropy.time import Time
from astropy.io import fits
from casatools import image, table, msmetadata, quanta, measures
import numpy as np
import logging, glob
from . import primary_beam
from casatasks import split
from .primary_beam import analytic_beam as beam 
from . import utils
from . import generate_calibrator_model
from . import selfcal


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
                                caltable_folder=caltable_folder)
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

def correct_Q_leakage(msname,factor, inplace=False, outms=None):
    '''
    This function implements a constant leakage correction for Q
    visibilities. If corrected data column is present, the corrected
    column is modified. If not, datacolumn is modified. However if
    inplace is set to False, then a MS is split and then the correction
    is done. The correction factor should be determined as
    
    Qfrac_beam-Qfrac_observed
    
    :param msname: MS to be corrected
    :type param: str
    :param factor: Correction factor
    :type factor: float
    :param inplace: Determines whether correction will be done inplace; optional
                    default: False, which means, by default the input MS will not be
                    modified.
    :type inplace: Boolean
    :param outms: If inplace is False, then outms is created and the data is modified.
    :type outms: str
    '''  
    
      
    if outms is None:
        outms=msname.replace(".ms","_Qleak_corrected.ms")
    
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
    tb=table()
    tb.open(msname,nomodify=False)
    try:
        data=tb.getcol(datacolumn)
        data[0,...]+=0.25*factor*(data[0,...]+data[3,...])
        data[3,...]+=-0.25*factor*(data[0,...]+data[3,...])
        tb.putcol(datacolumn,data)
        tb.flush()
        success=True
    except:
        pass
    finally:
        tb.close()
    
    if not success:
        logging.warning("Image based leakage correction of Stokes Q "+\
                        "was not successfull. Please proceed with caution.")
    return msname
 
    
        
    
    
    
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
    
    
