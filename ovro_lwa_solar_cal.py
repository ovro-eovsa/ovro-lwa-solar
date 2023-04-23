#!/usr/bin/env python

# This script is adapted from Marin Anderson's script named /opt/astro/utils/bin/gen_model_ms.py on astm.lwa.ovro.caltech.edu
# History: 2022-07-27 B. Chen, changed the use of casacore to the modular CASA 6 (test on 6.5, 
#                              but earlier versions should be fine);
# 		  			           Also restructured it as part of the solar OVRO-LWA calibration package for future updates

from casatasks import clearcal, ft, bandpass, applycal, flagdata, tclean, uvsub
from casatools import table, measures, componentlist,msmetadata
import math
import sys,os
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.wcs import WCS
from astropy.io import fits
import glob


tb = table()
me = measures()
cl = componentlist()
msmd=msmetadata()


def flux80_47(flux_hi, sp, ref_freq=80., output_freq=47.):
    # given a flux at 80 MHz and a sp_index,
    # return the flux at 47 MHz.
    return flux_hi * 10 ** (sp * math.log(output_freq / ref_freq, 10))


def conv_deg(dec):
    if 's' in dec:
        dec = dec.split('s')[0]
    if 'm' in dec:
        dec, ss = dec.split('m')
        if ss == '':
            ss = '0'
    dd, mm = dec.split('d')
    if dd.startswith('-'):
        neg = True
    else:
        neg = False
    deg = float(dd) + float(mm) / 60 + float(ss) / 3600
    return '%fdeg' % deg

def get_sun_pos(solar_visibility,str_output=True):
    tb.open(solar_visibility)
    t0 = tb.getcell('TIME', 0)
    tb.close()
    ovro = me.observatory('OVRO_MMA')
    time = me.epoch('UTC', '%fs' % t0)
    me.doframe(ovro)
    me.doframe(time)
    d0 = me.direction('SUN')
    d0_j2000 = me.measure(d0, 'J2000')
    if str_output==True:
        d0_j2000_str = 'J2000 %frad %frad' % (d0_j2000['m0']['value'], d0_j2000['m1']['value'])
        return d0_j2000_str
    return d0_j2000


def gen_model_cl(visibility, ref_freq=80.0, output_freq=47.0,
                 includesun=False, solar_flux=16000, solar_alpha=2.2,
                 modelcl=None, verbose=True, overwrite=True):
    """
    :param visibility: input visibility
    :param ref_freq: reference frequency of the preset flux values of bright sources
    :param output_freq: output frequency to be written into the CASA component list
    :param includesun: if True, add a precribed solar flux to the source list
    :return:
    """
    srcs = [{'label': 'CasA', 'flux': '16530', 'alpha': -0.72,
             'position': 'J2000 23h23m24s +58d48m54s'},
            {'label': 'CygA', 'flux': '16300', 'alpha': -0.58,
             'position': 'J2000 19h59m28.35663s +40d44m02.0970s'},
            {'label': 'TauA', 'flux': '1770', 'alpha': -0.27,
             'position': 'J2000 05h34m31.94s +22d00m52.2s'},
            {'label': 'VirA', 'flux': '2400', 'alpha': -0.86,
             'position': 'J2000 12h30m49.42338s +12d23m28.0439s'}]
    if includesun:
        srcs.append({'label': 'Sun', 'flux': str(solar_flux), 'alpha': solar_alpha,
                     'position': 'SUN'})

    tb.open(visibility)
    t0 = tb.getcell('TIME', 0)
    tb.close()
    # me.set_data_path('/opt/astro/casa-data')
    ovro = me.observatory('OVRO_MMA')
    time = me.epoch('UTC', '%fs' % t0)
    me.doframe(ovro)
    me.doframe(time)
    msmd.open(visibility)
    chan_freqs = msmd.chanfreqs(0)  
    msmd.done()
    avg_freq=0.5*(chan_freqs[0]+chan_freqs[-1])*1e-6

    for s in range(len(srcs) - 1, -1, -1):
        coord = srcs[s]['position'].split()
        d0 = None
        if len(coord) == 1:
            d0 = me.direction(coord[0])
            d0_j2000 = me.measure(d0, 'J2000')
            srcs[s]['position'] = 'J2000 %frad %frad' % (d0_j2000['m0']['value'], d0_j2000['m1']['value'])
        elif len(coord) == 3:
            coord[2] = conv_deg(coord[2])
            d0 = me.direction(coord[0], coord[1], coord[2])
        else:
            raise Exception("Unknown direction")
        d = me.measure(d0, 'AZEL')
        elev = d['m1']['value']
        if elev < 0:
            del srcs[s]
        else:
            scale = math.sin(elev) ** 1.6*output_freq/avg_freq
            print('scale {0:.2f}'.format(scale))
            srcs[s]['flux'] = flux80_47(float(srcs[s]['flux']), srcs[s]['alpha'],
                                        ref_freq=ref_freq, output_freq=output_freq) * scale

    cl.done()

    if not modelcl:
        modelcl = visibility.replace('.ms', '.cl')
    for s in srcs:
        cl.addcomponent(flux=s['flux'], dir=s['position'], index=s['alpha'],
                        spectrumtype='spectral index', freq='{0:f}MHz'.format(output_freq), label=s['label'])
        if verbose:
            print(
                "cl.addcomponent(flux=%s, dir='%s', index=%s, spectrumtype='spectral index', freq='47MHz', label='%s')" % (
                    s['flux'], s['position'], s['alpha'], s['label']))
    if os.path.exists(modelcl) and overwrite:
        os.system('rm -rf ' + modelcl)
    cl.rename(modelcl)
    cl.done()
    return modelcl


def write_source_file(file_handle,source_name,primary_beam,source_num):  #### works only if logarithimicSI is false
    
    try:
        calfilepath='/home/surajit/ovro-lwa-solar/defaults/'
        f1=open(calfilepath+source_name+".txt","r")
        j=0
        
        while True:
            line=f1.readline()
            if not line:
                break
            if source_num==0 and j==0:
                file_handle.write(line)
            elif j!=0:
                try:
                    splitted=line.split(',')
                    I_flux=float(splitted[4])
                 
                    beam_corrected_I_flux=I_flux*primary_beam
                    splitted[4]=str(beam_corrected_I_flux)
                    
                    for k,phrase in enumerate(splitted[5:]):
                        if k==0:
                            splitted[5+k]='['+str(float(phrase[1:])*primary_beam)
                        else:
                            if phrase[-1]==']':
                                splitted[5+k]=str(float(phrase[:-1])*primary_beam)+']'
                                break
                            else:
                                splitted[5+k]=str(float(phrase)*primary_beam)
                    line1=','.join(splitted)
                    if splitted[5+k+1]=='false':
                        file_handle.write(line1)
                    else:
                        raise RuntimeError("Function now works only if logarithmicSI is false")
                except IndexError:
                    pass
                
            j+=1
    finally:
        f1.close()
    
def gen_model_ms(visibility, ref_freq=80.0, output_freq=47.0,
                 includesun=True, solar_flux=16000, solar_alpha=2.2,
                 outputcl=None, verbose=True,filename='calibrator_source_list.txt'):
    """
    :param visibility: input visibility
    :param ref_freq: reference frequency of the preset flux values of bright sources
    :param output_freq: output frequency to be written into the CASA component list
    :param includesun: if True, add a precribed solar flux to the source list
    :return:
    """
    
    srcs = [{'label': 'CasA', 'flux': '16530', 'alpha': -0.72,
             'position': 'J2000 23h23m24s +58d48m54s'},
            {'label': 'CygA', 'flux': '16300', 'alpha': -0.58,
             'position': 'J2000 19h59m28.35663s +40d44m02.0970s'},
            {'label': 'TauA', 'flux': '1770', 'alpha': -0.27,
             'position': 'J2000 05h34m31.94s +22d00m52.2s'},
            {'label': 'VirA', 'flux': '2400', 'alpha': -0.86,
             'position': 'J2000 12h30m49.42338s +12d23m28.0439s'}]
    
    
    
    if includesun:
        srcs.append({'label': 'Sun', 'flux': str(solar_flux), 'alpha': solar_alpha,
                     'position': 'SUN'})

    tb.open(visibility)
    t0 = tb.getcell('TIME', 0)
    tb.close()
    # me.set_data_path('/opt/astro/casa-data')
    ovro = me.observatory('OVRO_MMA')
    time = me.epoch('UTC', '%fs' % t0)
    me.doframe(ovro)
    me.doframe(time)

    msmd.open(visibility)
    chan_freqs = msmd.chanfreqs(0)  
    msmd.done()
    avg_freq=0.5*(chan_freqs[0]+chan_freqs[-1])*1e-6
    
    f1=open(filename,'w')
    num_source=0
    for s in range(len(srcs) - 1, -1, -1):
        coord = srcs[s]['position'].split()
        d0 = None
        if len(coord) == 1:
            d0 = me.direction(coord[0])
            d0_j2000 = me.measure(d0, 'J2000')
            srcs[s]['position'] = 'J2000 %frad %frad' % (d0_j2000['m0']['value'], d0_j2000['m1']['value'])
        elif len(coord) == 3:
            coord[2] = conv_deg(coord[2])
            d0 = me.direction(coord[0], coord[1], coord[2])
        else:
            raise Exception("Unknown direction")
        d = me.measure(d0, 'AZEL')
        elev = d['m1']['value']
        if elev < 0:
            del srcs[s]
        else:
            scale = math.sin(elev) ** 1.6*output_freq/avg_freq
            print (srcs[s]['label'])
            print('scale {0:.2f}'.format(scale))
            write_source_file(f1,srcs[s]['label'],scale,num_source)
            num_source+=1   
    
    return 


def flag_ants_from_postcal_autocorr(msfile: str, tavg: bool = False, thresh: float = 8):
    """Generates a text file containing the bad antennas.
    DOES NOT ACTUALLY APPLY FLAGS. CURRENTLY SHOULD ONLY BE RUN ON SINGLE SPW MSs.
    
    Args:
        msfile
        tavg: If set to True, will time average before evaluating flags.
        thresh: Threshold to use for flagging. Default is 4.
        
    Returns:
        Path to the text file with the list of antennas to flag.
    """
    tb.open(msfile)
    tautos = tb.query('ANTENNA1=ANTENNA2')
    tb.close()
    # get CORRECTED_DATA
    try:
    	autos_corrected = tautos.getcol('CORRECTED_DATA')
    except RuntimeError:
    	autos_corrected = tautos.getcol('DATA')
    autos_flags     = tautos.getcol('FLAG')
    autos_antnums   = tautos.getcol('ANTENNA1')
    shape=autos_corrected.shape
    # autos_corrected.shape = (Nants*Nints, Nchans, Ncorrs)
    if shape[2]>4:
    	autos_corrected=np.swapaxes(autos_corrected,0,2)
    	autos_flags=np.swapaxes(autos_flags,0,2)
    	print("Shape updated")
    print (autos_corrected.shape)
    Nants = np.unique(autos_antnums).shape[0]
    Nints = int(autos_antnums.shape[0]/Nants)
    Ncorrs = autos_corrected.shape[-1]
    # average over frequency, reorder
    autos_corrected_mask = np.ma.masked_array(autos_corrected, mask=autos_flags, 
                                           fill_value=np.nan)
    autos_tseries = np.ma.mean(autos_corrected_mask, axis=1).reshape(Nints, Nants, Ncorrs).transpose(1,0,2)
    antnums_reorder = autos_antnums.reshape(Nints, Nants).transpose(1,0)
    # autos_tseries.shape = (Nants, Nints, Ncorrs)
    # if msfile has Nints>1, use time series; else just take median
    if autos_tseries.shape[1] == 1:
        arr_to_evaluate = autos_tseries[:,0,:]
    elif tavg:
        arr_to_evaluate = np.ma.mean(autos_tseries,axis=1)
    else:
        medant_tseries  = np.ma.median(autos_tseries, axis=0)
        arr_to_evaluate = np.ma.std(autos_tseries/medant_tseries, axis=1)
    # separate out core and expansion antennas
    inds_core = list(range(0,56)) + list(range(64,120)) + list(range(128,184)) + list(range(192,238))
    inds_exp  = list(range(56,64)) + list(range(120,128)) + list(range(184,192)) + list(range(238,246))
    medval_core = np.ma.median(arr_to_evaluate[inds_core,:], axis=0)
    medval_exp = np.ma.median(arr_to_evaluate[inds_exp,:], axis=0)
    stdval_core = np.ma.std(arr_to_evaluate[inds_core,:], axis=0)
    stdval_exp = np.ma.std(arr_to_evaluate[inds_exp,:], axis=0)
    # find 3sigma outliers, exclude, and recalculate stdval
    newinds_core = np.asarray(inds_core)[np.where( (arr_to_evaluate[inds_core,0] < medval_core[0]+3*stdval_core[0]) | 
                         (arr_to_evaluate[inds_core,3] < medval_core[3]+3*stdval_core[3]) )]
    newinds_exp = np.asarray(inds_exp)[np.where( (arr_to_evaluate[inds_exp,0] < medval_exp[0]+3*stdval_exp[0]) | 
                         (arr_to_evaluate[inds_exp,3] < medval_exp[3]+3*stdval_exp[3]) )]
    # exclude and recalculate
    medval_core = np.ma.median(arr_to_evaluate[newinds_core,:], axis=0)
    medval_exp = np.ma.median(arr_to_evaluate[newinds_exp,:], axis=0)
    stdval_core = np.ma.std(arr_to_evaluate[newinds_core,:], axis=0)
    stdval_exp = np.ma.std(arr_to_evaluate[newinds_exp,:], axis=0)

    newflagscore = np.asarray(inds_core)[np.where( (arr_to_evaluate[inds_core,0] > medval_core[0]+thresh*np.ma.min(stdval_core)) | 
                         (arr_to_evaluate[inds_core,3] > medval_core[3]+thresh*np.ma.min(stdval_core)) )]
    newflagsexp = np.asarray(inds_exp)[np.where( (arr_to_evaluate[inds_exp,0] > medval_exp[0]+thresh*np.ma.min(stdval_exp)) | 
                         (arr_to_evaluate[inds_exp,3] > medval_exp[3]+thresh*np.ma.min(stdval_exp)) )]
    flagsall = np.sort(np.append(newflagscore,newflagsexp))
    print (flagsall.size)
    if flagsall.size > 0:
        antflagfile = os.path.splitext(os.path.abspath(msfile))[0]+'.ants'
        print (antflagfile)
        if os.path.exists(antflagfile):
            existingflags = np.genfromtxt(antflagfile, delimiter=',', dtype=int)
            flagsall = np.append(flagsall, existingflags)
            flagsall = np.unique(flagsall)
        flagsallstr = [str(flag) for flag in flagsall]        	
        flagsallstr2 = ",".join(flagsallstr)
        print (flagsallstr2)
        with open(antflagfile,'w') as f:
            f.write(flagsallstr2)
        return antflagfile
    else:
        return None

def flag_bad_ants(msfile,flagfile=None,thresh=None):
	if thresh==None:
		ants=flag_ants_from_postcal_autocorr(msfile)
	else:
		ants=flag_ants_from_postcal_autocorr(msfile,thresh=thresh)
	antflagfile = os.path.splitext(os.path.abspath(msfile))[0]+'.ants'
	if flagfile==None:
		flagfile=os.path.splitext(os.path.abspath(msfile))[0]+'.flagfile'
	if os.path.isfile(antflagfile):
		with open(antflagfile,'r') as f:
			antenna_list=f.readline()
		with open(flagfile,'w') as f:
			f.write("mode=\'manual\' antenna=\'"+antenna_list+"\'")
			f.write("\n")
	files=glob.glob("/home/surajit/ovro-lwa-solar/defaults/*")
	for file1 in files:
		with open(file1,"r") as f:
			lines=f.readlines()
		str1=''
		for i in lines:
			str1+=i[:-1].replace('&','&&')+";"
		str1=str1[:-1]
		with open(flagfile,'a+') as f:
			f.write("mode=\'manual\' antenna=\'"+str1+"\'")
			f.write("\n")
	flagdata(vis=msfile,mode='list',inpfile=flagfile)
	return

def gen_calibration(visibility, modelcl=None, uvrange='', bcaltb=None):
    """
    This function is for doing initial self-calibrations using strong sources that are above the horizon
    It is recommended to use a dataset observed at night when the Sun is not in the field of view
    :param visibility: input CASA ms visibility for calibration
    :param modelcl: input model of strong sources as a component list, produced from gen_model_cl()
    """
    if not modelcl or not (os.path.exists(modelcl)):
        print('Model component list does not exist. Generating one from scratch.')
        modelcl = gen_model_cl(visibility)

    #Put the component list to the model column
    clearcal(visibility, addmodel=True)
    ft(visibility, complist=modelcl, usescratch=True)
    # Now do a bandpass calibration using the model component list
    if not bcaltb:
        bcaltb = os.path.splitext(visibility)[0] + '.bcal'
    bandpass(visibility, caltable=bcaltb, uvrange=uvrange, combine='scan,field,obs', fillgaps=1)
    return bcaltb


def apply_calibration(solar_visibility, gaintable=None, doflag=False, do_solar_imaging=True,
                      imagename='test'):
    # first do some flagging with Surajit's flagging function. Now just clip the zeros
    #flagdata(solar_visibility, mode='clip', clipzeros=True)
    if doflag:
        flag_bad_ants(solar_visibility)
    if not gaintable:
        print('No calibration table is provided. Abort... ')
    else:
        if type(gaintable) == str:
            gaintable=[gaintable]
    # Apply the calibration
    clearcal(solar_visibility)
    applycal(solar_visibility, gaintable=gaintable, flagbackup=False)
    sunpos = get_sun_pos(solar_visibility)
    if do_solar_imaging:
        tclean(solar_visibility, imagename=imagename, imsize=[512], cell=['2arcmin'],
               weighting='uniform', phasecenter=sunpos, niter=500)
        print('Solar image made {0:s}.image'.format(imagename))

def make_fullsky_image(msfile,imagename="full_sky",imsize=4096,cell='2arcmin',minuv=10,niter=500): ### minuv: minimum uv in lambda
    os.system("wsclean -no-update-model-required -no-dirty -weight uniform"+\
            " -name "+imagename+" -size "+str(imsize)+" "+str(imsize)+" -scale "+cell+\
            " -minuv-l "+str(minuv)+" -niter "+str(niter)+" "+msfile)
            
def get_solar_loc_pix(msfile,image="full_sky"):
    m=get_sun_pos(msfile,str_output=False)
    ra=m['m0']['value']
    dec=m['m1']['value']
    coord=SkyCoord(ra*u.rad,dec*u.rad,frame='icrs')
    head=fits.getheader(image+"-model.fits")
    w=WCS(head)
    from astropy.wcs.utils import skycoord_to_pixel
    pix=skycoord_to_pixel(coord,w)
    x=int(pix[0])
    y=int(pix[1])
    return x,y

def remove_sun_from_model(msfile,image="full_sky",area=100):
    x,y=get_solar_loc_pix(msfile,image)
    print (x,y)
    head=fits.getheader(image+"-model.fits")
    data=fits.getdata(image+"-model.fits")
    data[0,0,y-area//2:y+area//2,x-area//2:x+area//2]=0.0000
    fits.writeto(image+"_sun_only-model.fits",data,header=head,overwrite=True)

def predict_model(msfile,outms,image="full_sky_sun_only"):
    os.system("cp -r "+msfile+" "+outms)
    os.system("wsclean -predict -name "+image+" "+outms)

def remove_all_sources(msfile,imagename='full_sky',imsize=4096,cell='2arcmin',minuv=10):
    make_fullsky_image(msfile=msfile,imagename=imagename,imsize=imsize,cell=cell,minuv=minuv)
    remove_sun_from_model(msfile,image=imagename)
    outms=msfile[:-3]+"_sun_only.ms"
    predict_model(msfile,outms=outms,image=imagename+"_sun_only")
    uvsub(outms)
    return outms
    
def make_solar_image(msfile,imagename='sun_only',imsize=512,cell='2arcmin',minuv=10):
	sunpos = get_sun_pos(msfile)
	tclean(msfile, imagename=imagename, imsize=imsize, cell=cell,
              weighting='uniform', phasecenter=sunpos, niter=1000,\
               usemask='auto-multithresh')
               
def correct_ms_bug(msfile):
    tb.open(msfile+"/SPECTRAL_WINDOW",nomodify=False)
    meas_freq_ref=tb.getcol('MEAS_FREQ_REF')
    if meas_freq_ref[0]==0:
        meas_freq_ref[0]=1
    tb.putcol('MEAS_FREQ_REF',meas_freq_ref)
    tb.flush()
    tb.close()
    
def pipeline(solar_ms,calib_ms=None, bcal=None):
    if bcal==None:
        bcal=gen_calibration(calib_ms,uvrange='')
    correct_ms_bug(solar_ms)
    apply_calibration(solar_ms, gaintable=bcal, doflag=True, do_solar_imaging=False)
    outms=remove_all_sources(solar_ms)
    make_solar_image(outms)
