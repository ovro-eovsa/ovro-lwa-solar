#!/usr/bin/env python

# This script is adapted from Marin Anderson's script named /opt/astro/utils/bin/gen_model_ms.py on astm.lwa.ovro.caltech.edu
# History: 2022-07-27 B. Chen, changed the use of casacore to the modular CASA 6 (test on 6.5, 
#                              but earlier versions should be fine);
# 		  			           Also restructured it as part of the solar OVRO-LWA calibration package for future updates

from casatools import table, measures, componentlist
import math
import sys,os
import numpy as np
from casatasks import flagdata
import glob

tb = table()
me = measures()
cl = componentlist()


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


def gen_model_ms(visibility, ref_freq=80.0, output_freq=47.0,
                 includesun=True, solar_flux=16000, solar_alpha=2.2,
                 outputcl=None, verbose=True):
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
        srcs.append({'label': 'Sun', 'flux': string(solar_flux), 'alpha': solar_alpha,
                     'position': 'SUN'})

    tb.open(visibility)
    t0 = tb.getcell('TIME', 0)
    tb.close()
    # me.set_data_path('/opt/astro/casa-data')
    ovro = me.observatory('OVRO_MMA')
    time = me.epoch('UTC', '%fs' % t0)
    me.doframe(ovro)
    me.doframe(time)

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
            scale = math.sin(elev) ** 1.6
            print('scale {0:.1f}'.format(scale))
            srcs[s]['flux'] = flux80_47(float(srcs[s]['flux']), srcs[s]['alpha'],
                                        ref_freq=ref_freq, output_freq=output_freq) * scale

    cl.done()

    if not outputcl:
        outputcl = visibility.replace('.ms', '.cl')
    for s in srcs:
        cl.addcomponent(flux=s['flux'], dir=s['position'], index=s['alpha'],
                        spectrumtype='spectral index', freq='{0:f}MHz'.format(output_freq), label=s['label'])
        if verbose:
            print(
                "cl.addcomponent(flux=%s, dir='%s', index=%s, spectrumtype='spectral index', freq='47MHz', label='%s')" % (
                    s['flux'], s['position'], s['alpha'], s['label']))

    cl.rename(outputcl)
    cl.done()
    return outputcl


def flag_ants_from_postcal_autocorr(msfile: str, tavg: bool = False, thresh: float = 4):
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

def flag_bad(msfile,flagfile=None,thresh=None):
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
	files=glob.glob("defaults/*")
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


