"""
Pipeline for calibrating and imaging solar data.
It was initially adapted from Marin Anderson's script named /opt/astro/utils/bin/gen_model_ms.py
    on astm.lwa.ovro.caltech.edu in August 2022
Certain functions are adapted from the orca repository at https://github.com/ovro-lwa/distributed-pipeline

Requirements:
- A modular installation of CASA 6: https://casadocs.readthedocs.io/en/stable/notebooks/introduction.html#Modular-Packages
- A working version of wsclean for imaging (i.e., "wsclean" defined in the search path)
"""

from casatasks import clearcal, ft, bandpass, applycal, flagdata, tclean, flagmanager, uvsub
from casatools import table, measures, componentlist, msmetadata
import math
import sys,os
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.wcs import WCS
from astropy.io import fits
import matplotlib.pyplot as plt

tb = table()
me = measures()
cl = componentlist()


def flux80_47(flux_hi, sp, ref_freq=80., output_freq=47.):
    """
    Given a flux at 80 MHz and a sp_index, return the flux at 47 MHz.
    :param flux_hi: flux at the reference frequency
    :param sp: spectral index
    :param ref_freq: reference frequency in MHz
    :param output_freq: output frequency in MHz
    :return: flux caliculated at the output frequency
    """
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


def get_sun_pos(msfile, str_output=True):
    """
    Return J2000 RA and DEC coordinates of the solar disk center
    :param msfile: input CASA measurement set
    :param str_output: if True, return coordinate in string form acceptable by CASA tclean
        if False, return a dictionary in CASA measures format: https://casa.nrao.edu/docs/casaref/measures.measure.html
    :return: solar disk center coordinate in string or dictionary format
    """
    tb.open(msfile)
    t0 = tb.getcell('TIME', 0)
    tb.close()
    ovro = me.observatory('OVRO_MMA')
    time = me.epoch('UTC', '%fs' % t0)
    me.doframe(ovro)
    me.doframe(time)
    d0 = me.direction('SUN')
    d0_j2000 = me.measure(d0, 'J2000')
    if str_output == True:
        d0_j2000_str = 'J2000 %frad %frad' % (d0_j2000['m0']['value'], d0_j2000['m1']['value'])
        return d0_j2000_str
    return d0_j2000


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


def get_antids(msfile):
    """
    Read antenna ids from a measurement set and separate them to inner and expansion ones
    :param msfile: path to CASA measurement set
    :return: antenna ids for core antennas and expansion antennas
    """
    tb.open(msfile + '/ANTENNA')
    ms_ant_names = tb.getcol('NAME')
    tb.close()
    msmd = msmetadata()
    msmd.open(msfile)
    core_ant_name_list = ['LWA{0:03d}'.format(i + 1) for i in range(0, 256)]
    exp_ant_name_list = ['LWA{0:03d}'.format(i + 1) for i in range(256, 366)]
    core_ant_ids = []
    exp_ant_ids = []
    for ms_ant_name in ms_ant_names:
        if ms_ant_name in core_ant_name_list:
            core_ant_ids.append(msmd.antennaids(ms_ant_name)[0])
        if ms_ant_name in exp_ant_name_list:
            exp_ant_ids.append(msmd.antennaids(ms_ant_name)[0])

    msmd.close()
    return np.array(core_ant_ids), np.array(exp_ant_ids)

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
            scale = math.sin(elev) ** 1.6
            print (srcs[s]['label'])
            print('scale {0:.2f}'.format(scale))
            write_source_file(f1,srcs[s]['label'],scale,num_source)
            num_source+=1   
    
    return 

def gen_model_cl(msfile, ref_freq=80.0, output_freq=47.0,
                 includesun=False, solar_flux=16000, solar_alpha=2.2,
                 modelcl=None, verbose=True, overwrite=True):
    """
    Generate source models for bright sources as CASA clean components
    :param msfile: input visibility
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

    tb.open(msfile)
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
            print('scale {0:.3f}'.format(scale))
            srcs[s]['flux'] = flux80_47(float(srcs[s]['flux']), srcs[s]['alpha'],
                                        ref_freq=ref_freq, output_freq=output_freq) * scale

    cl.done()

    if not modelcl:
        modelcl = msfile.replace('.ms', '.cl')
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


def gen_ant_flags_from_autocorr(msfile, antflagfile=None, datacolumn='DATA', tavg=False,
                                thresh_core=1.0, thresh_exp=1.0, flag_exp_with_core_stat=True,
                                doappend=False, debug=False, doplot=False):
    """Generates a text file containing the bad antennas.
    DOES NOT ACTUALLY APPLY FLAGS. CURRENTLY SHOULD ONLY BE RUN ON SINGLE SPW MSs.

    Adapted from the flag_ants_from_postcal_autocorr() module in
    https://github.com/ovro-lwa/distributed-pipeline/blob/main/orca/flagging/flag_bad_ants.py
    
    Args:
        :param msfile: string
        :param datacolumn: specify which data column to use. Default to "DATA".
            Could be "CORRECTED_DATA" if the dataset is calibrated
        :param tavg: If set to True, will time average before evaluating flags.
        :param thresh: Threshold to use for flagging. Default is 4.
        :param antflagfile: Output file that contains the flagged antennas. If not defined, use msfile.replace('.ms', 'antflags')

    Returns:
        Path to the text file with the list of antennas to flag (antflagfile).
    """
    tb.open(msfile)
    tautos = tb.query('ANTENNA1=ANTENNA2')
    tb.close()
    msmd = msmetadata()
    msmd.open(msfile)
    # get data, either DATA or CORRECTED_DATA
    autos = tautos.getcol(datacolumn)
    autos_flags = tautos.getcol('FLAG')
    autos_antnums = tautos.getcol('ANTENNA1')
    shape = autos.shape
    # autos_corrected.shape = (Nants*Nints, Nchans, Ncorrs)
    if shape[2] > 4:
        autos = np.swapaxes(autos, 0, 2)
        autos_flags = np.swapaxes(autos_flags, 0, 2)
        print("Shape updated")
    print(autos.shape)
    Nants = np.unique(autos_antnums).shape[0]
    Nints = int(autos_antnums.shape[0] / Nants)
    Ncorrs = autos.shape[-1]
    # average over frequency, reorder
    autos_corrected_mask = np.ma.masked_array(autos, mask=autos_flags,
                                              fill_value=np.nan)
    # take average in channel
    autos_tseries = np.ma.mean(autos_corrected_mask, axis=1).reshape(Nints, Nants, Ncorrs).transpose(1, 0, 2)
    antnums_reorder = autos_antnums.reshape(Nints, Nants).transpose(1, 0)
    # autos_tseries.shape = (Nants, Nints, Ncorrs)
    # if msfile has Nints>1, use time series; else just take median
    if autos_tseries.shape[1] == 1:
        arr_to_evaluate = autos_tseries[:, 0, :]
    elif tavg:
        arr_to_evaluate = np.ma.mean(autos_tseries, axis=1)
    else:
        medant_tseries = np.ma.median(autos_tseries, axis=0)
        arr_to_evaluate = np.ma.std(autos_tseries / medant_tseries, axis=1)

    autos_ampdb = 10. * np.log10(np.abs(arr_to_evaluate / 1.e2))
    print('shape of arr_to_evaluate', arr_to_evaluate.shape)
    # separate out core and expansion antennas
    # inds_core = list(range(0, 56)) + list(range(64, 120)) + list(range(128, 184)) + list(range(192, 238))
    # inds_exp = list(range(56, 64)) + list(range(120, 128)) + list(range(184, 192)) + list(range(238, 246))
    inds_core, inds_exp = get_antids(msfile)
    medval_core = np.ma.median(autos_ampdb[inds_core, :], axis=0)
    medval_exp = np.ma.median(autos_ampdb[inds_exp, :], axis=0)
    stdval_core = np.ma.std(autos_ampdb[inds_core, :], axis=0)
    stdval_exp = np.ma.std(autos_ampdb[inds_exp, :], axis=0)
    if flag_exp_with_core_stat:
        print('!! Use core antenna statistics to flag outer antennas !!')
        medval_exp = medval_core
        stdval_exp = stdval_core
    if debug:
        print('=====Before filtering out those beyond 1 sigma=====')
        print('Median of core antennas', medval_core[0], medval_core[3])
        print('Standard deviation of core antennas', stdval_core[0], stdval_core[3])
        print('Median of outer antennas', medval_exp[0], medval_exp[3])
        print('Standard deviation of outer antennas', stdval_exp[0], stdval_exp[3])
    # find 1 sigma outliers, exclude, and recalculate stdval
    newinds_core = np.asarray(inds_core)[
        np.where(((autos_ampdb[inds_core, 0] < medval_core[0] + 1 * stdval_core[0]) &
                  (autos_ampdb[inds_core, 0] > medval_core[0] - 1 * stdval_core[0])) |
                 ((autos_ampdb[inds_core, 3] < medval_core[3] + 1 * stdval_core[3]) &
                  (autos_ampdb[inds_core, 3] > medval_core[3] - 1 * stdval_core[3])))]
    newinds_exp = np.asarray(inds_exp)[
        np.where(((autos_ampdb[inds_exp, 0] < medval_exp[0] + 2 * stdval_exp[0]) &
                  (autos_ampdb[inds_exp, 0] > medval_exp[0] - 2 * stdval_exp[0])) |
                 ((autos_ampdb[inds_exp, 3] < medval_exp[3] + 2 * stdval_exp[3]) &
                  (autos_ampdb[inds_exp, 3] > medval_exp[3] - 2 * stdval_exp[3])))]
    # exclude and recalculate
    medval_core = np.ma.median(autos_ampdb[newinds_core, :], axis=0)
    medval_exp = np.ma.median(autos_ampdb[newinds_exp, :], axis=0)
    stdval_core = np.ma.std(autos_ampdb[newinds_core, :], axis=0)
    stdval_exp = np.ma.std(autos_ampdb[newinds_exp, :], axis=0)
    if debug:
        print('=====After filtering out those beyond 1 sigma=====')
        print('Median of core antennas', medval_core[0], medval_core[3])
        print('Standard deviation of core antennas', stdval_core[0], stdval_core[3])
        print('Median of outer antennas', medval_exp[0], medval_exp[3])
        print('Standard deviation of outer antennas', stdval_exp[0], stdval_exp[3])

    flagscore = np.asarray(inds_core)[
        np.where((autos_ampdb[inds_core, 0] > medval_core[0] + thresh_core * stdval_core[0]) |
                 (autos_ampdb[inds_core, 0] < medval_core[0] - thresh_core * stdval_core[0]) |
                 (autos_ampdb[inds_core, 3] > medval_core[3] + thresh_core * stdval_core[3]) |
                 (autos_ampdb[inds_core, 3] < medval_core[3] - thresh_core * stdval_core[3]))]
    flagsexp = np.asarray(inds_exp)[
        np.where((autos_ampdb[inds_exp, 0] > medval_exp[0] + thresh_exp * stdval_exp[0]) |
                 (autos_ampdb[inds_exp, 0] < medval_exp[0] - thresh_exp * stdval_exp[0]) |
                 (autos_ampdb[inds_exp, 3] > medval_exp[3] + thresh_exp * stdval_exp[3]) |
                 (autos_ampdb[inds_exp, 3] < medval_exp[3] - thresh_exp * stdval_exp[3]))]
    flagsall = np.sort(np.append(flagscore, flagsexp))
    print('{0:d} bad antennas found out of {1:d} antennas'.format(flagsall.size, Nants))
    if flagsall.size > 0:
        if antflagfile is None:
            antflagfile = os.path.splitext(os.path.abspath(msfile))[0] + '.badants'
        print('Writing flags to ' + antflagfile)
        if os.path.exists(antflagfile) and doappend:
            existingflags = np.genfromtxt(antflagfile, delimiter=',', dtype=int)
            flagsall = np.append(flagsall, existingflags)
            flagsall = np.unique(flagsall)
        flagsallstr = [str(flag) for flag in flagsall]
        flag_core_ids = ",".join([str(flag) for flag in np.sort(flagscore)])
        flag_core_names = msmd.antennanames(flagscore)
        flag_core_vals = autos_ampdb[flagscore]
        flag_exp_ids = ",".join([str(flag) for flag in np.sort(flagsexp)])
        flag_exp_names = msmd.antennanames(flagsexp)
        flag_exp_vals = autos_ampdb[flagsexp]
        flagsallstr2 = ",".join(flagsallstr)
        print('flagged core antenna ids: ', flag_core_ids)
        print('flagged core antenna names: ', flag_core_names)
        print('flagged outer antenna ids: ', flag_exp_ids)
        print('flagged outer antenna names: ', flag_exp_names)
        msmd.close()
        with open(antflagfile, 'w') as f:
            f.write(flagsallstr2)
        if doplot:
            fig = plt.figure(figsize=(12, 5))
            for i, n in enumerate([0, 3]):
                ax = fig.add_subplot(1, 2, i + 1)
                if n == 0:
                    ax.set_title('Auto-correlation in XX')
                    upper_bound = thresh_core + np.max(stdval_core)
                if n == 3:
                    ax.set_title('Auto-correlation in YY')
                    thresh = thresh_exp
                ax.plot(inds_core, autos_ampdb[inds_core, n], 'ro', fillstyle='none', label='Inner')
                ax.plot(flagscore, autos_ampdb[flagscore, n], 'ro', fillstyle='full', label='Flagged Inner')
                ax.plot(inds_exp, autos_ampdb[inds_exp, n], 'bo', fillstyle='none', label='Outer')
                ax.plot(flagsexp, autos_ampdb[flagsexp, n], 'bo', fillstyle='full', label='Flagged Outer')
                ax.plot([0, Nants], [medval_core[n], medval_core[n]], 'r-')
                ax.plot([0, Nants], [medval_core[n] + thresh_core + stdval_core[n],
                                     medval_core[n] + thresh_core + stdval_core[n]], 'r--')
                ax.plot([0, Nants], [medval_core[n] - thresh_core * stdval_core[n],
                                     medval_core[n] - thresh_core * stdval_core[n]], 'r--')
                ax.plot([0, Nants], [medval_exp[n], medval_exp[n]], 'b-')
                ax.plot([0, Nants], [medval_exp[n] + thresh_exp * stdval_exp[n],
                                     medval_exp[n] + thresh_exp * stdval_exp[n]], 'b--')
                ax.plot([0, Nants], [medval_exp[n] - thresh_exp * stdval_exp[n],
                                     medval_exp[n] - thresh_exp * stdval_exp[n]], 'b--')
                ax.set_xlabel('Antenna ID')
                ax.set_ylabel('dB (avg over channels)')
                ax.set_ylim([-30, 10])

                ax.legend()
            fig.tight_layout()
            plt.show()

        if debug:
            return antflagfile, medval_core, stdval_core, flag_core_ids, flag_core_names, flag_core_vals, \
                   medval_exp, stdval_exp, flag_exp_ids, flag_exp_names, flag_exp_vals
        else:
            return antflagfile
    else:
        if debug:
            return 0, medval_core, stdval_core, medval_exp, stdval_exp
        else:
            return 0


def gen_ant_flags_tst(msfile: str, debug: bool = False) -> str:
    """Generates a text file containing the bad antennas.
    DOES NOT ACTUALLY APPLY FLAGS.

    Adapted from the flag_bad_ants() module in
    https://github.com/ovro-lwa/distributed-pipeline/blob/main/orca/flagging/flag_bad_ants.py

    Comment BC (April 7, 2023): Does not seem to work well with lots of antennas out
    Args:
        msfile: msfile to generate
    Returns:
        Path to the text file with list of antennas to flag.
    """
    nant, nspw, nchan = get_msinfo(msfile)
    tb.open(msfile)
    tautos = tb.query('ANTENNA1=ANTENNA2')

    # iterate over antenna, 1-->256
    datacolxx = np.zeros((nchan * nspw, nant))
    datacolyy = np.copy(datacolxx)
    for i in range(nspw):
        datacolxx[i * nchan:(i + 1) * nchan] = tb.getcol("DATA", nant * i, nant)[0]
        datacolyy[i * nchan:(i + 1) * nchan] = tb.getcol("DATA", nant * i, nant)[3]

    datacolxxamp = np.sqrt(np.real(datacolxx) ** 2. + np.imag(datacolxx) ** 2.)
    datacolyyamp = np.sqrt(np.real(datacolyy) ** 2. + np.imag(datacolyy) ** 2.)

    datacolxxampdb = 10 * np.log10(datacolxxamp / 1.e2)
    datacolyyampdb = 10 * np.log10(datacolyyamp / 1.e2)

    # median value for every antenna
    medamp_perantx = np.median(datacolxxampdb, axis=1)
    medamp_peranty = np.median(datacolyyampdb, axis=1)

    # get flags based on deviation from median amp
    xthresh_pos = np.median(medamp_perantx) + np.std(medamp_perantx)
    xthresh_neg = np.median(medamp_perantx) - 2 * np.std(medamp_perantx)
    ythresh_pos = np.median(medamp_peranty) + np.std(medamp_peranty)
    ythresh_neg = np.median(medamp_peranty) - 2 * np.std(medamp_peranty)
    flags = np.where((medamp_perantx > xthresh_pos) | (medamp_perantx < xthresh_neg) | \
                     (medamp_peranty > ythresh_pos) | (medamp_peranty < ythresh_neg) | \
                     np.isnan(medamp_perantx) | np.isnan(medamp_peranty))

    # use unflagged antennas to generate median spectrum
    flagmask = np.zeros((nchan * nspw, nant))
    flagmask[:, flags[0]] = 1
    datacolxxampdb_mask = np.ma.masked_array(datacolxxampdb, mask=flagmask, fill_value=np.nan)
    datacolyyampdb_mask = np.ma.masked_array(datacolyyampdb, mask=flagmask, fill_value=np.nan)

    medamp_allantsx = np.median(datacolxxampdb_mask, axis=1)
    medamp_allantsy = np.median(datacolyyampdb_mask, axis=1)

    stdarrayx = np.array([np.std(antarr / medamp_allantsx) for antarr in datacolxxampdb_mask.transpose()])
    stdarrayy = np.array([np.std(antarr / medamp_allantsy) for antarr in datacolyyampdb_mask.transpose()])

    # this threshold was manually selected...should be changed to something better at some point
    if nant > 256:
        thresh = 1
    else:
        thresh = 0.02
    flags2 = np.where((stdarrayx > thresh) | (stdarrayy > thresh))

    flagsall = np.sort(np.append(flags, flags2))
    flagsallstr = [str(flag) for flag in flagsall]
    flagsallstr2 = ",".join(flagsallstr)

    antflagfile = os.path.dirname(os.path.abspath(msfile)) + '/flag_bad_ants.ants'
    with open(antflagfile, 'w') as f:
        f.write(flagsallstr2)

    tb.close()
    if debug:
        return medamp_perantx, medamp_peranty, stdarrayx, stdarrayy
    else:
        return antflagfile


def flag_bad_ants(msfile, antflagfile=None, datacolumn='DATA', thresh_core=1.0, thresh_exp=1.0, clearflags=True):
    """
    Read the text file that contains flags for bad antennas, and apply the flags
    :param msfile: input CASA ms visibility for calibration
    :param thresh: Threshold to use for flagging. Default is 10.
    """
    if clearflags:
        flaglist = flagmanager(msfile, mode='list')
        # check if previous flags exist. If so, restore to original state
        if len(flaglist) > 1:
            flagmanager(msfile, mode='restore', versionname=flaglist[0]['name'])
    if antflagfile is None:
        antflagfile = os.path.splitext(os.path.abspath(msfile))[0] + '.badants'
        res = gen_ant_flags_from_autocorr(msfile, antflagfile=antflagfile, datacolumn=datacolumn,
                                          thresh_core=thresh_core, thresh_exp=thresh_exp)
    if os.path.isfile(antflagfile):
        with open(antflagfile, 'r') as f:
            antenna_list = f.readline()
            print('Applying flags for these antennas')
            print(antenna_list)
        flagdata(vis=msfile, mode='manual', antenna=antenna_list)
    else:
        print("No flag is found. Do nothing")
    return antflagfile


def gen_calibration(msfile, modelcl=None, uvrange='', bcaltb=None):
    """
    This function is for doing initial self-calibrations using strong sources that are above the horizon
    It is recommended to use a dataset observed at night when the Sun is not in the field of view
    :param uvrange: uv range to consider for calibration. Following CASA's syntax, e.g., '>1lambda'
    :param msfile: input CASA ms visibility for calibration
    :param modelcl: input model of strong sources as a component list, produced from gen_model_cl()
    """
    if not modelcl or not (os.path.exists(modelcl)):
        print('Model component list does not exist. Generating one from scratch.')
        modelcl = gen_model_cl(msfile)

    # Put the component list to the model column
    clearcal(msfile, addmodel=True)
    ft(msfile, complist=modelcl, usescratch=True)
    # Now do a bandpass calibration using the model component list
    if not bcaltb:
        bcaltb = os.path.splitext(msfile)[0] + '.bcal'
    bandpass(msfile, caltable=bcaltb, uvrange=uvrange, combine='scan,field,obs', fillgaps=1)
    return bcaltb


def apply_calibration(msfile, gaintable=None, doflag=False, antflagfile=None, do_solar_imaging=True,
                      imagename='test'):
    if doflag:
        flag_bad_ants(msfile, antflagfile=antflagfile)
    if not gaintable:
        print('No calibration table is provided. Abort... ')
    else:
        if type(gaintable) == str:
            gaintable = [gaintable]
    # Apply the calibration
    clearcal(msfile)
    applycal(msfile, gaintable=gaintable, flagbackup=True, applymode='calflag')
    sunpos = get_sun_pos(msfile)
    if do_solar_imaging:
        tclean(msfile, imagename=imagename, imsize=[512], cell=['1arcmin'],
               weighting='uniform', phasecenter=sunpos, niter=500)
        print('Solar image made {0:s}.image'.format(imagename))


def make_fullsky_image(msfile, imagename="allsky", imsize=4096, cell='2arcmin',
                       minuv=10):  ### minuv: minimum uv in lambda
    """
    Make all sky image with wsclean
    :param msfile: path to CASA measurement set
    :param imagename: output image name
    :param imsize: size of the image in pixels
    :param cell: pixel scale
    :param minuv: minimum uv to consider for imaging (in # of wavelengths)
    :return: produces wsclean images (fits), PSF, etc.
    """
    os.system("wsclean -no-update-model-required -weight uniform" +
              " -name " + imagename + " -size " + str(imsize) + " " + str(imsize) + " -scale " + cell +
              " -minuv-l " + str(minuv) + " -niter 1000 " + msfile)


def get_solar_loc_pix(msfile, image="allsky"):
    """
    Get the x, y pixel location of the Sun from an all-sky image
    :param msfile: path to CASA measurement set
    :param image: all sky image made from the measurement set
    :return: pixel value in X and Y for solar disk center
    """
    from astropy.wcs.utils import skycoord_to_pixel
    m = get_sun_pos(msfile, str_output=False)
    ra = m['m0']['value']
    dec = m['m1']['value']
    coord = SkyCoord(ra * u.rad, dec * u.rad, frame='icrs')
    head = fits.getheader(image + "-model.fits")
    w = WCS(head)
    pix = skycoord_to_pixel(coord, w)
    x = int(pix[0])
    y = int(pix[1])
    return x, y


def get_nonsolar_sources_loc_pix(msfile, image="allsky", verbose=False):
    """
    Converting the RA & DEC coordinates of nonsolar sources to image coordinates in X and Y
    :param image: input CASA image
    :return: an updated directionary of strong sources with 'xpix' and 'ypix' added
    """
    from astropy.wcs.utils import skycoord_to_pixel
    srcs = [{'label': 'CasA', 'position': 'J2000 23h23m24s +58d48m54s'},
            {'label': 'CygA', 'position': 'J2000 19h59m28.35663s +40d44m02.0970s'},
            {'label': 'TauA', 'position': 'J2000 05h34m31.94s +22d00m52.2s'},
            {'label': 'VirA', 'position': 'J2000 12h30m49.42338s +12d23m28.0439s'}]

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
            coord[2] = conv_deg(coord[2])
            d0 = me.direction(coord[0], coord[1], coord[2])
            d0_j2000 = me.measure(d0, 'J2000')
        else:
            raise Exception("Unknown direction")
        d = me.measure(d0, 'AZEL')
        elev = d['m1']['value']
        if elev > 0:
            ra = d0_j2000['m0']['value']
            dec = d0_j2000['m1']['value']
            coord = SkyCoord(ra * u.rad, dec * u.rad, frame='icrs')
            head = fits.getheader(image + "-model.fits")
            w = WCS(head)
            pix = skycoord_to_pixel(coord, w)
            x = int(pix[0])
            y = int(pix[1])
            srcs[i]['xpix'] = x
            srcs[i]['ypix'] = y
            if verbose:
                print('Found source {0:s} at pix x {1:d}, y {2:d}'.format(srcs[i]['label'], x, y))
        else:
            if verbose:
                print('Source {0:s} has a <0 elevation'.format(srcs[i]['label']))
            del srcs[i]
    return srcs


def gen_nonsolar_source_model(msfile, imagename="allsky", outimage=None, sol_area=200., src_area=20.,
                              remove_strong_sources_only=True, verbose=True):
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
    :return: FITS image with non-solar sources removed
    """

    solx, soly = get_solar_loc_pix(msfile, imagename)
    srcs = get_nonsolar_sources_loc_pix(msfile, imagename)
    head = fits.getheader(imagename + "-model.fits")
    if head['cunit1'] == 'deg':
        dx = np.abs(head['cdelt1'] * 60.)
    else:
        print(head['cunit1'] + ' not recognized as "deg". Model could be wrong.')
    if head['cunit2'] == 'deg':
        dy = np.abs(head['cdelt2'] * 60.)
    else:
        print(head['cunit2'] + ' not recognized as "deg". Model could be wrong.')
    data = fits.getdata(imagename + "-model.fits")
    if remove_strong_sources_only:
        new_data = np.zeros_like(data)
        src_area_xpix = src_area / dx
        src_area_ypix = src_area / dy
        for s in srcs:
            src_x = s['xpix']
            src_y = s['ypix']
            bbox = [[src_y - src_area_ypix // 2, src_y + src_area_ypix // 2],
                    [src_x - src_area_xpix // 2, src_x + src_area_xpix // 2]]
            slicey, slicex = slice(int(bbox[0][0]), int(bbox[0][1])+1), slice(int(bbox[1][0]), int(bbox[1][1])+1)
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
    fits.writeto(outimage + '-model.fits', new_data, header=head, overwrite=True)
    return outimage


def predict_model(msfile, outms, image="_no_sun"):
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
    os.system("wsclean -predict -name " + image + " " + outms)


def remove_nonsolar_sources(msfile, imagename='allsky', imsize=4096, cell='2arcmin', minuv=10):
    """
    Wrapping for removing the nonsolar sources from the solar measurement set
    :param msfile: input CASA measurement set
    :param imagename: name of the all sky image
    :param imsize: size of the image in pixels
    :param cell: pixel scale
    :param minuv: minimum uv to consider for imaging (in # of wavelengths)
    :return: a CASA measurement set with non-solar sources removed. Default name is "*_sun_only.ms"
    """
    make_fullsky_image(msfile=msfile, imagename=imagename, imsize=imsize, cell=cell, minuv=minuv)
    image_nosun = gen_nonsolar_source_model(msfile, imagename=imagename)
    outms = msfile[:-3] + "_sun_only.ms"
    predict_model(msfile, outms=outms, image=image_nosun)
    # uvsub(outms)
    tb.open(msfile)
    corrected_data = tb.getcol("CORRECTED_DATA")
    tb.close()
    tb.open(outms, nomodify=False)
    model = tb.getcol("MODEL_DATA")
    tb.putcol("CORRECTED_DATA", corrected_data - model)
    tb.flush()
    tb.close()
    return outms


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
    sunpos = get_sun_pos(msfile)
    tclean(msfile, imagename=imagename, uvrange=uvrange, imsize=imsize, cell=cell,
           weighting='uniform', phasecenter=sunpos, niter=niter, psfcutoff=psfcutoff)


def correct_ms_bug(msfile):
    """
    Temporary fix for the visibility files produced by the current pipeline.
    Update: initially found in August 2022. Not needed as of April 2023.
    :param msfile: input CASA measurement set
    """
    tb.open(msfile + "/SPECTRAL_WINDOW", nomodify=False)
    meas_freq_ref = tb.getcol('MEAS_FREQ_REF')
    if meas_freq_ref[0] == 0:
        meas_freq_ref[0] = 1
    tb.putcol('MEAS_FREQ_REF', meas_freq_ref)
    tb.flush()
    tb.close()


def pipeline(solar_ms, calib_ms=None, bcal=None, imagename='sun_only', imsize=512, cell='1arcmin'):
    """
    Pipeline to calibrate and imaging a solar visibility
    :param solar_ms: input solar measurement set
    :param calib_ms: (optional) input measurement set for generating the calibrations, usually is one observed at night
    :param bcal: (optional) bandpass calibration table. If not provided, use calib_ms to generate one.
    """
    if not bcal:
        if os.path.exists(calib_ms):
            flag_bad_ants(calib_ms)
            bcal = gen_calibration(calib_ms)
        else:
            print('Neither calib_ms nor bcal exists. Need to provide calibrations to continue. Abort..')
    #correct_ms_bug(solar_ms)
    apply_calibration(solar_ms, gaintable=bcal, doflag=True, do_solar_imaging=False)
    outms = remove_nonsolar_sources(solar_ms)
    make_solar_image(outms, imagename=imagename, imsize=imsize, cell=cell)
