#!/usr/bin/env python

# This script is adapted from Marin Anderson's script named /opt/astro/utils/bin/gen_model_ms.py on astm.lwa.ovro.caltech.edu
# History: 2022-07-27 B. Chen, changed the use of casacore to the modular CASA 6 (test on 6.5, 
#                              but earlier versions should be fine);
# 		  			           Also restructured it as part of the solar OVRO-LWA calibration package for future updates

from casatools import table, measures, componentlist
import math
import sys

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
