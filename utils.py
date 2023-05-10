import os, sys
from astropy.time import Time
from astropy.io import fits
from casatools import image, table, msmetadata
import numpy as np
import logging, glob


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
        max1[0], min1[0] = get_image_maxmin(imagename)
        print(max1, min1)
    else:
        if reorder and max1[1] > 0.001:
            max1[0], min1[0] = max1[1], min1[1]
        max1[1], min1[1] = get_image_maxmin(imagename)

        DR1 = max1[0] / abs(min1[0])
        DR2 = max1[1] / abs(min1[1])
        print(DR1, DR2)
        if (DR1 - DR2) / DR2 > 0.2:
            ### if max decreases by more than 20 percent
            ## absolute value of minimum increases by more than 20 percent
            if min1[1] < 0:
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
    pieces = msname.split('_')
    ymd = pieces[0]
    hms = pieces[1]
    mstime = Time(ymd[0:4] + "-" + ymd[4:6] + "-" + ymd[6:] +
                  'T' + hms[0:2] + ":" + hms[2:4] + ":" + hms[4:],
                  scale='utc', format='isot')
    return mstime


def get_timestr_from_name(msname):
    pieces = msname.split('_')
    return '_'.join(pieces[0:2])


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


def convert_to_heliocentric_coords(msname, imagename, helio_imagename=None, reftime=''):
    import datetime as dt
    from suncasa.utils import helioimage2fits as hf
    from casatasks import importfits

    if reftime == '':
        msmd = msmetadata()
        msmd.open(msname)
        times = msmd.timesforfield(0)
        msmd.done()
        time = Time(times[0] / 86400, scale='utc', format='mjd')
        time.format = 'datetime'

        tdt = dt.timedelta(seconds=3)

        t1 = time - tdt
        t2 = time + tdt
        print(t1.strftime("%Y/%m/%d/%H:%M:%S"))
        reftime = t1.strftime('%Y/%m/%d/%H:%M:%S') + "~" + t2.strftime('%Y/%m/%d/%H:%M:%S')
    print(reftime)
    temp_image = imagename + ".tmp"
    if helio_imagename is None:
        helio_imagename = imagename.replace('.fits', '.helio.fits')
    if not os.path.isdir(imagename):
        importfits(fitsimage=imagename, imagename=temp_image, overwrite=True)
    else:
        temp_image = imagename

    try:
        hf.imreg(vis=msname, imagefile=temp_image, timerange=reftime,
                 fitsfile=helio_imagename, usephacenter=True, verbose=True, toTb=True)
    except:
        logging.warning("Could not convert to helicentric coordinates")
        return helio_imagename
    return None


def convert_to_heliocentric_coords1(msname, imagename):
    import datetime as dt
    from suncasa.utils import helioimage2fits as hf
    from casatasks import importfits

    msmd = msmetadata()
    msmd.open(msname)
    times = msmd.timesforfield(0)
    msmd.done()
    time = Time(times[0] / 86400, scale='utc', format='mjd')
    time.format = 'datetime'

    tdt = dt.timedelta(seconds=60)

    t1 = time - tdt
    t2 = time + tdt

    time_str = t1.strftime('%Y/%m/%d/%H:%M:%S') + "~" + t2.strftime('%Y/%m/%d/%H:%M:%S')

    temp_image = 'temp_' + imagename + ".image"
    helio_image = imagename + ".helio"
    if os.path.isdir(imagename) == False:
        importfits(fitsimage=imagename, imagename=temp_image, overwrite=True)
    else:
        temp_image = imagename
    try:
        hf.imreg(vis=msname, imagefile=temp_image, timerange=time_str,
                 fitsfile=helio_image, usephacenter=True, verbose=True, toTb=True)
    except:
        logging.warning("Could not convert to helicentric coordinates")
        return helio_image
    return None
