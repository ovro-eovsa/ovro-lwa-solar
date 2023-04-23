import os,sys
from astropy.time import Time


def list_msfiles(filepath='20230318/'):
    """
    Find measurement sets across all lwacalim nodes under a file path.
    Return a list of dictionary containing their path, name, time, and frequency information
    :param filepath: path relative to /data0x/
    :return msfiles: a list of dictionary containing all ms files with path, name, time, and frequency
    """
    msfiles = []
    for i in range(1, 8):
        out = os.popen('ssh lwacalim0{0:d} ls /data0{1:d}/{2:s}/'.format(i, i, filepath)).read()
        names = out.split('\n')[:-1]
        for n in names:
            if n[-6:] == 'MHz.ms':
                pathstr = 'lwacalim0{0:d}:/data0{1:d}/20230318/{2:s}'.format(i, i, n)
                tmpstr = n[:15].replace('_', 'T')
                timestr = tmpstr[:4] + '-' + tmpstr[4:6] + '-' + tmpstr[6:11] + ':' + tmpstr[11:13] + ':' + tmpstr[13:]
                freqstr = n[16:21]
                msfiles.append({'path': pathstr, 'name': n, 'time': timestr, 'freq': freqstr})
    return msfiles
