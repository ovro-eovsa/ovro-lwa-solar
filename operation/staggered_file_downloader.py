import os, sys, glob, getopt
from astropy.time import Time, TimeDelta
import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation, get_body, AltAz
from time import sleep
import argparse
import timeit
import shlex, subprocess

def sun_riseset(date=Time.now(), observatory='ovro'):
    '''
    Given a date in Time object, determine the sun rise and set time as viewed from OVRO
    '''
    try:
        date_mjd = Time(date).mjd
    except Exception as e:
        logging.error(e)

    obs = EarthLocation.of_site(observatory)
    t0 = Time(int(date_mjd) + 13. / 24., format='mjd')
    sun_loc = get_body('sun', t0, location=obs)
    alt = sun_loc.transform_to(AltAz(obstime=t0, location=obs)).alt.degree
    while alt < 10.:
        t0 += TimeDelta(60., format='sec')
        alt = sun_loc.transform_to(AltAz(obstime=t0, location=obs)).alt.degree

    t1 = Time(int(date_mjd) + 22. / 24., format='mjd')
    sun_loc = get_body('sun', t1, location=obs)
    alt = sun_loc.transform_to(AltAz(obstime=t1, location=obs)).alt.degree
    while alt > 10.:
        t1 += TimeDelta(60., format='sec')
        alt = sun_loc.transform_to(AltAz(obstime=t1, location=obs)).alt.degree

    return t0, t1

def download_msfiles(msfiles, destination='/fast/bin.chen/20231014_eclipse/slow_working/', bands=None, verbose=True, server=None, maxthread=5):
    from multiprocessing.pool import ThreadPool
    """
    Parallelized downloading for msfiles returned from list_msfiles() to a destination.
    """
    inmsfiles_path = [f['path'] for f in msfiles]
    inmsfiles_name = [f['name'] for f in msfiles]
    inmsfiles_band = [f['freq'] for f in msfiles]
    omsfiles_path = []
    omsfiles_name = []
    if bands is None:
        omsfiles_path = inmsfiles_path
        omsfiles_name = inmsfiles_name
    else:
        for bd in bands:
            if bd in inmsfiles_band:
                idx = inmsfiles_band.index(bd)
                #omsfiles_server.append(inmsfiles_server[idx])
                omsfiles_path.append(inmsfiles_path[idx])
                omsfiles_name.append(inmsfiles_name[idx])

    nfile = len(omsfiles_path)
    if nfile == 0:
        print('No files to download. Abort...')
        return -1
    time_bg = timeit.default_timer() 
    if verbose:
        print('I am going to download {0:d} files'.format(nfile))

    tp = ThreadPool(maxthread)
    for omsfile_path in omsfiles_path:
        tp.apply_async(download_msfiles_cmd, args=(omsfile_path, server, destination))

    tp.close()
    tp.join()

    time_completed = timeit.default_timer() 
    if verbose:
        print('Downloading {0:d} files took in {1:.1f} s'.format(nfile, time_completed-time_bg))
    omsfiles = [destination + n for n in omsfiles_name]
    return omsfiles
    
        
def list_msfiles(intime, lustre=True, file_path='slow', server=None, time_interval='10s', 
                # bands=['32MHz', '36MHz', '41MHz', '46MHz', '50MHz', '55MHz', '59MHz', '64MHz', '69MHz', '73MHz', '78MHz', '82MHz']):
                bands=['82MHz']):
    """
    Return a list of visibilities to be copied for pipeline processing for a given time
    :param intime: astropy Time object
    :param server: name of the server to list available data
    :param lustre: if True, specific to lustre system on lwacalim nodes. If not, try your luck in combination with file_path
    :param file_path: file path to the data files. For lustre, it is either 'slow' or 'fast'. For other servers, provide full path to data.
    :param time_interval: Options are '10s', '1min', '10min'
    :param bands: bands to list/download. Default to 12 bands above 30 MHz. Full list of available bands is
            ['13MHz', '18MHz', '23MHz', '27MHz', '32MHz', '36MHz', '41MHz', '46MHz', '50MHz', '55MHz', '59MHz', '64MHz', '69MHz', '73MHz', '78MHz', '82MHz']
    """
    intimestr = intime.isot[:-4].replace('-','').replace(':','').replace('T','_')
    datestr = intime.isot[:10]
    hourstr = intime.isot[11:13]
    if time_interval == '10s':
        tstr = intimestr[:-1]
    if time_interval == '1min':
        tstr = intimestr[:-2]
    if time_interval == '10min':
        tstr = intimestr[:-3]

    msfiles = []
    print (file_path)
    if lustre:
        processes=[]
        for b in bands:
            pathstr = '/lustre/pipeline/{0:s}/{1:s}/{2:s}/{3:s}/'.format(file_path, b, datestr, hourstr)
            if server:
                cmd = 'ssh ' + server + ' ls ' + pathstr + ' | grep ' + tstr
            else:
                cmd = 'ls ' + pathstr + ' | grep ' + tstr
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
            processes.append(p)
            filenames = p.communicate()[0].decode('utf-8').split('\n')[:-1]
            for filename in filenames:
                if filename[-6:] == 'MHz.ms':
                    filestr = pathstr + filename
                    tmpstr = filename[:15].replace('_', 'T')
                    timestr = tmpstr[:4] + '-' + tmpstr[4:6] + '-' + tmpstr[6:11] + ':' + tmpstr[11:13] + ':' + tmpstr[13:]
                    freqstr = filename[16:21]
                    msfiles.append({'path': filestr, 'name': filename, 'time': timestr, 'freq': freqstr})
    else:
        if server:
            cmd = 'ssh ' + server + ' ls ' + file_path + ' | grep ' + tstr
        else:
            cmd = 'ls ' + file_path + ' | grep ' + tstr
        p = subprocess.run(cmd, capture_output=True, shell=True)
        filenames = p.stdout.decode('utf-8').split('\n')[:-1]
        for filename in filenames:
            if filename[-6:] == 'MHz.ms':
                pathstr = '{0:s}:{1:s}/{2:s}'.format(server, file_path, filename)
                tmpstr = filename[:15].replace('_', 'T')
                timestr = tmpstr[:4] + '-' + tmpstr[4:6] + '-' + tmpstr[6:11] + ':' + tmpstr[11:13] + ':' + tmpstr[13:]
                freqstr = filename[16:21]
                msfiles.append({'path': pathstr, 'name': filename, 'time': timestr, 'freq': freqstr})
    print (msfiles)
    return msfiles


def download_msfiles_cmd(msfile_path, server, destination):
    if server:
        p = subprocess.Popen(shlex.split('rsync -az --numeric-ids --info=progress2 --no-perms --no-owner --no-group {0:s}:{1:s} {2:s}'.format(server, msfile_path, destination)))
    else:
        p = subprocess.Popen(shlex.split('rsync -az --numeric-ids --info=progress2 --no-perms --no-owner --no-group {0:s} {1:s}'.format(msfile_path, destination)))
    std_out, std_err = p.communicate()
    if std_err:
        print(std_err)

def download_file(image_time,destination,min_nband=1,file_path='fast'):
    bands = ['82MHz']#['32MHz', '36MHz', '41MHz', '46MHz', '50MHz', '55MHz', '59MHz', '64MHz', '69MHz', '73MHz', '78MHz', '82MHz']
    
    msfiles0 = list_msfiles(image_time, file_path=file_path)
    if len(msfiles0) < min_nband:
        print('This time only has {0:d} subbands. Check nearby +-10s time.'.format(len(msfiles0)))
        image_time_before = image_time - TimeDelta(10., format='sec')
        msfiles0_before = list_msfiles(image_time_before, file_path=file_path)
        image_time_after = image_time + TimeDelta(10., format='sec')
        msfiles0_after = list_msfiles(image_time_after, file_path=file_path)
        if len(msfiles0_before) < min_nband and len(msfiles0_after) < min_nband:
            print('I cannot find a nearby time with at least {0:d} available subbands. Abort and wait for next time interval.'.format(min_nband))
            return False
        else:
            if len(msfiles0_before) > len(msfiles0_after):
                msfiles0 = msfiles0_before
                image_time = image_time_before
            else:
                msfiles0 = msfiles0_after
                image_time = image_time_after    
    msfiles = download_msfiles(msfiles0, destination=destination, bands=bands)
    return
    
def copy_files(time_start=Time.now(), time_interval=10., delay_from_now=180.,\
    server='lwacalim', file_path='fast', max_files=5,\
    proc_dir = '/fast/msurajit/realtime_pipeline/'):
    
    if type(time_start) is str:
        time_start=Time(time_start)
    visdir_work = proc_dir + 'fast_working/'
    if not os.path.exists(visdir_work):
        os.makedirs(visdir_work)
        
    (t_rise, t_set) = sun_riseset(time_start)
    
    if time_start < t_rise:
        twait = t_rise - time_start
        time_start += TimeDelta(twait.sec + 60., format='sec')
        sleep(twait.sec + 60.)
    
    while True:
        time1 = timeit.default_timer()
        if time_start > Time.now() - TimeDelta(delay_from_now, format='sec'):
            twait = time_start - Time.now()
            logging.info('{0:s}: Start time {1:s} is too close to current time. Wait {2:.1f} m to start.'.format(socket.gethostname(), time_start.isot, (twait.sec + delay_from_now) / 60.))
            sleep(twait.sec + delay_from_now)
            

        while True:
            num_files=len(glob.glob(os.path.join(visdir_work,"*.ms")))
            if num_files<max_files:
                download_file(time_start,visdir_work)
                time_start += TimeDelta(time_interval, format='sec')
                if time_start>t_set:
                    break
            else:
                sleep(5)
            
            
        if time_start > t_set:
            
            twait = t_rise_next - time_start
            logging.info('{0:s}: Sun is setting. Done for the day. Wait for {1:.1f} hours to start.'.format(socket.gethostname(), twait.value * 24.)) 
            time_start += TimeDelta(twait.sec + 60. + delay_by_node, format='sec')
            t_rise = t_rise_next
            t_set = t_set_next
            sleep(twait.sec + 60. + delay_by_node)    
        
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Solar realtime pipeline')
    parser.add_argument('prefix', type=str, help='Timestamp for the start time. Format YYYY-MM-DDTHH:MM')
    args = parser.parse_args()
    copy_files(args.prefix)        
    
