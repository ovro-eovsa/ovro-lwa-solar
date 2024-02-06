import sys, os
import shlex
import subprocess
from astropy.time import Time
def get_fitscopy_pid(mypid):
    # check whether a fits copy process is already running. Return PID if it is running, -1 if it is not.
    pidlist = subprocess.check_output(["pidof","python"]).split() # list of PIDs for all python processes
    for pid in pidlist:
        if mypid != pid:
            ps_out = str(subprocess.check_output(["ps", "-lfp" , pid]))
            ind = ps_out.find('daily_lwa_file_transfer.py') 
            if ind != -1:  # if this PID is running auto_beamcopy.py
                return str(pid)
    return -1


def copy_daily_fits(timestr, server='calim7', remote_fits_dir='/lustre/bin.chen/realtime_pipeline/fits/',
                local_fits_dir='/nas6/ovro-lwa-data/fits/'):
    """
    Purpose: rsync files from default directory of calim07:/lustre/bin.chen/realtime_pipeline/fits/*
        to pipeline:/nas6/ovro-lwa-data/fits/* and organize them according to date.
    timestr: input time '2024-01-02T18:00:00'
    """
    from time import sleep
    mjd = Time(timestr).mjd
    for day in [mjd-1, mjd]:
        daystr = Time(day,format='mjd').iso
        yyyy = daystr[:4]
        mm = daystr[5:7]
        dd = daystr[8:10]
        remotefolder = remote_fits_dir + '/' + yyyy + '/' + mm + '/' + dd
        command = 'ssh {0:s} ls {1:s} | grep fits'.format(server, remotefolder)
        p = subprocess.run(shlex.split(command), capture_output=True)
        filenames = p.stdout.decode('utf-8').split('\n')[:-1]
        nfiles = len(filenames)
        print('{0:s}: Found {1:d} fits files under {2:s}:{3:s}'.format(Time.now().iso[:19], nfiles, server, remotefolder))
        if nfiles > 0:
            localfolder = local_fits_dir + '/' + yyyy + '/' + mm + '/' + dd
            if not os.path.exists(localfolder):
                os.makedirs(localfolder)
                sleep(0.1)
            command = 'rsync -a --stats {0:s}:{1:s}/ {2:s}'.format(server, remotefolder, localfolder)
            print('{0:s}: Attemping to run {1:s}'.format(Time.now().iso[:19], command))
            res = subprocess.run(shlex.split(command), capture_output=True)
            output = res.stdout.decode('utf-8').split('\n')
            for outline in output:
                print(outline)

if __name__ == "__main__":
    mypid = os.getpid()
    pid = get_fitscopy_pid(bytes(str(mypid),encoding='UTF-8'))
    if pid != -1:
        print('Another copy of daily_lwa_file_transfer.py is already running. Type kill -KILL',pid,'to kill it.')
        exit()
    arglist = str(sys.argv)
    timestr = Time.now().iso
    if len(sys.argv) == 2:
        try:
            timestr = Time(sys.argv[1]).iso
        except:
            print('Cannot interpret', sys.argv[1], 'as a valid date/time string.')
            exit()
    # ndays is the number of days to go back in time for checking the existence of data
    copy_daily_fits(timestr)
    exit()
