import os,sys
from astropy.time import Time
from astropy.io import fits
from casatools import image
import numpy as np
from astropy.time import Time
import glob

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

def get_image_data(imagename):
     if os.path.isfile(imagename):
        data=np.squeeze(fits.getdata(imagename))       
     elif os.path.isdir(imagename):
        ia=image()
        ia.open(imagename)
        data=ia.getchunk()
        ia.close()
        data=np.squeeze(data)
     else:
        raise RuntimeError("Image does not exist")
     return data
    
def get_image_maxmin(imagename,local=True):
    data=get_image_data(imagename)
    maxval=np.nanmax(data)
    if local==True:
        maxpos=np.where(abs(data-maxval)<1e-5)
        max1=data[maxpos][0]
        min1=np.nanmin(data[maxpos[0][0]-100:maxpos[0][0]+100,\
                maxpos[1][0]-100:maxpos[1][0]+100])
    
        return max1,min1
    else:
        minval=np.nanmin(data)
    return maxval,minval
        
def check_image_quality(imagename,max1,min1,reorder=True):
        if max1[0]==0:
            max1[0],min1[0]=get_image_maxmin(imagename)
            print (max1,min1)
        else:
            if reorder==True and max1[1]>0.001:
                max1[0],min1[0]=max1[1],min1[1]
            max1[1],min1[1]=get_image_maxmin(imagename)
            
            DR1=max1[0]/abs(min1[0])
            DR2=max1[1]/abs(min1[1])
            print (DR1,DR2)
            if (DR1-DR2)/DR2>0.2:
                ### if max decreases by more than 20 percent 
                    ## absolute value of minimum increases by more than 20 percent
                if min1[1]<0:
                    return False
        return True
         
     
def restore_flag(msfile):
    from casatasks import flagmanager
    flag_tables=flagmanager(msfile)
    keys=flag_tables.keys()
    last_flagtable=flag_tables[len(keys)-2]['name']  #### last key is MS. 
    flagmanager(vis=msfile,mode='restore',versionname=last_flagtable)
    flagmanager(vis=msfile,mode='delete',versionname=last_flagtable)           
    return

def get_time_from_name(msname):
    pieces=msname.split('_')
    ymd=pieces[0]
    hms=pieces[1]
    mstime=Time(ymd[0:2]+"-"+ymd[2:4]+"-"+ymd[4:]+\
                'T'+hms[0:2]+":"+hms[2:4]+":"+hms[4:],\
                scale='utc',format='isot')
    return mstime
    
def get_selfcal_time_to_apply(msname):
    mstime=get_time_from_name(msname)
    caltables=glob.glob("caltables/*.gcal")
    times=np.unique(np.array(['_'.join(i.split('_')[0:2]) for i in caltables]))
    
    sep=np.zeros(times)
    for n,t1 in enumerate(times):
        caltime=get_time_from_name(t1)
        sep[n]=abs((caltime-mstime).seconds*86400)
        
    time_to_apply=times(np.argsort(sep)[0])
    return time_to_apply
        
                 
