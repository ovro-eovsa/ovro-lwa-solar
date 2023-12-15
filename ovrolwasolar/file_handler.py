import os
from astropy.time import Time
import datetime as dt
from casatasks import concat
import logging

def list_msfiles(file_path, distributed=True, nodes=[1, 2, 3, 4, 5, 6, 7, 8], server='lwacalim', verbose=False):
    """
    Find measurement sets across all lwacalim nodes under a file path.
    Return a list of dictionary containing their path, name, time, and frequency information

    :param filepath: path relative to /data0x/
    :param distributed: if True, then assume data are distributed in server and nodes ; 
                        if False, treat both server and file_path as absolute values
    :param nodes: list of integers; default to from 1 to 8 for lwacalim nodes
    :param server: name of the server; default to lwacalim
    :param verbose: whether or not to print extra information
    :return msfiles: a list of dictionary containing all ms files with path, name, time, and frequency
    """
    if verbose:
        print('Retrieving files from {0:s}:{1:s}'.format(server, file_path))
    msfiles = []
    if not distributed:
        out = os.popen('ssh {0:s} ls {1:s}/'.format(server, file_path)).read()
        names = out.split('\n')[:-1]
        for n in names:
            if n[-6:] == 'MHz.ms':
                pathstr = '{0:s}:{1:s}/{2:s}'.format(server, file_path, n)
                tmpstr = n[:15].replace('_', 'T')
                timestr = tmpstr[:4] + '-' + tmpstr[4:6] + '-' + tmpstr[6:11] + ':' + tmpstr[11:13] + ':' + tmpstr[13:]
                freqstr = n[16:21]
                msfiles.append({'path': pathstr, 'name': n, 'time': timestr, 'freq': freqstr})
    else:
        for i in nodes:
            out = os.popen('ssh {0:s}0{1:d} ls /data0{2:d}/{3:s}/'.format(server, i, i, file_path)).read()
            names = out.split('\n')[:-1]
            for n in names:
                if n[-6:] == 'MHz.ms':
                    pathstr = '{0:s}0{1:d}:/data0{2:d}/{3:s}/{4:s}'.format(server, i, i, file_path, n)
                    tmpstr = n[:15].replace('_', 'T')
                    timestr = tmpstr[:4] + '-' + tmpstr[4:6] + '-' + tmpstr[6:11] + ':' + tmpstr[11:13] + ':' + tmpstr[13:]
                    freqstr = n[16:21]
                    msfiles.append({'path': pathstr, 'name': n, 'time': timestr, 'freq': freqstr})

    return msfiles     
    
def list_msfiles1(file_path):
    """
    Find measurement sets across all lwacalim nodes under a file path.
    Return a list of dictionary containing their path, name, time, and frequency information

    :param filepath: path relative to /data0x/
    :return msfiles: a list of dictionary containing all ms files with path, name, time, and frequency
    """
    import datetime as dt
    msfiles = []
    
    starttime=dt.datetime(2023,5,2,17,50,0)
    endtime=dt.datetime(2023,5,2,18,50,0)
    tdt=dt.timedelta(seconds=8)
    
    st=starttime
    freqstr='78MHz'
    while st<=endtime:
        pathstr='lwacalim07:/data07/20230502_solar/name'
        timestr=st.strftime("%Y%m%d_%H%M%S")
        name=timestr+"_"+freqstr+".ms"
        tmpstr = timestr.replace('_', 'T')
        timestr = tmpstr[:4] + '-' + tmpstr[4:6] + '-' + tmpstr[6:11] + ':' + tmpstr[11:13] + ':' + tmpstr[13:]
        msfiles.append({'path': pathstr, 'name': name, 'time': timestr, 'freq': freqstr})
        st+=tdt
        
    st=starttime
    freqstr='31MHz'
    while st<=endtime:
        pathstr='lwacalim05:/data05/20230502_solar/name'
        timestr=st.strftime("%Y%m%d_%H%M%S")
        name=timestr+"_"+freqstr+".ms"
        tmpstr = timestr.replace('_', 'T')
        timestr = tmpstr[:4] + '-' + tmpstr[4:6] + '-' + tmpstr[6:11] + ':' + tmpstr[11:13] + ':' + tmpstr[13:]
        msfiles.append({'path': pathstr, 'name': name, 'time': timestr, 'freq': freqstr})
        st+=tdt
    return msfiles


def get_filename_from_time(timestr,freq):    
    temp=timestr.split('T')
    ymd=temp[0][0:4]+temp[0][5:7]+temp[0][8:]
    hms=temp[1][0:2]+temp[1][3:5]+temp[1][6:]
    filename=ymd+"_"+hms+"_"+freq+".ms"
    return filename
        
        
def file_downloader(file_list,path,freq):
    num_files=len(file_list)
    
    vis=[]
   
    for i in range(num_files):
        filename=get_filename_from_time(file_list[i],freq)
        if os.path.isdir(filename)==False:
            filepath=path+"/"+filename
            cmd_str="scp -r {0:s} ./".format(filepath)
           # print (cmd_str)
            os.system(cmd_str)
            logging.debug('Downloading {0:s}'.format(filename))
        vis.append(filename)
    
    if len(vis)>1:
        concat(vis=vis,concatvis='temp_'+filename+'.ms')
        logging.debug('Concatted all the files together to '+\
                    'achieve the required time integration')
        temp_filename=get_filename_from_time(file_list[num_files-1],freq)
        integrated_file=get_filename_from_time(file_list[0],freq)
        
        os.system("mv  "+'temp_'+temp_filename+'.ms '+ integrated_file)
        logging.debug("Created concatted file {0:s}".format(integrated_file))
        return integrated_file
    else:
        return vis[0]
          
class File_Handler:
    def __init__(self,time_duration,freqstr,file_path,start=None,end=None, \
                    time_integration=10,time_cadence=100,observation_integration=10):  ### time in seconds
        self.duration=time_duration
        self.file_path=file_path
        if isinstance(freqstr,list)==True:
            self.freqstr=freqstr
        elif isinstance(freqstr,str):
            self.freqstr=[freqstr]                                                     
        else:
            raise RuntimeError("Format unknown")
        self.num_freqs=len(self.freqstr)
        self.start=start
        self.end=end
        self.observation_integration=observation_integration
        self.time_integration=time_integration  ###seconds
        self.time_cadence=time_cadence
        self.selfcal_times=[]
        self.current_selfcal_time=''
        self.full_DI_selfcal_solint=4*3600 ### 4 hours
        self.partial_DI_selfcal_solint=1*3600 ### 1 hours
        self.full_DD_selfcal_solint=30*60 ### 30 minutes
        self.partial_DD_selfcal_solint=10*60 ## 10 minutes
        self.image_times=[]
        self.file_loop_started=False
        self.current_file_index=0
        self.current_path=None
        
    @property
    def start(self):
        return self._start
        
    @start.setter
    def start(self,value):
        self._start=value
    
    @property
    def end(self):
        return self._end
        
    @end.setter
    def end(self,value):
        self._end=value 
    
    def parse_duration(self,get_end=False):
        if get_end==False:
            times=self.duration.split('~')[0].replace('/','-')
            
        else:
            times=self.duration.split('~')[1].replace('/','-')
            
        times=times[0:10]+"T"+times[11:]
        
        
        return Time(times, format='isot',scale='utc')
       
        
    def get_selfcal_times_paths(self):
        msfiles=list_msfiles(self.file_path)   
        filtered_msfiles=self.filter_msfiles(msfiles,self._start,self._end,self.freqstr)  ### contain file list for first frequency
         
        if len(filtered_msfiles)==0:
        	return
        
        self.image_times=self.get_image_times(filtered_msfiles)  ### this is sorted
        del filtered_msfiles
        
        self.get_selfcal_times()
        self.num_selfcal_times=len(self.selfcal_times)
        self.unique_file_locs=self.get_unique_file_locs(msfiles)
        return
        
    def get_unique_file_locs(self,msfiles):
         
         unique_paths=[]
          
         for freq in self.freqstr:
            for f1 in msfiles:
                path=f1['path']
                if f1['freq']==freq:
                    unique_paths.append('/'.join(path.split('/')[:-1]))
                    break
         return unique_paths           
                    
    def get_current_path(self,current_freq):
       for freq,path in zip(self.freqstr,self.unique_file_locs):
           
           if freq==current_freq:
               self.current_path=path
               break
       
    
                    
    def get_current_file_for_imaging(self,current_freq):
        self.current_file_index+=1
        try:
            file_path=file_downloader(self.image_times[self.current_file_index],self.current_path,current_freq)
           
            return file_path
        except:
            return None
   
    def get_current_file_for_selfcal(self,current_freq):
        self.get_current_path(current_freq)
        if self.file_loop_started==False:
            file_path=file_downloader(self.selfcal_times[self.current_file_index],self.current_path,current_freq)
           
            self.file_loop_started=True
            return file_path
        else:
            self.current_file_index+=1
            try:
                file_path=file_downloader(self.selfcal_times[self.current_file_index],self.current_path,current_freq)
                
                return file_path
            except:
                self.current_file_index=-1
                return None         
        
        
    def get_selfcal_times(self):
        ### we can put here any restriction we want on the full selfcal times
        ### like elevetaion limit of Sun etc.
        ### For now I am not putting any restriction
        
        
        
        #### Putting in time for the full DI selfcal first ####
        self.selfcal_times.append(self.image_times[0])
        
        
        prev_time=Time(self.selfcal_times[0][0],format='isot')
        
        for j,t1 in enumerate(self.image_times):
            if j==0:
                continue
            
            current_time=Time(t1[0],format='isot')
            
            diff=(current_time-prev_time).value*86400
            if diff>=self.full_DI_selfcal_solint:
                self.selfcal_times.append(t1)
                prev_time=current_time
        
        if self.selfcal_times[-1][0]!=t1:
            self.selfcal_times.append(self.image_times[-1])  ### will end with the last time
                                                             ### so that if needed if we can
                                                             ### implement interpolation later
        
        
        #### Putting in times for partial DI selfcal next ###
        prev_time=Time(self.selfcal_times[0][0],format='isot')
           
        for j,t1 in enumerate(self.image_times):
            if j==0:
                continue
            current_time=Time(t1[0],format='isot')
            diff=(current_time-prev_time).value*86400
            if diff>=self.partial_DI_selfcal_solint:
                self.selfcal_times.append(t1)
                prev_time=current_time
        
        #### Putting in times for full DD selfcal next ###
                
        prev_time=Time(self.selfcal_times[0][0],format='isot')
           
        for j,t1 in enumerate(self.image_times):
            if j==0:
                continue
            current_time=Time(t1[0],format='isot')
            diff=(current_time-prev_time).value*86400
            if diff>=self.full_DD_selfcal_solint:
                self.selfcal_times.append(t1)
                prev_time=current_time
       
       #### Putting in times for partial DD selfcal next ###
        
        prev_time=Time(self.selfcal_times[0][0],format='isot')
           
        for j,t1 in enumerate(self.image_times):
            if j==0:
                continue
            current_time=Time(t1[0],format='isot')
            diff=(current_time-prev_time).value*86400
            if diff>=self.partial_DD_selfcal_solint:
                self.selfcal_times.append(t1)
                prev_time=current_time 
      
      ### The times might be repeated. But when sending to image, we can check if the image exists      
           
        return
        
        

    @staticmethod
    def filter_msfiles(msfiles,tstart,tend,freqstr):
        '''
        msfiles: List of msfiles returned by list_msfiles
        tstart,tend: Start and end of observations. Both in Astropy time format
        freqstr: Frequency str. Format: ["66MHz","78 MHz"]
        '''     
        num_files=len(msfiles) 
        
        
        filtered=[]
        
        current_freq=''    
        for j,f1 in enumerate(msfiles):
            timestr=f1['time']
            t1=Time(timestr,scale='utc',format='isot')
            start_diff=(t1-tstart).value
            end_diff=(tend-t1).value
            
            if len(current_freq)==0: 
                if start_diff>=0 and end_diff>=0 and f1['freq'] in freqstr:
                    filtered.append(f1)
                    current_freq=f1['freq']
            else:
                if start_diff>=0 and end_diff>=0 and f1['freq']==current_freq:
                    filtered.append(f1)    
                    
        return filtered   


    def get_files_for_integration(self,sorted_msfiles,current_index,current_time):
        integrated=0
        temp=[]
        f1=sorted_msfiles[current_index]
        while integrated<self.time_integration:
            if integrated==0:
    	        temp.append(f1['time'])
    	        current_time_for_integration=Time(f1['time'],format='isot')
    	        diff=(current_time_for_integration-current_time).value*86400
    	        
            else:
                f1=sorted_msfiles[current_index]                     
                current_time_for_integration=Time(f1['time'],format='isot')
                diff=(current_time_for_integration-current_time).value*86400
                temp.append(f1['time'])
            integrated+=diff+self.observation_integration
            
            current_index+=1    	
            
        return current_index,temp
               
    def get_image_times(self,msfiles):
    
        sorted_msfiles=sorted(msfiles,key=lambda file1:file1['time'])

        
        len_msfiles=len(sorted_msfiles)
        
        filtered=[]

        prev_time=Time(sorted_msfiles[0]['time'],format='isot')
        
        current_index,temp=self.get_files_for_integration(sorted_msfiles,0,prev_time)
        
        filtered.append(temp)
        j=current_index

        while j<len_msfiles:
            f1=sorted_msfiles[j]
            
            current_time=Time(f1['time'],format='isot')
            diff=(current_time-prev_time).value*86400
            
            if round(diff)>=self.time_cadence:
            
                current_index,temp=self.get_files_for_integration(sorted_msfiles,j,current_time)
                filtered.append(temp)
                prev_time=current_time
                del temp
                j=current_index
                
            else:
                j+=1
        
        return filtered


        