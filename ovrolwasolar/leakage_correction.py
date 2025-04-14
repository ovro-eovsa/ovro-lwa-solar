from . import refraction_correction as rfc
from astropy.io import fits
import pandas as pd
from shutil import copyfile
from astropy.time import Time
from suncasa.io import ndfits
import numpy as np
from . import utils
import os,logging,h5py

class leakage_database():
    def __init__(self,database):
        self.stokes_num=4
        self.leakage_database=database
        self.leakage_freq_sep=5
        self.low_freq=17.5
        self.high_freq=90.5
        
    
    def add_database_headers(self):
        with h5py.File(self.leakage_database,'a') as hf:
            hf.attrs['freq_sep']=self.leakage_freq_sep
            hf.attrs['low_freq']=self.low_freq
            hf.attrs['high_freq']=self.high_freq
        
    def write_leakage_frac_to_database(self,datetime_mjd,alt,az,freqs,leak_frac):
        '''
        :param leakage_alt/az_sep: The alt/az separation in which data will be written to database
                                 in MHz. This can vary in each solve instance.
        :type : float
        :param datetime_mjd: mjd of the time you want to write
        :type datetime_mjd: float
        :param alt: Altitude in degree
        :type alt: float
        :param az: Azimuth in degree
        :type az: float
        :param freqs: Frequencies for which the data is being written
        :type freqs: numpy ndarray
        :param leak_frac: Array with shape num_freqs x 4. the 4 pols are 
                            arranged in I_frac, Q_frac, U_frac, Vfrac. I_frac is ignored
        :type leak_frac: list
        
        Only write to database if you have values for Q,U,V
        '''
        
        entry_to_write={}
        entry_to_write['datetime_mjd']=datetime_mjd
        entry_to_write['alt']=alt
        entry_to_write['az']=az
        
        for j,freq1 in enumerate(freqs):
            entry_to_write['Q/I_'+str(int(freq1))+"MHz"]=leak_frac[j,1]
            entry_to_write['U/I_'+str(int(freq1))+"MHz"]=leak_frac[j,2]
            entry_to_write['V/I_'+str(int(freq1))+"MHz"]=leak_frac[j,3]
            
        self.add_leakage_entry(entry_to_write)
        self.add_database_headers()
        return
    
    def remove_rows_from_leakage_database(database,mjd_to_drop):
        '''
        Reads the database file, filters out and then recreates with same name
        '''
        data=pd.read_hdf(database,'I_leakage')
        datetime_mjd=data['datetime_mjd']
        pos=np.where(abs(datetime_mjd-mjd_to_drop)>=1)[0]
        
        keys=data.keys()
        
        new_data={}
        for key in keys:
            new_data[key]=np.array(data[key])[pos]
        
        df=pd.DataFrame(new_data)
        df.to_hdf(database,key='I_leakage',index=False,format='table',complevel=3)
        return
    
    def add_leakage_entry(self,entry):
        '''
        entry should be a pandas dataframe/dictionary with the following keys
        datetime_mjd, az (degrees), el (degrees), Q/I leakage, U/I leakage, V/I leakage
        
        column names: datetime_mjd, az,el, Q/I_34MHz, U/I_34MHz, V/I_34MHz, \
                        Q/I_34MHz_mean_sub, U/I_34MHz_mean_sub, V/I_34MHz_mean_sub
        Similar to this we do the following frequencies 34, 39, 43, 48, 52,\
        57, 62, 66, 71, 75, 80 and 84 MHz. 
        
        Leakage values more than 100, implies no solution found.
        
        Also if database exists, we assume that some entries also exist. 
        
        Only appends entries which have unique datetimes
        '''
        if not hasattr(self,"leakage_database"):
            if not database:
                raise RuntimeError("Please provide leakage database name")
            self.leakage_database=database
         
        df=pd.DataFrame(entry,index=[0])
        
        if not os.path.isfile(self.leakage_database):
            logging.debug("Leakage database does not exist. Creating one")
            ## using table format for appending, searching, selecting subsets
            df.to_hdf(self.leakage_database,key='I_leakage',index=False,format='table',complevel=3)
            return
        ## This will only be done if the file already exists
        existing_df=pd.read_hdf(self.leakage_database,key='I_leakage',columns=['datetime_mjd'])
        
        ## remove duplicate datetime entries
        old_datetimes=existing_df['datetime_mjd']
        new_datetimes=df['datetime_mjd']
        times,old_frame_ind, new_frame_ids=np.intersect1d(old_datetimes,new_datetimes, return_indices=True)
        
        if new_frame_ids.size==new_datetimes.size:
            logging.debug("No unique entries found. Exiting")
            print ("No unique entries found. Exiting")
            return
            
        df.drop(index=new_frame_ids)
        
        ## append the new entries to the database
        ## using table format for appending, searching, selecting subsets
        df.to_hdf(self.leakage_database,key='I_leakage',mode='r+',append=True,\
                        index=False,complevel=3,format='table')
        return
    
    def get_leakage_from_database(self):
        '''
        This reads the database and produces the leakage fractions for all
        stokes by doing a nearest neighbour interpolation in alt-az and linear
        interpolation in frequency.
        '''
        self.determine_beam_leakage_fractions_from_db(max_pol_ind=3)
    
    def determine_leakage_fractions_from_db(self,alt,az,freqs):
        '''
        This function reads in the leakage database and then uses it to calculate the
        leakage fraction at all the data times
        :param freqs: Frequencies for which leakage is needed in MHz
        :param alt: altitude of source in degree
        :param az: azimuth of source in degree
        '''
        
        with h5py.File(self.leakage_database,'r') as hf:
            freqs_in_db=np.arange(hf.attrs['low_freq'],hf.attrs['high_freq'],hf.attrs['freq_sep'],dtype=int)
        
        data=pd.read_hdf(self.leakage_database,key='I_leakage',columns=['az','alt'])
        database_az=np.array(data['az'])
        database_alt=np.array(data['alt'])
        
        num_freqs=freqs.size
        leakage_fractions=np.zeros((4,num_freqs))
        
        for i in range(num_freqs):
            data_freq=freqs[i]
            freq1,freq2=self.choose_freqs_to_load(freqs_in_db,data_freq)
            print (data_freq,freqs_in_db,freq1,freq2)
            
            if not freq1 or not freq2:
                leakage_fractions[1:,i]=np.nan
                continue
            for j,pol in zip(range(1,4),['Q/I','U/I','V/I']):
                
                leakage_fractions[j,i]=self.calculate_leakages_from_database(data_freq,freq1,freq2,\
                                            pol,az,alt,database_az,database_alt)  
        return leakage_fractions
    
    @staticmethod    
    def choose_freqs_to_load(freqs_in_db,data_freq):
        '''
        This function provides the 2 nearest frequencies (greater and smaller) to the
        current frequency for which the leakages are needed. This is done to reduce
        the amount of data loaded. We will only load these particular columns
        :param freqs_in_db: Frequencies (in MHz) present in the database. This is not read, but generated
                            based on the header present in the hdf5 file.
        :type freqs_in_db : np.array
        :param data_freq: The current frequency in MHz for which leakages are required
        :type data_freq: float
        '''
        if data_freq<freqs_in_db[0]:
            return freqs_in_db[0],None
        if data_freq>freqs_in_db[-1]:
            return  freqs_in_db[-1], None  
        
        indices=np.argsort(np.abs(freqs_in_db-data_freq))
        
        return freqs_in_db[indices[0]],freqs_in_db[indices[1]]    

    def calculate_leakages_from_database(self,data_freq,freq1,freq2,pol,az,alt,database_az,database_alt):
        columns=[pol+"_"+str(int(freq1))+"MHz",pol+"_"+str(int(freq2))+"MHz"]

        data=pd.read_hdf(self.leakage_database,key='I_leakage',columns=columns)
        
        leak1_database_azalt=np.array(data[columns[0]])
        leak1_data_azalt=griddata((database_az,database_alt),leak1_database_azalt,(az,alt),method='nearest')
        
        leak2_database_azalt=np.array(data[columns[1]])
        leak2_data_azalt=griddata((database_az,database_alt),leak2_database_azalt,(az,alt),method='nearest')
        
        leak_data_azalt=(leak2_data_azalt-leak1_data_azalt)/(freq2-freq1)*(data_freq-freq1)+\
                                         leak1_data_azalt ## fitting line
        
        if abs(freq2-freq1)<=self.freq_sep: ### I am putting a small tolerance here.
            return leak_data_azalt
        if abs(freq2-freq)>self.freq_sep:
            return leak2_data_azalt
        if abs(freq1-freq)>self.freq_sep:
            return leak1_data_azalt
        return np.nan

def find_robust_median(data,thresh=5):
    median=np.nanmedian(data)
    for i in range(5):
        mad=np.nanmedian(np.abs(data-median))
        pos=np.where(np.abs(data-median)>thresh*mad)[0]
        data[pos]=np.nan
        median=np.nanmedian(data)
    return median

def determine_leakage_single_freq(stokes_data,frequency, background_factor, \
                                    min_size, overbright, stokes_order, min_pix=100):
    
    Iind=stokes_order.index('I')
    Idata=stokes_data[Iind,:,:]
    pos=find_quiet_sun_pixels(Idata,frequency, background_factor, min_size, overbright)
  
    if pos[0].size<min_pix:
        return False,np.zeros(4)
    
    leak_frac=np.zeros(len(stokes_order))
    for j,stokes in enumerate(stokes_order):
        if stokes!='I':
            leak=find_robust_median(stokes_data[j][pos]/Idata[pos])
            leak_frac[j]=leak
    return True,leak_frac
    
        
def find_quiet_sun_pixels(Idata,frequency, background_factor, min_size, overbright):
    thresh=rfc.thresh_func(frequency)*background_factor
    pos=np.where((Idata>thresh) & (Idata<overbright))
    return pos
    
                     
def determine_multifreq_leakage(fname,overbright=1.0e6,background_factor=1/8,\
                                min_size_50=1000,\
                                bands=['32MHz', '36MHz', '41MHz', \
                    '46MHz', '50MHz', '55MHz', '59MHz', '64MHz', '69MHz', '73MHz', \
                    '78MHz', '82MHz']):
    '''
    image_cube is multi-frequency, multi-stokes image 4D
    images for one time and frequency. While presence of I
    image is essential, either all of Q,U,V images, or any 1
    or 2 of them can be present.
    Shape: Stokes X Freq x spatial axis x spatial axis
    '''
    
    
    meta, data = ndfits.read(fname)
    
    meta_header=meta['header']
    
    shape=data.shape
    num_stokes=shape[0]
    num_freqs=shape[1]
    
    leak_frac=np.zeros((num_freqs,4))
    
    stokes_order=meta_header['polorder'].split(',')
    
    arranging_order={'I':0,'Q':1,'U':2,'V':3}
    
    for freq_ind in range(num_freqs):
        frequency=meta['ref_cfreqs'][freq_ind]
        min_size = min_size_50/(meta_header['CDELT1']/60.)**2./(frequency/50e6)**2.
        success,leak=determine_leakage_single_freq(data[:,freq_ind,:],frequency,\
                                            background_factor, min_size, overbright, stokes_order)
        if success:
            for j,stokes in enumerate(stokes_order):
                leak_frac[freq_ind,arranging_order[stokes]]=leak[j]
        else:
            leak_frac[freq_ind,:]=-1000 ### dummy number to show 
                                                               ### that this could not be determined
    return leak_frac

def write_to_database(fname,leak_frac,outfile):
    meta, data = ndfits.read(fname)    
    meta_header=meta['header']
    freqs=meta['ref_cfreqs']*1e-6
    
    datetime=Time(meta_header['DATE-OBS'])
    az,alt=utils.get_solar_altaz_multiple_times(datetime)
    
    db=leakage_database(outfile)
    pos=np.where(leak_frac>-10)[0] ## searching for dummy number. Dummy number is -1000
    if pos.size!=0:
        db.write_leakage_frac_to_database(datetime.mjd,alt,az,freqs[pos],leak_frac[pos])
    return
    
def do_leakage_correction(image_cube,primary_beam_database,outfile=None):
    if outfile is None:
        outfile = './' + os.path.basename(image_cube).replace('lev1.5','lev2.5')
        outfile = './' + os.path.basename(image_cube).replace('lev1','lev2')
        
    copyfile(image_cube, outfile)
    meta, data = ndfits.read(image_cube)
    
    shape=data.shape
    num_stokes=shape[0]
    num_freqs=shape[1]
    freq_MHz=meta['ref_cfreqs']*1e-6
    meta_header=meta['header']
    stokes_order=meta_header['polorder'].split(',')
    
    datetime=Time(meta_header['DATE-OBS'])
    az,alt=utils.get_solar_altaz_multiple_times(datetime)
    
    db=leakage_database(primary_beam_database)
    leak_frac=db.determine_leakage_fractions_from_db(alt,az,freq_MHz)
    print (leak_frac)
    arrange_order={'I':0,'Q':1,'U':2,'V':3}
    
    I_ind=stokes_order.index('I')
    leakage_to_write=np.zeros((num_freqs,len(stokes_order)))
    
    for freq_ind,frequency in enumerate(freq_MHz):
        for stokes_ind,stokes in enumerate(stokes_order):
            if stokes!='I':
                data[stokes_ind,freq_ind,:,:]-=leak_frac[arrange_order[stokes],freq_ind]*\
                                                data[I_ind,freq_ind,:,:]
                if np.isnan(leak_frac[arrange_order[stokes],freq_ind]):
                    leak_frac[arrange_order[stokes],freq_ind]=-1000
                leakage_to_write[freq_ind,stokes_ind]=leak_frac[arrange_order[stokes],freq_ind]
    
    cols=[]
    for stokes_ind,stokes in enumerate(stokes_order):
        fitscol=fits.Column(name=stokes+"_leak",format='E',array=leakage_to_write[:,stokes_ind])  
                    ## E stands for single precision float (32-bit). Change to D for double precision
                    ## see https://docs.astropy.org/en/stable/io/fits/usage/table.html#column-creation
        cols.append(fitscol)
    header={}
    header['leakage_corrected']=True
    header['dummy_leakage']=-1000
    ndfits.update(outfile,new_data=data,new_columns=cols, new_header_entries=header)
    return outfile
    


    
                
                                                
    
    
            
        
        
    
    
    

            
    
    
