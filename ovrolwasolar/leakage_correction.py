from . import refraction_correction as rfc
from astropy.io import fits
import pandas as pd
from shutil import copyfile
from astropy.time import Time
from suncasa.io import ndfits
import numpy as np
from . import utils
import os,logging,h5py
from scipy.interpolate import griddata,interp1d

class leakage_database():
    def __init__(self,database):
        self.stokes_num=4
        self.leakage_database=database
        self.leakage_freq_sep=1
        self.low_freq=30
        self.high_freq=90
        
    
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
            freq1,freq2=self.choose_freqs_to_load(freqs_in_db,data_freq,self.leakage_freq_sep)

            
            if not freq1 and not freq2:
                leakage_fractions[1:,i]=np.nan
                continue
            for j,pol in zip(range(1,4),['Q/I','U/I','V/I']):
                leakage_fractions[j,i]=self.calculate_leakages_from_database(data_freq,freq1,freq2,\
                                            pol,az,alt,database_az,database_alt)  
        return leakage_fractions
    
    @staticmethod    
    def choose_freqs_to_load(freqs_in_db,data_freq,freq_sep):
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

        if data_freq<freqs_in_db[0]-freq_sep/2:
            return None,None
        if data_freq>freqs_in_db[-1]+freq_sep/2:
            return  None, None  
        
        indices=np.argsort(np.abs(freqs_in_db-data_freq))
        
        return freqs_in_db[indices[0]],freqs_in_db[indices[1]]    

    def calculate_leakages_from_database(self,data_freq,freq1,freq2,pol,az,alt,database_az,database_alt):
    
        if freq1:
            columns=[pol+"_"+str(int(freq1))+"MHz"]
            try:
                data=pd.read_hdf(self.leakage_database,key='I_leakage',columns=columns)
                leak1_database_azalt=np.array(data[columns[0]])
                pos=np.where(leak1_database_azalt>-10)  ###-1000 is used as dummy leakage
                leak1_data_azalt=griddata((database_az[pos],database_alt[pos]),leak1_database_azalt[pos],(az,alt),method='nearest')  
            except KeyError:
                logging.warning("Expected Frequency does not exist:%s ",str(int(freq1))+"MHz")
                freq1=None
        
        if freq2:
            columns=[pol+"_"+str(int(freq2))+"MHz"]
            try:
                data=pd.read_hdf(self.leakage_database,key='I_leakage',columns=columns)
                leak2_database_azalt=np.array(data[columns[0]])
                pos=np.where(leak2_database_azalt>-10)  ###-1000 is used as dummy leakage
                leak2_data_azalt=griddata((database_az[pos],database_alt[pos]),leak2_database_azalt[pos],(az,alt),method='nearest')  
            except KeyError:
                logging.warning("Expected Frequency does not exist:%s ",str(int(freq2))+"MHz")
                freq2=None

        
        
        
        if freq1 and freq2:
            leak_data_azalt=(leak2_data_azalt-leak1_data_azalt)/(freq2-freq1)*(data_freq-freq1)+\
                                         leak1_data_azalt ## fitting line
            return leak_data_azalt
            
        if freq1:
            return leak1_data_azalt
        
        if freq2:
            return leak2_data_azalt
        
        return np.nan

def find_robust_median(data,thresh=5):
    median=np.nanmedian(data)
    for i in range(5):
        mad=np.nanmedian(np.abs(data-median))
        pos=np.where(np.abs(data-median)>thresh*mad)[0]
        data[pos]=np.nan
        median=np.nanmedian(data)
    return median
    
def plot_stokes_images(stokes_data,leak_frac,stokes_order):
    import matplotlib.pyplot as plt
    for j,pol in enumerate(stokes_order):
        if pol=='I':
            Idata=stokes_data[j]
            
    shape=Idata.shape
    x=np.arange(0,shape[0],1)
    X,Y=np.meshgrid(x,x)
    contour_levels=np.array([0.05,0.2,0.4,0.6])*np.nanmax(Idata)
    
    fig,ax=plt.subplots(nrows=2,ncols=2,figsize=[10,8])
    ax=ax.flatten()
            
    for j,ax1 in enumerate(ax):
        im=ax1.imshow(stokes_data[j]-leak_frac[j]*Idata,origin='lower') 
        plt.colorbar(im,ax=ax1)
        ax1.set_title(stokes_order[j])
        ax1.contour(X,Y,Idata,levels=contour_levels,colors='r')
        ax1.set_xlim([75,170])
        ax1.set_ylim([75,170])
    return fig
           

def determine_leakage_single_freq(stokes_data,frequency, background_factor, \
                                    min_size, overbright, stokes_order, min_pix=100,\
                                    doplot=False):
    
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
    
    if doplot:
        fig1=plot_stokes_images(stokes_data,[0,0,0,0],stokes_order)
        fig2=plot_stokes_images(stokes_data,leak_frac,stokes_order)
        plt.show()
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
    
    obstime=Time(meta_header['DATE-OBS'])
    az,alt=utils.get_solar_altaz_multiple_times(obstime)
    
    sinc_beam_val=np.sin(alt*np.pi/180)**1.6
    
    for freq_ind in range(num_freqs):
        frequency=meta['ref_cfreqs'][freq_ind]
        min_size = min_size_50/(meta_header['CDELT1']/60.)**2./(frequency/50e6)**2.
        success,leak=determine_leakage_single_freq(data[:,freq_ind,:]/sinc_beam_val,frequency,\
                                            background_factor, min_size, overbright, stokes_order)
        #### leak is a fractional. So dividing all Stokes by same value does not change the leakage fraction
        #### The only time absolute values are used is when determining the quiet Sun, and there a simple
        #### sin beam is fine.
        if success:
            for j,stokes in enumerate(stokes_order):
                leak_frac[freq_ind,arranging_order[stokes]]=leak[j]
        else:
            leak_frac[freq_ind,:]=-1000 ### dummy number to show 
                                                               ### that this could not be determined
    return leak_frac

def write_to_database(fname,leak_frac,database,low_freq=30,high_freq=90,freq_sep=1):
    meta, data = ndfits.read(fname)    
    meta_header=meta['header']
    freqs=meta['ref_cfreqs']*1e-6
    
    datetime=Time(meta_header['DATE-OBS'])
    az,alt=utils.get_solar_altaz_multiple_times(datetime)
    
    db=leakage_database(database)
    db.low_freq=low_freq
    db.high_freq=high_freq
    db.leakage_freq_sep=freq_sep
    
    
    
    
    database_freqs=np.arange(low_freq,high_freq,freq_sep,dtype=int)
    shape=leak_frac.shape
    num_pol=shape[1]
    leak_frac_database=np.zeros((database_freqs.size,num_pol))
    for i in range(num_pol):
        leak_frac_pol=leak_frac[:,i]
        pos=np.where(leak_frac_pol>-10)[0] ## searching for dummy number. Dummy number is -1000
        if pos.size>0:
            leak_frac_func=interp1d(freqs[pos],leak_frac_pol[pos],bounds_error=False,fill_value=np.nan) ###out of bounds gives nan
            leak_frac_database[:,i]=leak_frac_func(database_freqs)
        else:
            leak_frac_database[:,i]=np.nan
        pos=np.where(np.isnan(leak_frac_database)==True)
        leak_frac_database[pos]=-1000
        db.write_leakage_frac_to_database(datetime.mjd,alt,az,database_freqs,leak_frac_database)
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
    


    
                
                                                
    
    
            
        
        
    
    
    

            
    
    
