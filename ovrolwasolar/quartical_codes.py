import xarray,dask.array as da,dask
import numpy as np,os
from daskms.experimental.zarr import xds_from_zarr,xds_to_zarr

def mean_channel_leakage_based_flagging(gains,gain_flags,max_leak_amp=0.8,flagmode='calonly',thresh=5):
    '''
    This calculate the mean leakage over channels for each antenna. 
    If mean leakage is greater than 1, flags it.
    '''
    shape=gains.shape
    
    amp_gain=np.abs(gains)
    
    mean_amp_gain=np.expand_dims(np.nanmedian(amp_gain,axis=1),axis=1)
    pos=np.where(mean_amp_gain[:,:,:,:,1]>max_leak_amp)
    ants=pos[2]
    for i in ants:
        gains[:,:,i,:,:]=np.nan
    
    pos=np.where(mean_amp_gain[:,:,:,:,2]>max_leak_amp)
    ants=pos[2]
    for i in ants:
        gains[:,:,i,:,:]=np.nan
    
    if flagmode!='calonly':
        gain_flags[(pos[0],shape[1]*np.ones(len(pos[0]),dtype=int),pos[2],pos[3])]=1
        
    mean_amp_gain=np.expand_dims(np.nanmedian(amp_gain,axis=1),axis=1)
    std_amp_gain=np.expand_dims(np.nanstd(amp_gain,axis=1),axis=1)
    
    diff=amp_gain-mean_amp_gain
    
    pos=np.where(abs(diff)>thresh*std_amp_gain)
    gains[(pos[0],pos[1],pos[2],pos[3],1*np.ones(len(pos[0]),dtype=int))]=np.nan
    gains[(pos[0],pos[1],pos[2],pos[3],2*np.ones(len(pos[0]),dtype=int))]=np.nan
    
    if flagmode!='calonly':
        gain_flags[(pos[0],pos[1],pos[2],pos[3])]=1
        
    return    

def mean_antenna_leakage_based_flagging(gains,gain_flags,flagmode='calonly',thresh=5):
    shape=gains.shape
    
    amp_gain=np.abs(gains)
    
    mean_amp_gain=np.expand_dims(np.nanmedian(amp_gain,axis=2),axis=2)
    
    mean_amp_gain=np.expand_dims(np.nanmedian(amp_gain,axis=2),axis=2)
    std_amp_gain=np.expand_dims(np.nanstd(amp_gain,axis=2),axis=2)
    
    diff=amp_gain-mean_amp_gain
    
    pos=np.where(abs(diff)>thresh*std_amp_gain)
    gains[(pos[0],pos[1],pos[2],pos[3],1*np.ones(len(pos[0]),dtype=int))]=np.nan
    gains[(pos[0],pos[1],pos[2],pos[3],2*np.ones(len(pos[0]),dtype=int))]=np.nan
    
    if flagmode!='calonly':
        gain_flags[(pos[0],pos[1],pos[2],pos[3])]=1
    return
    
def std_antenna_leakage_based_flagging(gains,gain_flags,flagmode='calonly',thresh=3):
    shape=gains.shape
    
    amp_gain=np.abs(gains)
    std_gain=np.expand_dims(np.nanstd(amp_gain,axis=1),axis=1)
    mean_std_gain=np.expand_dims(np.nanmedian(std_gain,axis=2),axis=2)
    std_std_gain=np.expand_dims(np.nanstd(std_gain,axis=2),axis=2)
    
    
    diff=std_gain-mean_std_gain
    
    pos=np.where(abs(diff)>thresh*std_std_gain)
    
    ants=pos[2]
    
    for i in ants:
        gains[:,:,i,:,:]=np.nan
    
    if flagmode!='calonly':
        gain_flags[(pos[0],shape[1]*np.ones(len(pos[0]),dtype=int),pos[2],pos[3])]=1
    return
    
def max_leak_based_flagging(gains,gain_flags,flagmode='calonly',max_leak_amp=1.0):
    pos=np.where(np.abs(gains[:,:,:,:,1])>max_leak_amp)
    
    gains[(pos[0],pos[1],pos[2],pos[3],1*np.ones(len(pos[0]),dtype=int))]=np.nan
    gains[(pos[0],pos[1],pos[2],pos[3],2*np.ones(len(pos[0]),dtype=int))]=np.nan
    
    if flagmode!='calonly':
        gain_flags[(pos[0],pos[1],pos[2],pos[3])]=1
    
    pos=np.where(np.abs(gains[:,:,:,:,2])>max_leak_amp)
    
    gains[(pos[0],pos[1],pos[2],pos[3],1*np.ones(len(pos[0]),dtype=int))]=np.nan
    gains[(pos[0],pos[1],pos[2],pos[3],2*np.ones(len(pos[0]),dtype=int))]=np.nan
    if flagmode!='calonly':
        gain_flags[(pos[0],pos[1],pos[2],pos[3])]=1
    return
    
def flag_quartical_caltable(caltable,term_name='D',thresh=5,max_leak_amp=1.0,flagmode='calonly',inplace=False):
    data=xds_from_zarr(caltable+"::"+term_name)
    gains=np.asarray(data[0].gains)
    #### shape= (gain_time, gain_freq, antenna, direction, correlation)  for leakage solutions
    gain_flags=np.asarray(data[0].gain_flags)
   
    pos=np.where(gain_flags==1)
    gains[(pos[0],pos[1],pos[2],pos[3],1*np.ones(len(pos[0]),dtype=int))]=np.nan
    gains[(pos[0],pos[1],pos[2],pos[3],2*np.ones(len(pos[0]),dtype=int))]=np.nan
    
    mean_channel_leakage_based_flagging(gains,gain_flags,max_leak_amp=max_leak_amp,flagmode=flagmode,thresh=thresh)
    
    mean_antenna_leakage_based_flagging(gains,gain_flags,flagmode=flagmode,thresh=thresh)
    
    max_leak_based_flagging(gains,gain_flags,flagmode=flagmode,max_leak_amp=max_leak_amp)
    
    std_antenna_leakage_based_flagging(gains,gain_flags,flagmode=flagmode,thresh=thresh)
    
    pos=np.where(np.isnan(gains))
    gains[pos]=0.0
    
    data[0].update({'gains': (['gain_t', 'gain_f', 'ant', 'dir', 'corr'],gains)})
    data[0].update({'gain_flags': (['gain_t', 'gain_f', 'ant', 'dir'],gain_flags)})
    
    if inplace==False:
        output_name=caltable+'.modified'
    else:
        os.system('rm -rf '+caltable)
        output_name=caltable
    output_path = f"{output_name}{'::' + term_name}"
    write_xds_list = xds_to_zarr(data, output_path)
    dask.compute(write_xds_list)
    return output_name
	
   
