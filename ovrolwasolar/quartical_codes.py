import xarray,dask.array as da,dask
import numpy as np,os
from daskms.experimental.zarr import xds_from_zarr,xds_to_zarr

def mean_channel_leakage_based_flagging(gains,gain_flags,max_leak_amp=1.0,thresh=5):
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
    
    
        
    mean_amp_gain=np.expand_dims(np.nanmedian(amp_gain,axis=1),axis=1)
    std_amp_gain=np.expand_dims(np.nanstd(amp_gain,axis=1),axis=1)
    
    diff=amp_gain-mean_amp_gain
    
    pos=np.where(abs(diff)>thresh*std_amp_gain)
    gains[(pos[0],pos[1],pos[2],pos[3],1*np.ones(len(pos[0]),dtype=int))]=np.nan
    gains[(pos[0],pos[1],pos[2],pos[3],2*np.ones(len(pos[0]),dtype=int))]=np.nan
    
   
        
    return    

def mean_antenna_leakage_based_flagging(gains,gain_flags,thresh=5,max_mean_leakage=0.5):
    shape=gains.shape
    
    amp_gain=np.abs(gains)
    
    mean_amp_gain=np.expand_dims(np.nanmedian(amp_gain,axis=2),axis=2)
    
    pos=np.where(mean_amp_gain>max_mean_leakage)
    ants=pos[2]
    for i in ants:
        gains[:,:,i,:,:]=np.nan

    mean_amp_gain=np.expand_dims(np.nanmedian(amp_gain,axis=2),axis=2)
    
    std_amp_gain=np.expand_dims(np.nanstd(amp_gain,axis=2),axis=2)
    
    diff=amp_gain-mean_amp_gain
    
    pos=np.where(abs(diff)>thresh*std_amp_gain)
    gains[(pos[0],pos[1],pos[2],pos[3],1*np.ones(len(pos[0]),dtype=int))]=np.nan
    gains[(pos[0],pos[1],pos[2],pos[3],2*np.ones(len(pos[0]),dtype=int))]=np.nan
    
    
    return
    
def std_antenna_leakage_based_flagging(gains,gain_flags,thresh=3):
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
    
    
    return
    
def max_leak_based_flagging(gains,gain_flags,flagmode='calonly',max_leak_amp=0.7):
    pos=np.where(np.abs(gains[:,:,:,:,1])>max_leak_amp)
    
    gains[(pos[0],pos[1],pos[2],pos[3],1*np.ones(len(pos[0]),dtype=int))]=np.nan
    gains[(pos[0],pos[1],pos[2],pos[3],2*np.ones(len(pos[0]),dtype=int))]=np.nan
    
    
    
    pos=np.where(np.abs(gains[:,:,:,:,2])>max_leak_amp)
    
    gains[(pos[0],pos[1],pos[2],pos[3],1*np.ones(len(pos[0]),dtype=int))]=np.nan
    gains[(pos[0],pos[1],pos[2],pos[3],2*np.ones(len(pos[0]),dtype=int))]=np.nan
   
    return
    
def flag_quartical_caltable(caltable,term_name='D',thresh=5,max_leak_amp=1.0,\
                            flagmode='calonly',inplace=False, outcal=None):
    data=xds_from_zarr(caltable+"::"+term_name)
    axis_names=data[0].attrs['GAIN_AXES']
    gains=np.asarray(data[0].gains)
    #### shape= (gain_time, gain_freq, antenna, direction, correlation)  for leakage solutions
    gain_flags=np.asarray(data[0].gain_flags)
   
    pos=np.where(gain_flags==1)
    gains[(pos[0],pos[1],pos[2],pos[3],1*np.ones(len(pos[0]),dtype=int))]=np.nan
    gains[(pos[0],pos[1],pos[2],pos[3],2*np.ones(len(pos[0]),dtype=int))]=np.nan
    
    
    mean_channel_leakage_based_flagging(gains,gain_flags,max_leak_amp=max_leak_amp,thresh=thresh)
    
    mean_antenna_leakage_based_flagging(gains,gain_flags,thresh=thresh)
    
    max_leak_based_flagging(gains,gain_flags,max_leak_amp=max_leak_amp)
    
    std_antenna_leakage_based_flagging(gains,gain_flags,thresh=thresh)
    
    if flagmode=='calonly':
        pos=np.where(np.isnan(gains))
        gains[pos]=0.0
        gain_flags_new=np.zeros_like(gain_flags)
        data[0].update({'gains': (axis_names,gains)})
        data[0].update({'gain_flags': (axis_names[:-1],gain_flags_new)})
    else:
        pos=np.where(np.isnan(gains))
        gain_flags[(pos[0],pos[1],pos[2],pos[3])]=1
        print (gain_flags[0,0,10,0])
        gains[pos]=0.0
        data[0].update({'gains': (axis_names,gains)})
        data[0].update({'gain_flags': (axis_names[:-1],gain_flags)})
    
    
    
    if inplace==False:
        if outcal is None:
            outcal=caltable+'.modified'
    else:
        os.system('rm -rf '+caltable)
        outcal=caltable
    output_path = f"{outcal}{'::' + term_name}"
    write_xds_list = xds_to_zarr(data, output_path)
    dask.compute(write_xds_list)
    return outcal

def do_poldist_norm(caltable,soltype='D',inplace=False,outcal=None):
    '''
    Function to make poldistortion normalization (Normalization of full Jones solutions)
    Note : for mathematical expression, look at equation 21 of Kansabanik et al. 2022, ApJ, 932:110 
    Code adapated from P-AIRCARS (https://github.com/devojyoti96/P-AIRCARS/blob/quartical/paircarstools/basic_func.py#L1193)
    Parameters
    ----------
    caltable : str
	    Name of the full Jones QuartiCal caltable
    inplace : bool
	    Overwrite the input caltable (if not, a new caltable will be written)
    Returns
    -------
    str
	    New caltable name
    '''
    gains = xds_from_zarr(caltable+"::"+soltype)
    axis_names=gains[0].attrs['GAIN_AXES']
    gain_data=gains[0].gains.to_numpy()
    gain_flag=gains[0].gain_flags.to_numpy()
    gain_flag=gain_flag.astype('bool')
    for i in range(gain_data.shape[-1]):
	    gain_data[...,i][gain_flag]=np.nan
    ### shape of gain is (gain_time, gain_freq, antenna, direction, correlation)
    ### we break the 4 correlations into 2 x 2
    gain_data=gain_data.reshape(gain_data.shape[0],gain_data.shape[1],gain_data.shape[2],gain_data.shape[3],2,2)
    for t in range(gain_data.shape[0]):
	    for f in range(gain_data.shape[1]):
		    for d in range(gain_data.shape[3]):
			    g=gain_data[t,f,:,d,...]
			    if np.abs(np.nansum(g))==0:
				    pass
			    else:
				    gH=np.transpose(g.conj(),axes=((0,2,1)))
				    gH_dot_g=np.matmul(gH,g)
				    gH_dot_g_sum=np.nansum(gH_dot_g,axis=0)
				    gH_dot_g_sum_inv=np.linalg.inv(gH_dot_g_sum)
				    gH_sum=np.nansum(gH,axis=0)
				    X=np.linalg.inv(np.matmul(gH_dot_g_sum_inv,gH_sum))  # Poldistortion matrix
				    g_cor=np.matmul(g,np.linalg.inv(X))
				    gain_data[t,f,:,d,...]=g_cor		
    gain_data_cor=gain_data.reshape(gain_data.shape[0],gain_data.shape[1],gain_data.shape[2],gain_data.shape[3],4)
    gains[0].update({'gains': (axis_names,gain_data_cor)})
    term_name=soltype
    if inplace==False:
        if outcal is None:
	        outcal=caltable+'.poldist'
    else:
	    os.system('rm -rf '+caltable+'*')
	    outcal=caltable
    output_path = f"{outcal}{'::' + soltype}"
    write_xds_list = xds_to_zarr(gains, output_path)
    dask.compute(write_xds_list)
    return outcal

def remove_quartical_flags(caltable,soltype='D',inplace=False,outcal=None):
    data=xds_from_zarr(caltable+"::"+soltype)
    axis_names=data[0].attrs['GAIN_AXES']
    gains=np.asarray(data[0].gains)
    #### shape= (gain_time, gain_freq, antenna, direction, correlation)  for leakage solutions
    gain_flags=np.asarray(data[0].gain_flags)
    pos=np.where(np.isnan(gains))
    gains[pos]=0.0
    gain_flags[(pos[0],pos[1],pos[2],pos[3])]=0
    
    data[0].update({'gains': (axis_names,gains)})
    data[0].update({'gain_flags': (axis_names[:-1],gain_flags)})
    
    if inplace==False:
        if outcal is None:
            outcal=caltable+'.unflagged'
    else:
        os.system('rm -rf '+caltable)
        outcal=caltable
    output_path = f"{outcal}{'::' + soltype}"
    write_xds_list = xds_to_zarr(data, output_path)
    dask.compute(write_xds_list)
    return outcal
    

   
