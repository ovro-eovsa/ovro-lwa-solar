import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from functools import partial
from suncasa import dspec
from astropy.coordinates import get_sun, EarthLocation,SkyCoord, AltAz
from astropy.time import Time
from .primary_beam import jones_beam as beam
import h5py,os
from scipy.interpolate import interpn
import matplotlib.colors as colors
import lmfit,logging

def compute_primary_beam_from_beamfiles(freqs,tims):
    '''
    :param tims: A astropy Time object. Can have multiple times
    :param freq: list/array of frequencies in MHz
    
    This function returns the I,Q,U,V factors, normalised to their
    corresponding I values. Thus we get 1,Q/I,U/I, V/I. The shape 
    is (num_freqs,4,num_times)
    '''
    observing_location=EarthLocation.of_site('ovro')
    solar_loc=get_sun(tims)
    frame = AltAz(location=observing_location, obstime=tims)
    azel=solar_loc.transform_to(frame)
    az=azel.az.value
    alt=azel.alt.value

    num_tims=len(alt)
    num_freqs=freqs.size
    factors=np.zeros((num_freqs,num_tims,4))
    for j,freq1 in enumerate(freqs):
        if j==0:
            beamfac=beam(freq=freq1)
            beamfac.read_beam_file()    
        else:
            beamfac.freq=freq1
   
        beamfac.srcjones(az=az,el=alt)
        
        for i in range(num_tims):
            pol_fac=beamfac.get_source_pol_factors(np.squeeze(beamfac.jones_matrices)[i,:,:])
            factors[j,i,0]=np.real(0.5*(pol_fac[0,0]+pol_fac[1,1]))
            factors[j,i,1]=np.real(0.5*(pol_fac[0,0]-pol_fac[1,1]))
            factors[j,i,2]=np.real(0.5*(pol_fac[0,1]+pol_fac[1,0]))
            factors[j,i,3]=np.real(-1j*0.5*(pol_fac[0,1]-pol_fac[1,0]))
    return np.swapaxes(factors/np.expand_dims(factors[:,:,0],axis=2),1,2) ## all factors are in respect to I value

def rotate_UV(params,U,V,I=None,Umodel=None,Vmodel=None,subtract_mean_leak=True,return_corrected=False):
    '''
    This function rotates the Stokes vector in the UV plane by an angle theta. Theta, whe positive,
    implies rotation is in counterclockwise direction.
    :param params: This can be a float/lmfit parameter
    :param U: Observed U
    :param V: observed V
    :param I: observed I. Observed I is assumed to be the true I as well. 
                            Required if, return_corrected is False
    :param Umodel: Umodel is the predicted U due to the primary beam leakage
                   of an unpolarised source. Required if, return_corrected is False
    :param Vmodel: Vmodel is the predicted U due to the primary beam leakage
                   of an unpolarised source. Required if, return_corrected is False
    :param subtract_mean_leak: If True, a mean leakage will be subtracted, before
                                enforcing the zero U condition. The mean_leakage
                                is interpreted as a constant error in the primary 
                                beam model. Default is True
    :param return_corrected: If True, the corrected U and V will be returned. By,
                                corrected, we just mean the rotated U and V. Default
                                is False
    if params is a float/list/ndarray, this function will return the sum of squared residuals
    if not, it returns array of residuals. Lmfit uses the second option, whereas scipy.minimize
    uses the first option.
    '''
    theta=params
    if not isinstance(theta,float):
        try:
            theta=params['theta'].value
        except IndexError:
            pass
    Ucor=U*np.cos(theta)+V*np.sin(theta)
    Vcor=V*np.cos(theta)-U*np.sin(theta)
    if return_corrected:
        return Ucor,Vcor
    if subtract_mean_leak:
        mean_leak=np.nanmean((Ucor-Umodel)/I)
    else:
        mean_leak=0.0
    if isinstance(params,np.ndarray) or isinstance(params,list) or isinstance(params,float):
        sum1=np.nansum((Ucor-mean_leak*I-Umodel)**2)  ### This difference is coming from the way
                                                      ### lmfit uses the minimising function and 
                                                      ### how the scipy.minimize uses it.
    else:
        sum1=(Ucor-mean_leak*I-Umodel)
    return (sum1)
    


def robust_line_fit(data):
    '''
    This function implements a robust line fit method. It flags all points outside the 3sigma
    and then recomputes the fit. It will try this for 5 times. X-axis: number of points
    Y-axis : data
    '''
    num_trials=5
    data=data.squeeze()
    for i in range(num_trials):
        num_points=data.size
        if num_points==0:
            return np.nan*np.ones(2)
        x=np.arange(0,num_points)
        poly=np.polyfit(x,data,deg=1)
        res=data-np.poly1d(poly)(x)
        std=np.std(res)
        pos=np.where(np.abs(res)<3*std)
        data=data[pos]
        x=x[pos]
    return poly


def get_primary_beam(freqs,times,freq_sep=1,tim_sep=300,outfile="primary_beam.hdf5",overwrite=False):
    '''
    This function is responsible for producing the primary beam for all times and frequencies the user
    needs. However since this can be quite compute intensive, this function computes the I,Q,U,V beam
    at some time-frequency grid and then interpolates the I,Q,U,V beams for all times and frequencies.
    The time-frequency grid over which the computation occurs is determined by freq_sep and time_sep.
    The default freq_sep and time_sep are 1MHz and 300s respectively. The grid-computed beam is written
    into the outfile. If outfile does not exist, or user wants to overwrite it, then only the file will be
    computed again.
    
    :param freqs: Frequencies over which the user wants I,Q,U,V beams. in MHz
    :param times: Times over which the user wants I,Q,U,V beams. Astropy time objects.
    :param freq_sep: Frequency separation of grid in MHz. Default is 1 MHz
    :param time_sep: Time separation of grid in s. Default is 300s
    :param outfile: the computed beam will be writted here. Note that this is the computed beam, NOT the
                    interpolated beam. Interpolated beam is returned to user, not stored on disk.
    :param overwrite: Overwrite the outfile, if True.
    
    :return interpolated_beam
    '''
    min_freq=min(freqs) ##MHz
    max_freq=max(freqs) ## MHz
    freqs_to_compute_beam=np.arange(min_freq,max_freq,freq_sep)
    if  freqs_to_compute_beam[-1]!=max_freq:
        freqs_to_compute_beam=np.append(freqs_to_compute_beam,max_freq)
    
    mjd_times=times.mjd
    min_tim=min(mjd_times)
    max_tim=max(mjd_times)
    times_to_compute_beam=np.arange(min_tim,max_tim,tim_sep/86400)
    if times_to_compute_beam[-1]!=max_tim:
        times_to_compute_beam=np.append(times_to_compute_beam,max_tim)
   
    if not os.path.isfile(outfile) or overwrite:
        beam=compute_primary_beam_from_beamfiles(freqs_to_compute_beam,Time(times_to_compute_beam,\
                                                    format='mjd',scale='utc'))
        logging.debug("Primary beam successfully computed.")
        with h5py.File(outfile,"w") as  hf:
            hf.create_dataset("freqs_MHz",data=freqs_to_compute_beam)
            hf.create_dataset("times_mjd",data=times_to_compute_beam)
            hf.create_dataset("beam",data=beam)
            hf.attrs['freq_unit']='MHz'
            hf.attrs['time_unit']='mjd'
            hf.attrs['beam0']='freq'
            hf.attrs['beam1']='stokes'
            hf.attrs['beam2']='time'
            hf.attrs['stokes']='IQUV'
    else:
        with h5py.File(outfile,'r') as hf:
            beam=np.array(hf['beam'])
            times_to_compute_beam=np.array(hf['times_mjd'])
            freqs_to_compute_beam=np.array(hf['freqs_MHz'])
    
    points=(freqs_to_compute_beam,times_to_compute_beam)
    

    Freqs_MHz,MJD_times=np.meshgrid(mjd_times,freqs,indexing='xy')
    compute_points=(MJD_times,Freqs_MHz)
    interpolated_beam=[None]*4
    for i in range(1,4):
        interpolated_beam[i]=interpn(points,beam[:,i,:],compute_points,bounds_error=False,fill_value=np.nan)
        if i==1:
            shape=interpolated_beam[i].shape
            interpolated_beam[0]=np.ones(shape)
    interpolated_beam=np.array(interpolated_beam)
    return interpolated_beam


def get_stokes_data(filename,time_avg=1,freq_avg=1):
    '''
    This function reads the beamfile with provided time_avg and freq_avg.
    '''
    stokes=['I','Q','U','V']
    for j,pol in enumerate(stokes):
        ds=dspec.Dspec()
        ds.read([filename],source='lwa',timebin=time_avg,freqbin=freq_avg,stokes=pol)
        data=np.squeeze(ds.data)
        if j==0:
            freqs=ds.freq_axis  ###  in Hz
            times=ds.time_axis
            shape=data.shape
            num_times=shape[1]
            num_freqs=shape[0]
            stokes_data=np.zeros((4,num_freqs,num_times))
        stokes_data[j,:,:]=data
    return stokes_data,freqs,times
    
def determine_crosshand_phase(stokes_data,primary_beam,freqs):
    '''
    This method determines the crosshad phase by minimising the Stokes U from the Sun.
    It tries several minimisation techniques in decreasing order of priority/compute cost.
    It also tries to break degeneracies as well. However, this function has multiple parameters
    hardcoded, for lack of a better way. The minimisation can fail easily, and the user is advised
    to work through the code to improve it or tune the code according to their needs as necessary.
    :param stokes_data: The stokes data from the beam.
    :param primary_beam: Primary beam. It should have the same format as that returned by
                        the function get_primary_beam. The time and frequency resolution of
                        of the stokes_data and primary_beam should be exactly same.
    :param freqs: Frequencies in MHz corresponding to the data
    :return crosshand_phase for each frequency. 
    '''
    shape=stokes_data.shape
    num_freqs=shape[1]
    
    theta=np.zeros(num_freqs)
    
    max_nfev=1000
    
    for i in range(num_freqs):
          
        Umodel=primary_beam[2,i,:]*stokes_data[0,i,:]
        Vmodel=primary_beam[3,i,:]*stokes_data[0,i,:]
        
        red_chi=1000
        
        res1=minimize(rotate_UV,0,args=(stokes_data[2,i,:],stokes_data[3,i,:],\
                        stokes_data[0,i,:],Umodel,Vmodel,True),method='Nelder-Mead',\
                        bounds=[[-3.14159,3.14159]])
        if res1.success:
            solved_theta=res1.x
        red_chi=res1.fun/(Umodel.size-1)
        print (i,red_chi)
        if red_chi>0.001:
            red_chi=1000
        else:
            red_chi=1e-5
        
        
        
        max_iter=3
        iter_num=0
        methods=['Nelder','basinhopping','basinhopping','basinhopping','basinhopping']
        while iter_num<max_iter and red_chi>1e-4:
            method=methods[iter_num]
        
            fit_params = lmfit.Parameters()
            fit_params.add_many(('theta',0, True, -3.14159, 3.14159, None, None))
            
            if method=='Nelder':
	            fit_kws = {'max_nfev': max_nfev, 'tol': 0.01}
	            mini = lmfit.Minimizer(rotate_UV,\
			            fit_params, fcn_args=(stokes_data[2,i,:],stokes_data[3,i,:]),\
                        fcn_kws={'I': stokes_data[0,i,:],'Umodel': Umodel, 'Vmodel': Vmodel},\
                        nan_policy='omit',max_nfev=max_nfev)
            else:
	            fit_kws={'niter':50,'T':90.0, 'stepsize':0.8+iter_num*0.2, 'interval':25,\
	                         'minimizer_kwargs':{'method':'Nelder-Mead'}}
	            mini = lmfit.Minimizer(rotate_UV,\
			            fit_params, fcn_args=(stokes_data[2,i,:],stokes_data[3,i,:]),\
                        fcn_kws={'I': stokes_data[0,i,:],'Umodel': Umodel, 'Vmodel': Vmodel},\
                        nan_policy='omit')
        
            res1 = mini.minimize(method=method, **fit_kws)
            red_chi=res1.redchi
            if res1.nfev<10 or abs(res1.params['theta'].value-res1.init_vals[0])<1e-6:
                red_chi=1000
            iter_num+=1

            del fit_params
            #print (lmfit.fit_report(res1, show_correl=True))
        
        #
       
        iter_num=0
        
        while iter_num<max_iter and red_chi>1e-4:
            method=methods[iter_num]
        
            fit_params = lmfit.Parameters()
            fit_params.add_many(('theta',0, True, -3.14159, 3.14159, None, None))
            
            if method=='Nelder':
	            fit_kws = {'max_nfev': max_nfev, 'tol': 0.01}
	            mini = lmfit.Minimizer(rotate_UV,\
			            fit_params, fcn_args=(stokes_data[2,i,:],stokes_data[3,i,:]),\
                        fcn_kws={'I': stokes_data[0,i,:],'Umodel': Umodel, \
                        'Vmodel': Vmodel,'subtract_mean_leak':False},\
                        nan_policy='omit',max_nfev=max_nfev)
            else:
	            fit_kws={'niter':50,'T':90.0, 'stepsize':0.8+iter_num*0.2, 'interval':25, 'minimizer_kwargs':{'method':'Nelder-Mead'}}
	            mini = lmfit.Minimizer(rotate_UV,\
			            fit_params, fcn_args=(stokes_data[2,i,:],stokes_data[3,i,:]),\
                        fcn_kws={'I': stokes_data[0,i,:],'Umodel': Umodel, \
                        'Vmodel': Vmodel,'subtract_mean_leak':False},\
                        nan_policy='omit')
        
            res2 = mini.minimize(method=method, **fit_kws)
            red_chi=res2.redchi
            if res2.nfev<10 or abs(res2.params['theta'].value-res2.init_vals[0])<1e-6:
                red_chi=1000
            iter_num+=1
            del fit_params
        
        if res1.success and iter_num==0: ### iter_num was reinitialised to zero before the second round. It did not enter res2 mnimization
            res=res1
        elif not res1.success and res2.success:
            res=res2
        else:
            Udata_corrected1,Vdata_corrected1=rotate_UV(res1.params['theta'].value,stokes_data[2,i,:],stokes_data[3,i,:],\
                                                            stokes_data[0,i,:],Umodel,Vmodel,return_corrected=True)
            mean_leak=np.nanmean((Udata_corrected1-Umodel)/stokes_data[0,i,:])
            resU=Udata_corrected1-mean_leak*stokes_data[0,i,:]-Umodel
            corr1=abs(np.corrcoef(resU,Vdata_corrected1)[0,1])
                
            Udata_corrected2,Vdata_corrected2=rotate_UV(res2.params['theta'].value,stokes_data[2,i,:],stokes_data[3,i,:],\
                                                            stokes_data[0,i,:],Umodel,Vmodel,return_corrected=True)
            mean_leak=np.nanmean((Udata_corrected2-Umodel)/stokes_data[0,i,:])
            resU=Udata_corrected2-mean_leak*stokes_data[0,i,:]-Umodel
            corr2=abs(np.corrcoef(resU,Vdata_corrected2)[0,1])

            if corr1>corr2:
                res=res2

            else:
                res=res1
            solved_theta=res.params['theta'].value
                
        solutions=np.zeros(2)
        residual=np.zeros(2)
        
        
        for j,angle in enumerate([solved_theta,solved_theta+3.14159/2]):  ### 3.14159: pi
            solutions[j]=angle
            residual[j]= np.sum(np.abs(rotate_UV(angle,stokes_data[2,i,:],stokes_data[3,i,:], I=stokes_data[0,i,:],\
                                    Umodel=Umodel,Vmodel=Vmodel)))
        
       
        max_ind_original=np.nanargmax(np.abs(stokes_data[3,i,:]))
        max_val_original=stokes_data[3,i,max_ind_original]
        

        theta[i]=solutions[np.argmin(residual)]
        Udata_corrected,Vdata_corrected=rotate_UV(theta[i],stokes_data[2,i,:],stokes_data[3,i,:],return_corrected=True)
#        ind=np.nanargmax(np.abs(Vdata_corrected))
        max_val_corrected=Vdata_corrected[max_ind_original]

        
        
        if max_val_corrected*max_val_original<0:
            theta[i]-=np.pi


        if theta[i]>3.14159:
            theta[i]-=2*3.14159
        elif theta[i]<-3.14159:
            theta[i]+=2*3.14159
    
    return theta

def align_theta_with_freq(theta,freqs):
    '''
    This function tries to break the pi/2pi degeneracy in the solved crosshand phase.
    It uses  the range of frequencies between 52-58 MHz as a control range. This is
    almost the central part of the "good" observing band. The function tries the minimize
    the difference between the solved phase and the median part of this frequency range,
    by using the available degeneracies.
    :param theta: Solved crosshand theta in radians
    :param freqs: Frequency in MHz
    '''
    pos=np.where((freqs>52) & (freqs<58))[0]
    med_theta=np.nanmedian(theta[pos])

    ind=np.argmin(np.abs(theta[pos]-med_theta))+pos[0]
    
    num_freqs=freqs.size
    
    aligned_theta=np.zeros_like(theta)
    aligned_theta[...]=theta[...]
    
    for i,j in zip(range(ind-1,-1,-1),range(ind+1,num_freqs,1)):
        additives=np.array([0,1,-1,2,-2])  ####  np.pi degeneracy: sign of V, standard 2*np.pi degeneracy
        
        diff=np.array([abs(aligned_theta[i]+add*np.pi-med_theta) for add in additives])
        ind=np.nanargmin(diff)
        aligned_theta[i]+=additives[ind]*np.pi
        
        diff=np.array([abs(aligned_theta[j]+add*np.pi-med_theta) for add in additives])
        ind=np.nanargmin(diff)
        aligned_theta[j]+=additives[ind]*np.pi
    aligned_theta[aligned_theta<-np.pi]+=2*np.pi
    aligned_theta[aligned_theta>np.pi]-=2*np.pi
    logging.debug("Crosshand phase have been successfully aligned.")
    return aligned_theta
        

def correct_crosshand_phase(stokes_data,primary_beam,freqs,\
                            crosshand_theta=None,outfile=None, \
                            doplot=False,figname='crosshand_phase_with_freq.png'):
    '''
    This function corrects the crosshand phase of the beam data. If crosshand_theta
    is not provided, then it is solved. If provided, it has to be a numpy ndarray
    If outfile is provided, then the solved crosshand_phase is written down, along
    with the corresponding frequency. If not provided, the information is written in
    a file name, naming of which is discussed below. If doplot is True, the crosshand
    phase vs frequency is plotted and saved as figname is provided. Outfile, when not 
    provided, is same as figname, except the suffix like ".png",".pdf" etc. replaced by 
    ".txt". Crosshand_theta which could not be solved are replaced by 1000, when writing
    :param stokes_data: The stokes data from the beam.
    :param primary_beam: Primary beam. It should have the same format as that returned by
                        the function get_primary_beam. The time and frequency resolution of
                        of the stokes_data and primary_beam should be exactly same.
    :param freqs: Frequencies in MHz corresponding to the data
    :param crosshand_theta: can be np.ndarray/None
    '''
    if not isinstance(crosshand_theta,np.ndarray):
        crosshand_theta=determine_crosshand_phase(stokes_data,primary_beam,freqs)
        logging.debug("Crosshand phase have been successfully determined.")
        if doplot:
            plt.plot(freqs,crosshand_theta,'o-')
            plt.savefig(figname)
            plt.close()
        if not outfile:
            pieces=figname.split('.')
            if num_pieces==1:
                outfile=figname+".png"
            else:
                outfile='.'.join(pieces[:-1])+".txt"
            
        pos=np.where(np.isnan(crosshand_theta)==True)
        crosshand_theta[pos]=1e3
        np.savetxt(outfile,np.array([freqs,crosshand_theta]).T)
        crosshand_theta[pos]=np.nan
            
    
    corrected_stokes_data=np.zeros_like(stokes_data)
    corrected_stokes_data[0,...]=stokes_data[0,...]
    corrected_stokes_data[1,...]=stokes_data[1,...]
   
    cos_theta=np.expand_dims(np.cos(crosshand_theta),axis=1)
    sin_theta=np.expand_dims(np.sin(crosshand_theta),axis=1)
    
    corrected_stokes_data[2,...]=stokes_data[2,...]*cos_theta+stokes_data[3,...]*sin_theta
    corrected_stokes_data[3,...]=-stokes_data[2,...]*sin_theta+stokes_data[3,...]*cos_theta
    logging.debug("Crosshand phase have been successfully corrected.")
    return corrected_stokes_data 
    
def determine_stokesI_leakage(pol_frac,QU_only=False):
    '''
    This function determines the leakage from Stokes I by fitiing a straight line to
    the polarisation fraction of each Stokes parameter. This is done on each frequency.
    :param pol_frac: Polarisation fraction of Q,U and V.
    :param QU_only: Leakage subtraction will be done only on Stokes V.
    :return a npy ndarray of polyfit results. Shape: num_stokes x num_freqs x 2 (numpy polyfit convention)
    '''
    num_freqs=pol_frac.shape[1]
    poly=np.zeros((4,num_freqs,2))
    poly[0,...]=0
    if QU_only:
        max_pol_ind=2
    else:
        max_pol_ind=3
    #print (pol_frac[1,0,:])
    for j in range(1,max_pol_ind+1):
        for i in range(num_freqs):
            pos=np.where(np.isnan(pol_frac[j,i,:])==False)
            if pos[0].size!=0:
                poly[j,i,:]=np.nan
            poly[j,i,:]=robust_line_fit(pol_frac[j,i,pos])
    return poly
            
def correct_leakage_from_stokesI(stokes_data,primary_beam,freqs=None,polynomial=None,\
                                    outfile='stokesI_leakage.npy',QU_only=False, \
                                    doplot=False,figname=None):
    '''
    This function corrects the leakage from Stokes I. 
    :param stokes_data: The stokes data from the beam.
    :param primary_beam: Primary beam. It should have the same format as that returned by
                        the function get_primary_beam. The time and frequency resolution of
                        of the stokes_data and primary_beam should be exactly same.
    :param freqs: Frequency in MHz. Is used only for plotting
    :param polynomial: If provided, will be used for correcting. Otherwise will be solved for
                        If provided, it should be of type np.ndarray
    :param outfile: The determined leakage parameters will be saved into this file.
    :param QU_only: If True, leakage correction will be done only for Q and U
    :param doplot: If True, the variation of leakage parameters with frequency will be plotted
    :param figname: If provided, the figure will be saved as figname.
    :return stokes_corrected_data. Note the primary_beam effect is still present.
    '''
    frac_pol=np.zeros_like(stokes_data)
    if QU_only:
        max_pol_ind=2
    else:
        max_pol_ind=3
    
    for i in range(1,max_pol_ind+1):
        frac_pol[i,...]=stokes_data[i,:,:]/stokes_data[0,:,:]-primary_beam[i,:,:]
        
    if not isinstance(polynomial,np.ndarray):
        poly=determine_stokesI_leakage(frac_pol,QU_only=QU_only)
        logging.debug("Leakage from Stokes I to other Stokes parameters have been successfully determined.")
        np.save(outfile,poly)
        if doplot:
            fig,ax=plt.subplots(nrows=1,ncols=2)
            colors=['r','b','k','c']
            stokes=['Q','U','V']
            for pol,color,label in zip(range(1,4),colors,stokes):
                ax[0].plot(freqs,poly[pol,:,0],label=label,color=color)
                ax[1].plot(freqs,poly[pol,:,1],label=label,color=color)
            plt.legend()
            if figname:
                plt.savefig(figname)
            plt.show()
            
    
    stokes_corrected=np.zeros_like(stokes_data)
    
    shape=stokes_data.shape
    num_freqs=shape[1]
    num_tims=shape[2]
    
    tims=np.arange(0,num_tims,1)
    
    stokes_corrected[0,...]=stokes_data[0,...]
    for j in range(1,max_pol_ind+1):
        for i in range(num_freqs):
            stokes_corrected[j,i,:]=(frac_pol[j,i,:]-np.poly1d(poly[j,i,:])(tims)+primary_beam[j,i,:])*\
                                        stokes_data[0,i,:]
    if QU_only:
        stokes_corrected[3,...]=stokes_data[3,...]
    logging.debug("Leakage from Stokes I to other Stokes parameters have been successfully corrected.")
    return stokes_corrected
    

