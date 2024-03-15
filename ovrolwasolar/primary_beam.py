import numpy as np
import glob, logging, math, os
from casatools import msmetadata
from scipy.interpolate import griddata as gd

def knn_search(arr,grid):
    '''
    Find 'nearest neighbor' of array of points in multi-dimensional grid
    
    Source: glowingpython.blogspot.com/2012/04/k-nearest-neighbor-search.html
    '''
    gridsize = grid.shape[1]
    dists    = np.sqrt(((grid - arr[:,:gridsize])**2.).sum(axis=0))
    return np.argsort(dists)[0]
    
def primary_beam_correction_val(pol,jones_matrix):
    if pol=='XX':
        return jones_matrix[0,0]**2
    if pol=='YY':
        return jones_matrix[1,1]**2
                
    if pol=='I':
        return 0.5*(jones_matrix[1,1]**2+jones_matrix[0,0]**2)


class analytic_beam():
    def __init__(self,msfile=None,beam_file_path='/opt/beam/',freq=None):
        return
    def srcjones(self,az,el): ### az, el in degrees
        num_sources=np.size(el)
        self.jones_matrices=np.zeros((num_sources,2,2))
        for i in range(num_sources):
            self.jones_matrices[i,:,:]=np.sqrt(np.sin(el[i]*np.pi/180)**1.6*np.identity(2))
        return 
    
    @staticmethod    
    def get_source_pol_factors(jones_matrix):
        return jones_matrix**2
            
class woody_beam():
    def __init__(self,msfile=None,beam_file_path='/opt/beam/',freq=None):
        try:
            self.beam_file_path=beam_file_path
        except:
            self._beam_file_path=None
        self.beamIQUV=None
        self.Ibeam=None
        self.Ubeam=None
        self.Qbeam=None
        self.Vbeam=None
        self.freq=freq
        if self.freq is None:
            self.msfile=msfile
        
    def ctrl_freq(self):
        if self.freq is not None:
            return
        msmd=msmetadata()
        msmd.open(self.msfile)
        self.freq = msmd.meanfreq(0)*1e-6
        msmd.done()
        
    def get_beam_file(self):
        all_beam_files=glob.glob(self.beam_file_path+'/beamIQUV_*.npz')
        all_freqs=np.array([float(file1.split('/')[-1].split('_')[-1].split('.npz')[0]) for file1 in all_beam_files])
        self.ctrl_freq()
        diff=abs(all_freqs-self.freq)
        ind=np.argsort(diff)[0]
        return all_beam_files[ind]
    
    def read_beam_file(self):
        try:
            self.azelgrid = np.load(self.beam_file_path+'/azelgrid.npy')
            self.gridsize = self.azelgrid.shape[-1]
            # load 4096x4096 grid of IQUV values, for given msfile CRFREQ
            self.beamIQUVfile = self.get_beam_file()#self.beam_file_path+'/beamIQUV_'+str(self.freq)+'.npz'
            if os.path.exists(self.beamIQUVfile):
                self.beamIQUV = np.load(self.beamIQUVfile)
                self.Ibeam    = self.beamIQUV['I']
                self.Qbeam    = self.beamIQUV['Q']
                self.Ubeam    = self.beamIQUV['U']
                self.Vbeam    = self.beamIQUV['V']
                logging.debug('Beam files read successfully')
            else:
                raise RuntimeError    
        except OSError:
            logging.warning("Beam file does not exist in give path."+\
                    "Switching to analytical beam.")   
            self.beamIQUV = np.nan

    @property
    def beam_file_path(self):
        return self._beam_file_path
        
    @beam_file_path.setter
    def beam_file_path(self,value):
        files=glob.glob(value+"*.npy")
        if len(files)!=0:
            self._beam_file_path=value
        else:
            logging.warning("Beam file does not exist in give path."+\
                    "Switching to analytical beam.")
            self._beam_file_path=None
    
    @property
    def msfile(self):
        return self._msfile
    
    @msfile.setter
    def msfile(self,value):
        if os.path.isdir(value):
            self._msfile=value
        else:
            raise RuntimeError                              
       
    def srcjones(self,az,el):
        """
        The function name is srcjones to keep naming consistent with Jones_beam class.
        But this function returns I,Q,U,V beams.
        Compute beam scaling factor
        Args:
            az: azimuth in degrees
            el: elevation in degrees

        Returns: [I,Q,U,V] flux factors, where for an unpolarized source [I,Q,U,V] = [1,0,0,0]

        """
        num_sources=len(az)
        
        self.jones_matrices=np.zeros((num_sources,2,2),dtype='complex')
        self.read_beam_file()
        
        for i in range(num_sources):
            try:
                if self.beam_file_path is not None:
                    # index where grid equals source az el values
                    index = knn_search(np.array([ [az[i]], [el[i]] ]), self.azelgrid.reshape(2,self.gridsize*self.gridsize))
                    Ifctr = self.Ibeam.reshape(self.gridsize*self.gridsize)[index]
                    Qfctr = self.Qbeam.reshape(self.gridsize*self.gridsize)[index]
                    Ufctr = self.Ubeam.reshape(self.gridsize*self.gridsize)[index]
                    Vfctr = self.Vbeam.reshape(self.gridsize*self.gridsize)[index]
                    self.jones_matrices[i,:,:]=np.array([[Ifctr+Qfctr,Ufctr-1j*Vfctr],[Ufctr+1j*Vfctr,Ifctr-Qfctr]])
                else:
                    raise RuntimeError
            except:
                Ifctr=math.sin(el*np.pi/180)**1.6
                self.jones_matrices[i,:,:]= np.array([[Ifctr,0],[0,Ifctr]])
        return
    
    @staticmethod
    def get_source_pol_factors(jones_matrix):  ### in [[XX,XY],[YX,YY]] format
        '''
        I am assuming that the source is unpolarised. At these low frequencies this is a good assumption.
        Since the jones matrix in this class is essentially the source pol factors, just returning.
        '''
        
        return  jones_matrix    
                
                
class jones_beam:
    """
    For loading and returning LWA dipole beam values (derived from simulations made by Nivedita)
    """
    
    def __init__(self,msfile=None,beam_file_path='/data07/msurajit/primary_beam_files/',freq=None):
        try:
            self.beam_file_path=beam_file_path
        except:
            self._beam_file_path=None
        self.freq=freq
        if self.freq is None:
            self.msfile=msfile
        self.num_theta=181
        self.num_phi=361
        self.start_freq=10
        self.freq_step=1
        self.num_freqs=91
        self.num_header=10
        self.beam_files=['LWA_x_10to100.ffe','LWA_y_10to100.ffe']  ### assume order [X,Y]
        self.e_theta=[]
        self.e_phi=[]
        self.gain_theta=[]
        self.gain_phi=[]
        
        
    def ctrl_freq(self):
        if self.freq is not None:
            return
        msmd=msmetadata()
        msmd.open(self.msfile)
        self.freq = msmd.meanfreq(0) * 1e-6
        msmd.done()
        
    
    @property
    def beam_file_path(self):
        return self._beam_file_path
        
    @beam_file_path.setter
    def beam_file_path(self,value):
        if value and os.path.isdir(value):
            self._beam_file_path=value
        else:
            logging.warning("Beam file does not exist in give path."+\
                    "Switching to analytical beam.")
            self._beam_file_path=None
    
    @property
    def msfile(self):
        return self._msfile
    
    @msfile.setter
    def msfile(self,value):
        if os.path.isdir(value):
            self._msfile=value
        else:
            raise RuntimeError  
    
    def read_beam_file(self,datafile):  ### freq in MHz
        freq_index=(int(self.freq)-self.start_freq)
        tot_params=self.num_theta*self.num_phi
        

        tot_lines=(tot_params+self.num_header)*self.num_freqs
        header=(tot_params+self.num_header)*freq_index
        
  
        data=np.genfromtxt(datafile,skip_header=int(header),max_rows=int(tot_params))

        e_theta=data[:,2]+1j*data[:,3]
        e_phi=data[:,4]+1j*data[:,5]
        theta=data[:,0]
        phi=data[:,1]
        gain_total=10**(data[:,8]/10)
        gain_theta=10**(data[:,6]/10)
        gain_phi=10**(data[:,7]/10)
        
        
        e_theta=e_theta.reshape(self.num_phi,self.num_theta)
        e_phi=e_phi.reshape(self.num_phi,self.num_theta)
        gain_theta=gain_theta.reshape(self.num_phi,self.num_theta)
        gain_phi=gain_phi.reshape(self.num_phi,self.num_theta)
        gain_total=gain_total.reshape(self.num_phi,self.num_theta)

        theta=theta.reshape(self.num_phi,self.num_theta)
        theta=theta[:,:91].flatten()
    
        phi=phi.reshape(self.num_phi,self.num_theta)
        
        phi=phi[:,:91].flatten()
        
        
        e_theta=e_theta[:,:91].flatten()
        e_phi=e_phi[:,:91].flatten()
        gain_theta=gain_theta[:,:91].flatten()
        gain_phi=gain_phi[:,:91].flatten()
        gain_total=gain_total[:,:91].flatten()
       
    
        return  theta,phi,e_theta,e_phi, gain_theta, gain_phi, gain_total 
        
            

    def srcjones(self,az,el):
        """Compute beam scaling factor
        Args:
            (az,el) coordinates

        Returns: Jones matrix at coordinates (az,el)

        """
        
        print (az,el)
        el=90-el
        
        self.ctrl_freq()
        
        if len(self.gain_theta)<2:
            print (self.beam_file_path)
            if self.beam_file_path is not None:
                for file1 in self.beam_files:
                    datafile=self.beam_file_path+"/"+file1
                   
                    if os.path.isfile(datafile):
                        theta,phi,e_theta,e_phi,gain_theta,gain_phi,gain_total=self.read_beam_file(datafile)
                        self.e_theta.append(e_theta)
                        self.e_phi.append(e_phi)
                        self.gain_theta.append(gain_theta)
                        self.gain_phi.append(gain_phi)
                    else:
                        raise RuntimeError("Beam file does not exist. Switching to analytical beam")  
            
            else:
                raise RuntimeError("Beam file does not exist. Switching to analytical beam")
        
       
        num_sources=len(az)
        
        jones_matrices=np.zeros((num_sources,2,2),dtype='complex')
        
        max_e1=np.max(np.abs(np.array(e_theta)))
        max_e2=np.max(np.abs(np.array(e_phi)))
        max_e=max(max_e1,max_e2)
        
        #print (np.size(P),np.size(grid_el),np.shape(self.gain_theta[0]))
        sources_e_theta_x=gd((phi,theta), self.e_theta[0], (az,el), method='nearest')
        sources_e_theta_y=gd((phi,theta), self.e_theta[1], (az,el), method='nearest')
        sources_e_phi_x=gd((phi,theta), self.e_phi[0], (az,el), method='nearest')
        sources_e_phi_y=gd((phi,theta), self.e_phi[1], (az,el), method='nearest')
        
        for i in range(num_sources):
            jones_matrices[i,:,:]=[[sources_e_theta_x[i],sources_e_phi_x[i]],\
                                    [sources_e_theta_y[i],sources_e_phi_y[i]]] #### interchanging theta and phi
                                    						  ### as it seems phi values are larger
        
        self.jones_matrices=jones_matrices/max_e  ### normalising
        print (self.jones_matrices)
        
        return
        

    @staticmethod
    def get_source_pol_factors(jones_matrix):  ### in [[XX,XY],[YX,YY]] format
        '''
        I am assuming that the source is unpolarised. At these low frequencies this is a good assumption.
        '''
        J1=jones_matrix
        J2=np.zeros_like(J1)
        
        J2[0,0]=np.conj(J1[0,0])
        J2[0,1]=np.conj(J1[1,0])
        J2[1,0]=np.conj(J1[0,1])
        J2[1,1]=np.conj(J1[1,1])
        
        J3=np.matmul(J1,J2)  
        #J3=J3/np.sum(np.abs(J3))*2 #### check normalisation
        return  J3          
