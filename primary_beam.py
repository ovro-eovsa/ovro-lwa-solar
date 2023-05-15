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
        self.freq=None
        self.msfile=msfile
        
    def ctrl_freq(self):
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
       
    def srcIQUV(self,az,el):
        """Compute beam scaling factor
        Args:
            az: azimuth in degrees
            el: elevation in degrees

        Returns: [I,Q,U,V] flux factors, where for an unpolarized source [I,Q,U,V] = [1,0,0,0]

        """
        try:
            if self.beam_file_path is not None:
                # index where grid equals source az el values
                index = knn_search(np.array([ [az], [el] ]), self.azelgrid.reshape(2,self.gridsize*self.gridsize))
                Ifctr = self.Ibeam.reshape(self.gridsize*self.gridsize)[index]
                Qfctr = self.Qbeam.reshape(self.gridsize*self.gridsize)[index]
                Ufctr = self.Ubeam.reshape(self.gridsize*self.gridsize)[index]
                Vfctr = self.Vbeam.reshape(self.gridsize*self.gridsize)[index]
                return np.array([[np.sqrt(Ifctr),0],[0,np.sqrt(Ifctr)]])
            else:
                raise RuntimeError
        except:
            Ifctr=math.sin(el*np.pi/180)**1.6
            return np.array([[np.sqrt(Ifctr),0],[0,np.sqrt(Ifctr)]])

                
                
class jones_beam:
    """
    For loading and returning LWA dipole beam values (derived from DW beam simulations) on the ASTM.
    Last edit: 11 September 2020
    """
    
    def __init__(self,msfile=None,beam_file_path='/opt/beam',freq=None):
        try:
            self.beam_file_path=beam_file_path
        except:
            self._beam_file_path=None
        self.beamIQUV=None
        self.Ibeam=None
        self.Ubeam=None
        self.Qbeam=None
        self.Vbeam=None
        self.freq=None
        self.msfile=msfile
        
    def ctrl_freq(self):
        msmd=msmetadata()
        msmd.open(self.msfile)
        chan_freqs = msmd.chanfreqs(0)
        msmd.done()
        self.freq = 0.5 * (chan_freqs[0] + chan_freqs[-1]) * 1e-6
    
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
        
    def read_beam_file(self):
        try:
            self.ctrl_freq()
            self.beamjonesfile = self.beam_file_path+'/beamLudwig3rd.npz'
            if os.path.exists(self.beamjonesfile):
                self.beamjones = np.load(self.beamjonesfile)
                self.Co = self.beamjones['cofull']
                self.Cx = self.beamjones['cxfull']
                self.Corot90 = self.beamjones['cofull_rot90']
                self.Cxnrot90 = self.beamjones['cxfull_nrot90']
                self.l = self.beamjones['lfull']
                self.m = self.beamjones['mfull']
                self.freqs = self.beamjones['frqvals']#self.beamjones['freqfull']
                num_freqs=self.freqs.size
                num_l=self.l.size
                self.freqs=np.repeat(self.freqs,num_l) #### This works only if Co shape is (num_freqs x num_l)
                self.l=np.tile(self.l,num_freqs)
                self.m=np.tile(self.m,num_freqs)
            else:
                raise RuntimeError    
        except:
            logging.warning("Beam file does not exist in give path."+\
                    "Switching to analytical beam.")   
            

    def srcjones(self,l,m):
        """Compute beam scaling factor
        Args:
            (l,m) coordinates

        Returns: Jones matrix at coordinates (l,m)

        """
        try:
            if self.beam_file_path is not None:
                coval = gd( (self.l, self.m, self.freqs), \
                        self.Co.ravel(), (l, m, self.freq), method='linear')
                cxval = gd( (self.l, self.m, self.freqs), \
                        self.Cx.ravel(), (l, m, self.freq), method='linear')
                corot90val  = gd( (self.l, self.m, self.freqs), \
                          self.Corot90.ravel(), (l, m, self.freq), method='linear')
                cxnrot90val = gd( (self.l, self.m, self.freqs), \
                          self.Cxnrot90.ravel(), (l, m, self.freq), method='linear')
                Jonesmat = np.array([ [coval,       cxval     ], 
                                  [cxnrot90val, corot90val] ])
                return Jonesmat
            else:
                raise RuntimeError
        except OSError:
            return np.nan
