import numpy as np
import math,os,logging
from casatasks import clearcal, ft
from casatools import table, measures, componentlist, msmetadata
from .primary_beam import analytic_beam as beam

from astropy.io import fits
from . import primary_beam, utils

def conv_deg(dec):
    if 's' in dec:
        dec = dec.split('s')[0]
    if 'm' in dec:
        dec, ss = dec.split('m')
        if ss == '':
            ss = '0'
    dd, mm = dec.split('d')
    if dd.startswith('-'):
        neg = True
    else:
        neg = False
    deg = float(dd) + float(mm) / 60 + float(ss) / 3600
    return '%fdeg' % deg

class model_generation():
    def __init__(self,vis=None,filename='calibrator_source_list.txt',pol='I,Q,U,V',separate_pol=True,model=True):
        '''
        vis: Name of MS
        min_beam_val: The minimum primary beam value, below which the source
                      will not be considered during calibration
        separate_pol: If True, separate polarisations will be analysed separately
               and the resultant model can be different. This parameter
               is only used if WSClean clean components are used for 
               generating the source model
        pol: Polarisations for which model is needed. Can be I,Q,U,V, I, XX,YY
        model: This is used only if WSClean components are used. This tells the code
               that we are generating a model (units Jy/pixel) and not an image units Jy/beam).
        '''
        self.vis=vis
        self.min_beam_val=0.01
        self.separate_pol=separate_pol
        self.pol=pol
        self.calfilepath = '/data07/msurajit/ovro-lwa-solar/defaults/'
        self.outpath=os.getcwd()+"/"
        self.polarisations=self.pol.split(',')
        self.num_pol=len(self.polarisations)
        self.filename=filename
        self.ref_freq=80.0
        self.output_freq=None
        self.includesun=False
        self.solar_flux=16000
        self.solar_alpha=2.2
        self.modelcl=None
        self.verbose=True
        self.overwrite=True
        self.predict=True
        self.model=model
        self.point_source_model_needed=True
        if (self.separate_pol==True and self.num_pol==1) or \
            (self.separate_pol==False and self.num_pol!=1):
            raise RuntimeError("The two keywords \'separate_pol\' and \'pol\' are inconsistent")
        
        
    @property
    def vis(self):
        return self._vis
    
    @vis.setter
    def vis(self,value):
        if os.path.isdir(value):
            self._vis=value
        else:
            logging.error("MSfile is not found.")
            raise RuntimeError("MSfile is not found")
    
    @property
    def calfilepath(self):
        return self._calfilepath    
    
    @calfilepath.setter
    def calfilepath(self,value):
        if os.path.exists(value):
            self._calfilepath=value
        else:
            self._calfilepath=None
            
    @property
    def filename(self):
        return self._filename
        
    @filename.setter
    def filename(self,value):
        if self.separate_pol==True:
            print (value)
            self._filename=[value+"."+pol_str for pol_str in self.polarisations]
        else:
            self._filename=[value]

    @property
    def pol(self):
        return self._pol
        
    @pol.setter
    def pol(self,value):
        temp=value.split(',')
        for i in temp:
            if i not in ['I','Q','U','V','XX','YY','XY','YX']:
                raise RuntimeError("Only XX,YY,XY,YX,I,Q,U,V recognised now.")
        self._pol=value
        

    def gen_model_file(self):
        """
        :param filename: output txt file that contains clean components for all visible strong calibration sources
            in wsclean format
        :param visibility: input visibility
        :return: N/A
        """


        srcs,az,el=self.get_risen_source_list()                  

        pb=beam(msfile=self.vis)
        pb.srcjones(np.array(az),np.array(el))
        
        self.file_handle = [open(self.filename[i], 'w') for i in range(self.num_pol)]
        
        s=0    
        for azev,elev in zip(az,el):
            matrix=pb.get_source_pol_factors(pb.jones_matrices[s,:,:])
            if matrix[0,0] > self.min_beam_val and matrix[1,1] > self.min_beam_val:
                for i in range(self.num_pol):
                    self.write_source_file(i,srcs[s]['label'], matrix, s)
            s+=1
            
        return    

    
    def write_source_file(self,current_pol_index,source_name, jones_matrix, source_num):  #### works only if logarithimicSI is false

        
        primary_beam_value=self.primary_beam_value(current_pol_index,jones_matrix)
        try:
            f1 = open(self.calfilepath + source_name+ ".txt", "r")
            j = 0

            while True:
                line = f1.readline()
                if not line:
                    break
                if source_num == 0 and j == 0:
                    self.file_handle[current_pol_index].write(line)
                elif j != 0:
                    try:
                        splitted = line.split(',')
                        I_flux = float(splitted[4])
                        
                        beam_corrected_I_flux = I_flux * primary_beam_value
                        splitted[4] = str(beam_corrected_I_flux)

                        for k, phrase in enumerate(splitted[5:]):
                            if k == 0:
                                splitted[5 + k] = '[' + str(float(phrase[1:]) * primary_beam_value)
                            else:
                                if phrase[-1] == ']':
                                    splitted[5 + k] = str(float(phrase[:-1]) * primary_beam_value) + ']'
                                    break
                                else:
                                    splitted[5 + k] = str(float(phrase) * primary_beam_value)
                        line1 = ','.join(splitted)
                        line1 = line1[:-1] + " "+ "\n"
                        if splitted[5 + k + 1] == 'false':
                            self.file_handle[current_pol_index].write(line1)
                        else:
                            raise RuntimeError("Function now works only if logarithmicSI is false")
                    except IndexError:
                        pass

                j += 1
        finally:
            f1.close()

    @staticmethod
    def primary_beam_value(current_pol_index,jones_matrix):  
        XX_factor=np.abs(jones_matrix[0,0])
        YY_factor=np.abs(jones_matrix[1,1])
        XY_factor=jones_matrix[0,1]
        I_factor=0.5*(XX_factor+YY_factor)
        Q_factor=0.5*(XX_factor-YY_factor)
        U_factor=np.real(XY_factor)
        V_factor=-np.imag(XY_factor)
        if current_pol_index==0:
            return I_factor
        elif current_pol_index==1:
            return Q_factor
        elif current_pol_index==2:
            return U_factor
        else:
            return V_factor
    

    def ctrl_freq(self):
        msmd=msmetadata()
        msmd.open(self.vis)
        self.avg_freq=msmd.meanfreq(0)*1e-6
        msmd.done()
        if not self.output_freq:
            self.output_freq=self.avg_freq
    
    def flux80_47(self,flux_hi, sp,jones_matrix):
        """
        Given a flux at 80 MHz and a sp_index, return the flux at 47 MHz.

        :param flux_hi: flux at the reference frequency
        :param sp: spectral index
        :param ref_freq: reference frequency in MHz
        :param output_freq: output frequency in MHz
        :return: flux caliculated at the output frequency
        """
        
        freq_I=flux_hi * 10 ** (sp * math.log(self.output_freq / self.ref_freq, 10))
        
        I_flux=self.primary_beam_value(0,jones_matrix)*freq_I
        Q_flux=self.primary_beam_value(1,jones_matrix)*freq_I
        U_flux=self.primary_beam_value(2,jones_matrix)*freq_I
        V_flux=self.primary_beam_value(3,jones_matrix)*freq_I
        print (I_flux,Q_flux,U_flux,V_flux)    
        return [I_flux,Q_flux, U_flux, V_flux]
    
    
    def get_risen_source_list(self):
        srcs = utils.get_strong_source_list()
        az=[]
        el=[]
        
        if self.includesun:
            srcs.append({'label': 'Sun', 'flux': str(solar_flux), 'alpha': solar_alpha,
                         'position': 'SUN'})
        
        tb=table()
        tb.open(self.vis)
        t0 = tb.getcell('TIME', 0)
        tb.close()
        
        me=measures()
        ovro = me.observatory('OVRO_MMA')
        time = me.epoch('UTC', '%fs' % t0)
        me.doframe(ovro)
        me.doframe(time)


        for s in range(len(srcs) - 1, -1, -1):
            coord = srcs[s]['position'].split()
            d0 = None
            if len(coord) == 1:
                d0 = me.direction(coord[0])
                d0_j2000 = me.measure(d0, 'J2000')
                srcs[s]['position'] = 'J2000 %frad %frad' % (d0_j2000['m0']['value'], d0_j2000['m1']['value'])
            elif len(coord) == 3:
                coord[2] = conv_deg(coord[2])
                d0 = me.direction(coord[0], coord[1], coord[2])
            else:
                raise Exception("Unknown direction")
            d = me.measure(d0, 'AZEL')
            elev = d['m1']['value']*180/np.pi
            azev=d['m0']['value']*180/np.pi

            if azev<0:
                azev=360+azev
            if elev<0:
                del srcs[s]
            elif np.sin(d['m0']['value'])**1.6<self.min_beam_val:
                del srcs[s]
            else:
                srcs[s]['el']=elev
                srcs[s]['az']=azev
        
        return srcs
     
    def point_source_model(self):
        srcs=self.get_risen_source_list()                  
        
        
        
        cl = componentlist()  
        cl.done()

        
        modelcl = self.vis.replace('.ms', '.cl')
        
         
        for s in srcs:
            pb=beam(msfile=self.vis)
            pb.srcjones(np.array([s['az']]),np.array([s['el']]))
            matrix=pb.get_source_pol_factors(pb.jones_matrices[0,:,:])

            if matrix[0,0] > self.min_beam_val and matrix[1,1] > self.min_beam_val:
                
                s['flux'] = self.flux80_47(float(s['flux']), s['alpha'], matrix) 
                
                cl.addcomponent(flux=s['flux'], dir=s['position'], index=s['alpha'], polarization='Stokes', 
                            spectrumtype='spectral index', freq='{0:f}MHz'.format(self.output_freq), label=s['label'])
            
                logging.debug(
                        "cl.addcomponent(flux=%s, dir='%s', index=%s, spectrumtype='spectral index', label='%s', polarization='Stokes')" % (
                            s['flux'], s['position'], s['alpha'], s['label']))
            
        
        
            
        if os.path.exists(modelcl) and self.overwrite:
            os.system('rm -rf ' + modelcl)
        cl.rename(modelcl)
        cl.done()
        return modelcl, True


    def reset_image(self,imagename):
        ##### making the residual a blank image
        logging.debug("Setting all values of dummy image to 0.")
        if self.pol=='I':
            hdu = fits.open(imagename + "-residual.fits", mode="update")
            hdu[0].data *= 0.0
            hdu.flush()
            hdu.close()
        else:
            for pola in self.polarisations:
                hdu = fits.open(imagename +"-"+pola+ "-residual.fits", mode="update")
                hdu[0].data *= 0.0
                hdu.flush()
                hdu.close()
    
    def generate_model_from_component_list(self,imagename):
        if self.pol=='I':
            os.system("wsclean -no-dirty -no-update-model-required -restore-list " + \
                      imagename + "-residual.fits " + self.filename[0] + " "+self.outpath+"calibrator-model.fits " + self.vis)
        
        else:
            for i,pola in enumerate(self.polarisations):
                os.system("wsclean -no-dirty -no-update-model-required -restore-list " + \
                      imagename +"-"+pola+ "-residual.fits " + self.filename[i] + " "+self.outpath+"calibrator-"+pola+"-model.fits " + self.vis)
        
        if os.path.isfile("calibrator-model.fits") == False and \
                (os.path.isfile("calibrator-I-model.fits")==False or \
                    os.path.isfile("calibrator-Q-model.fits")==False or \
                    os.path.isfile("calibrator-U-model.fits")==False or \
                    os.path.isfile("calibrator-V-model.fits")==False):
                    
                logging.warning("Calibrator model not generated. Proceeding with point source model")
                raise RuntimeError("WSClean version 3.3 or above. Proceeding with point source model")
        
        if self.model==True:
            for i,pola in enumerate(self.polarisations):
                self.correct_for_restoring_beam("calibrator-"+pola+"-model.fits")
                
    def correct_for_restoring_beam(self,image):
        from casatasks import imhead
        a=imhead(image)
        major=a['restoringbeam']['major']['value']
        minor=a['restoringbeam']['minor']['value']
        cell=abs(a['incr'][0])*180/3.14159*3600
        major_pix=major/cell
        minor_pix=minor/cell
        area=np.pi*major_pix*minor_pix/(4*np.log(2))
        hdu=fits.open(image,mode='update')
        hdu[0].data/=area
        hdu.flush()
        hdu.close()

    
    def check_negative_in_model(self):
        if self.pol=='I':
            max1, min1 = utils.get_image_maxmin(self.outpath+"calibrator-model.fits", local=False)
            if min1 < 0 and (max1 / max(abs(min1), 0.000001)) < 10000:  ### some small negative is tolerable
                raise RuntimeError("Negative in model. Going for point source model")
        else:
            for pola in self.polarisations:
                max1, min1 = utils.get_image_maxmin(self.outpath+"calibrator-"+pola+"-model.fits", local=False)
                if min1 < 0 and (max1 / max(abs(min1), 0.000001)) < 10000:  ### some small negative is tolerable
                    raise RuntimeError("Negative in model. Going for point source model")
                    
    def do_prediction(self):
        if self.predict==False:
            return
        os.system("wsclean -predict -pol "+self.pol+" -name "+self.outpath+"calibrator " + self.vis)
        
    def gen_dummy_image(self,imagename):
        os.system("wsclean -no-dirty -no-update-model-required -size 4096 4096 " + \
                      "-scale 2arcmin -niter 10 -pol "+self.pol+" -name " + imagename + " " + self.vis)            
        
    
    def gen_model_cl(self):
        """
        Generate source models for bright sources as CASA clean components
        
        :param msfile: input visibility
        :param ref_freq: reference frequency of the preset flux values of bright sources
        :param output_freq: output frequency to be written into the CASA component list
        :param includesun: if True, add a precribed solar flux to the source list
        :return:
        """
        self.ctrl_freq()
        if self.avg_freq>60:
            self.point_source_model_needed=True

        if self.point_source_model_needed==True:
           modelcl, ft_needed = self.point_source_model()
           return modelcl, ft_needed    

        if self.includesun == True:
            logging.debug("User wants to add solar model")
            logging.debug("Proceeding to use point source model generation scheme.")
            modelcl, ft_needed = self.point_source_model()
            return modelcl, ft_needed
        try:
            
            logging.debug("Generating component list using Gasperin et al. (2021)")
            
            self.gen_model_file()
            imagename = self.outpath+"dummy"
            logging.debug("Generating a dummy image")
            
            print ("Generating dummy image")
            self.gen_dummy_image(imagename)
                      
            self.reset_image(imagename)           
            
            self.generate_model_from_component_list(imagename)
            
            logging.debug("Model file generated using the clean component list")
            
            self.check_negative_in_model()
            
            self.do_prediction()
            return None, False
        except:
            logging.warning("Negative in model. Going for point source model.") 
            modelcl, ft_needed = self.point_source_model()
            return modelcl, ft_needed    



