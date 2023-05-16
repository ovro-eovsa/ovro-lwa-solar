import numpy as np
import math,os,logging
from casatasks import clearcal, ft
from casatools import table, measures, componentlist, msmetadata
from primary_beam import woody_beam as beam
import utils
from astropy.io import fits
import primary_beam

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
    def __init__(self,vis=None,filename='calibrator_source_list.txt',pol='XX,YY',separate_pol=True):
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
        self.point_source_model_needed=False
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
            if i not in ['XX','YY','I']:
                raise RuntimeError("Only XX,YY,I recognised now.")
        self._pol=value
        

    def gen_model_file(self):
        """
        :param filename: output txt file that contains clean components for all visible strong calibration sources
            in wsclean format
        :param visibility: input visibility
        :return: N/A
        """

        srcs = utils.get_strong_source_list()

        print("generating model file")

        tb=table()
        tb.open(self.vis)
        t0 = tb.getcell('TIME', 0)
        tb.close()
        
        me = measures()
        ovro = me.observatory('OVRO_MMA')
        time = me.epoch('UTC', '%fs' % t0)
        me.doframe(ovro)
        me.doframe(time)
        
        pb=beam(msfile=self.vis)
        

        self.file_handle = [open(self.filename[i], 'w') for i in range(self.num_pol)]
        num_source = 0
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
            az=d['m0']['value']*180/np.pi
            if elev<0:
                del srcs[s]
            else:
                jones_matrix=pb.srcIQUV(az=az,el=elev)#math.sin(elev) ** 1.6
                if jones_matrix[0,0]**2 < self.min_beam_val or jones_matrix[1,1]**2< self.min_beam_val:
                    del srcs[s]
                else:
                    print(srcs[s]['label'])
                    
                    for i in range(self.num_pol):
                        self.write_source_file(i,srcs[s]['label'], jones_matrix, num_source)
                    num_source += 1

        return    

    
    def write_source_file(self,current_pol_index,source_name, jones_matrix, source_num):  #### works only if logarithimicSI is false

        try:
            f1 = open(self.calfilepath + source_name[:-1]+ "-sources.txt", "r")
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
                        primary_beam_value=self.primary_beam_value(current_pol_index,jones_matrix)
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

    def primary_beam_value(self,current_pol_index,jones_matrix):  
        
        if self.polarisations[current_pol_index]=='XX':
            return primary_beam.primary_beam_correction_val('XX',jones_matrix)
        if self.polarisations[current_pol_index]=='YY':
            return primary_beam.primary_beam_correction_val('YY',jones_matrix)
            
        if self.polarisations[current_pol_index]=='I':
            return primary_beam.primary_beam_correction_val('I',jones_matrix)
    

    def ctrl_freq(self):
        msmd=msmetadata()
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
        if self.polarisations[0]=='I':
            pb_val=primary_beam.primary_beam_correction_val('I',jones_matrix)
            return flux_hi * 10 ** (sp * math.log(self.output_freq / self.ref_freq, 10))*pb_val
        else:
            xx_pb_val=primary_beam.primary_beam_correction_val('XX',jones_matrix)
            yy_pb_val=primary_beam.primary_beam_correction_val('YY',jones_matrix)
            flux_val=flux_hi * 10 ** (sp * math.log(self.output_freq / self.ref_freq, 10))
            I_flux=0.5*(xx_pb_val*flux_val+yy_pb_val*flux_val)
            Q_flux=0.5*(xx_pb_val*flux_val-yy_pb_val*flux_val)
            return [I_flux,Q_flux, 0, 0]
    
    
    def point_source_model(self):
                           
        srcs = utils.get_strong_source_list()
        
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

        cl = componentlist()        
                
        pb=beam(msfile=self.vis)
        

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
            az=d['m0']['value']*180/np.pi
            if elev<0:
                del srcs[s]
            else:
                jones_matrix=pb.srcIQUV(az=az,el=elev)
                if jones_matrix[0,0]**2 < self.min_beam_val or jones_matrix[1,1]**2< self.min_beam_val:
                    del srcs[s]
                else:
                    print (srcs[s]['label'])
                    srcs[s]['flux'] = self.flux80_47(float(srcs[s]['flux']), srcs[s]['alpha'], jones_matrix) 

        cl.done()

        
        modelcl = self.vis.replace('.ms', '.cl')
        for s in srcs:
            cl.addcomponent(flux=s['flux'], dir=s['position'], index=s['alpha'],
                            spectrumtype='spectral index', freq='{0:f}MHz'.format(self.output_freq), label=s['label'])
            
            logging.debug(
                    "cl.addcomponent(flux=%s, dir='%s', index=%s, spectrumtype='spectral index', freq='47MHz', label='%s', polarization='Stokes')" % (
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
                (os.path.isfile("calibrator-XX-model.fits")==False or \
                    os.path.isfile("calibrator-YY-model.fits")==False):
                    
                logging.warning("Calibrator model not generated. Proceeding with point source model")
                raise RuntimeError("WSClean version 3.3 or above. Proceeding with point source model")
    
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
        if self.avg_freq>75:
            self.point_source_model_needed=True

        if self.point_source_model_needed==True:
           modelcl, ft_needed = self.point_source_model()
           return modelcl, ft_needed    

        if self.includesun == True:
            logging.info("User wants to add solar model")
            logging.info("Proceeding to use point source model generation scheme.")
            modelcl, ft_needed = self.point_source_model()
            return modelcl, ft_needed
        try:
            
            logging.info("Generating component list using Gasperin et al. (2021)")
            
            self.gen_model_file()
            imagename = self.outpath+"dummy"
            logging.debug("Generating a dummy image")
            
            
            self.gen_dummy_image(imagename)
                      
            self.reset_image(imagename)           
            
            self.generate_model_from_component_list(imagename)
            
            logging.info("Model file generated using the clean component list")
            
            self.check_negative_in_model()
            
            self.do_prediction()
            return None, False
        except:
            modelcl, ft_needed = self.point_source_model()
            return modelcl, ft_needed    



