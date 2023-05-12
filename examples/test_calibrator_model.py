from generate_calibrator_model import model_generation


msfile='/home/surajit/Downloads/ovro_typeIII/quiet_sun/82MHz/20230309_191023_82MHz.ms'

md=model_generation(vis=msfile,pol='I',separate_pol=False)
md.calfilepath='/home/surajit/ovro-lwa-solar/defaults/'

#md.gen_model_file()
md.gen_model_cl()

