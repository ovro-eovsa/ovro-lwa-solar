import sys
sys.path.append('/data07/msurajit/ovro-lwa-solar')
from ovrolwasolar.primary_beam import analytic_beam as beam
import numpy as np

filepath='/data07/msurajit/primary_beam_files/'
msfile='/home/surajit/Downloads/20230502/41MHz/final_pipeline_cme_movie/20230502_190039_41MHz.ms'
az=np.array([50,180])  ### in degrees
el=np.array([60,30])  ### in degrees

pb=beam(msfile=msfile,beam_file_path=filepath)

pb.srcjones(az=az,el=el)

pol_frac=pb.get_source_pol_factors(pb.jones_matrices[0,:,:])
print (pol_frac)

pol_frac=pb.get_source_pol_factors(pb.jones_matrices[1,:,:])
print (pol_frac)


