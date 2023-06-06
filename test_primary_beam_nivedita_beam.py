import sys
sys.path.append('/data07/msurajit/ovro-lwa-solar')
from primary_beam import jones_beam
import numpy as np

filepath='/data07/msurajit/primary_beam_files/'
msfile='/home/surajit/Downloads/20230502/41MHz/final_pipeline_cme_movie/20230502_190039_41MHz.ms'
az=np.array([50,180])  ### in degrees
el=np.array([60,30])  ### in degrees

pb=jones_beam(msfile=msfile,beam_file_path=filepath)

jones_matrices=pb.srcjones(az=az*np.pi/180,el=el*np.pi/180)

np.save("jones_matrices.npy",jones_matrices)

