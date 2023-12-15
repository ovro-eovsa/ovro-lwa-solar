from ovrolwasolar.primary_beam import woody_beam

filepath='/opt/beam/'
msfile='/data07/msurajit/20230502/78MHz/20230502_194508_78MHz_calibrated_selfcalibrated_sun_only.ms'
az=50  ### in degrees
el=60  ### in degrees

pb=woody_beam(msfile=msfile,beam_file_path=filepath)

pb.srcjones([az],[el])

j=pb.get_source_pol_factors(pb.jones_matrices[0,:,:])
print (j)
#print (Ibeam,Qbeam,Ubeam,Vbeam)
#print (pb.srcjones(l=0.35,m=0.76))
