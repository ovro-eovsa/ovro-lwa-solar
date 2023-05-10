from primary_beam import primary_beam

filepath='/home/surajit/Downloads'
msfile='/home/surajit/Downloads/20230502/41MHz/final_pipeline_cme_movie/20230502_190039_41MHz_calibrated_selfcalibrated_sun_only.ms'
az=50  ### in degrees
el=60  ### in degrees

pb=primary_beam(msfile=msfile,beam_file_path=filepath)

pb.read_beam_file()
Ibeam,Qbeam,Ubeam,Vbeam=pb.srcIQUV(az=az,el=el)
print (Ibeam,Qbeam,Ubeam,Vbeam)
