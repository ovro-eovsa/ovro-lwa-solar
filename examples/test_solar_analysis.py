import sys
sys.path.append('/data07/msurajit/ovro-lwa-solar2/')
import solar_pipeline as lwa

time_duration='2023/05/02/18:00:00~2023/05/02/18:30:00'
freqstr=['82MHz']
file_path='20230502_solar'

lwa.solar_pipeline(time_duration,freqstr,file_path,time_cadence=300)
