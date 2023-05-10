import sys
sys.path.append('/data07/msurajit/ovro-lwa-solar2/')
import solar_pipeline as lwa

time_duration='2023/05/02/20:00:00~2023/05/02/21:00:00'
calib_time_duration='2023/05/02/10:00:00~2023/05/02/11:00:00'
freqstr=['73MHz']
file_path='20230502_solar'

lwa.solar_pipeline(time_duration,calib_time_duration,freqstr,file_path,time_cadence=300)
