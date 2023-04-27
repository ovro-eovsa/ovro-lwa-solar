import sys
sys.path.append('/home/surajit/ovro-lwa-solar')
import solar_pipeline as lwa
import os


os.chdir('73MHz')

solar_ms='20230309_191023_73MHz.ms'
calib_ms='20230310_042936_73MHz.ms'

lwa.pipeline(solar_ms,calib_ms=calib_ms,selfcal=True)
