import sys
sys.path.append('../')
import ovrolwasolar.solar_pipeline as lwasp
import os

os.chdir('../../data/test001/')

solar_ms='20230919_202113_55MHz.ms'
calib_ms='20230919_053329_55MHz.ms'

lwasp.image_ms(solar_ms,calib_ms=calib_ms,logging_level='debug')
