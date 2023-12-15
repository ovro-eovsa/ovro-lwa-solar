import sys
sys.path.append('/data07/msurajit/ovro-lwa-solar1')
from ovrolwasolar import solar_pipeline as lwa
import os,glob

calib_ms='20230319_010731_55MHz.ms'

selfcal_times=['221509','223002','224525','230008']
selfcal_ms=['20230318_'+t+'_55MHz.ms' for t in selfcal_times]

num_times=len(selfcal_times)

for t in range(num_times):
    imagename='sun_only_'+selfcal_times[t]+'_55MHz'
    if t==0:
        lwa.pipeline(solar_ms=selfcal_ms[t],calib_ms=calib_ms,imagename=imagename,selfcal=True)
        bcal=glob.glob("*.bcal")[0]
    else:
        lwa.pipeline(solar_ms=selfcal_ms[t],bcal=bcal,imagename=imagename,selfcal=True)


times=['222010','222501','223503','224004','225006','225507']
img_ms=['20230318_'+t+'_55MHz.ms' for t in times]

num_times=len(times)

for t in range(num_times):
    imagename='sun_only_'+times[t]+"_55MHz"
    lwa.apply_solutions_and_image(img_ms[t],bcal,imagename)



