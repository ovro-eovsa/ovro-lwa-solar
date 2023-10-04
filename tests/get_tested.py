import solar_pipeline as lwa
import os,glob



def run_tests(calib_ms=None, slow_solar_ms1=None, slow_solar_ms2=None, \
                fast_solar_ms1=None,test_slow=False, test_fast=False):



    if test_slow:

        ###############################################################################
        '''
        1. Calibration ms=True, bandpass table=False, DI_selfcal_full=True, DD_selfcal_full=True, fluxscaling=False, slow=True
        '''

        imagename='sun_only_test01'
        lwa.image_ms(calib_ms=calib_ms,imagename=imagename,solar_ms=slow_solar_ms1) 
        os.system("cp analysis.log analysis_test01.log")

        ###############################################################################3

        ## Copying the caltable folder for later usage
        os.system("cp -r caltables caltables_copy")
        caltable_name=glob.glob("caltables/*.bcal")[0]

        ###############################################################################
        '''
        2. Calibration ms=False, bandpass table=True, DI_selfcal_full=True, DD_selfcal_full=True, fluxscaling=False, slow=True
        '''
        imagename='sun_only_test02'
        lwa.image_ms(bcal=caltable_name,imagename=imagename,solar_ms=slow_solar_ms2,solint_full_DI_selfcal=1, \
                        solint_partial_DI_selfcal=1, solint_full_DD_selfcal=1,solint_partial_DD_selfcal=1)
		        ### set both full DI and DD selfcal will both happen when the difference is more than 1s.
        os.system("cp analysis.log analysis_test02.log")
        ###############################################################################

        '''
        3. Calibration ms=False, bandpass table=True, DI_selfcal_full=False, DI_selfcal_partial=True, DD_selfcal_full=True, fluxscaling=False, slow=True		
        '''
        os.system("rm -rf caltables/"+slow_solar_ms2[:-3]+"*")
        imagename='sun_only_test03'
        lwa.image_ms(bcal=caltable_name,imagename=imagename,solar_ms=slow_solar_ms2,solint_full_DI_selfcal=10000, \
				        solint_partial_DI_selfcal=1, solint_full_DD_selfcal=1, solint_partial_DD_selfcal=1)
        os.system("cp analysis.log analysis_test03.log")		        
        ###############################################################################
        '''
        4. Calibration ms=False, bandpass table=True, DI_selfcal_full=False, DI_selfcal_partial=False, DD_selfcal_full=True, 
        fluxscaling=False, slow=True
        '''
        os.system("rm -rf caltables/"+slow_solar_ms2[:-3]+"*")
        imagename='sun_only_test04'
        lwa.image_ms(bcal=caltable_name,imagename=imagename,solar_ms=slow_solar_ms2,solint_full_DI_selfcal=10000, \
				        solint_partial_DI_selfcal=100000, solint_full_DD_selfcal=1, solint_partial_DD_selfcal=1)
        os.system("cp analysis.log analysis_test04.log")	
        ###############################################################################
        '''
        5. Calibration ms=False, bandpass table=True, DI_selfcal_full=False, DI_selfcal_partial=False, DD_selfcal_full=False, 
        DD_selfcal_partial=True, fluxscaling=False, slow=True
        '''
        os.system("rm -rf caltables/"+slow_solar_ms2[:-3]+"*")
        imagename='sun_only_test05'
        lwa.image_ms(bcal=caltable_name,imagename=imagename,solar_ms=slow_solar_ms2,solint_full_DI_selfcal=10000, \
				        solint_partial_DI_selfcal=100000, solint_full_DD_selfcal=1000000, solint_partial_DD_selfcal=1)
        os.system("cp analysis.log analysis_test05.log")	
        ###############################################################################
        '''
        6. Calibration ms=False, bandpass table=True, DI_selfcal_full=False, DI_selfcal_partial=False, DD_selfcal_full=False, 
        DD_selfcal_partial=False, fluxscaling=False, slow=True
        '''

        os.system("rm -rf caltables/"+slow_solar_ms2[:-3]+"*")
        imagename='sun_only_test06'
        lwa.image_ms(bcal=caltable_name,imagename=imagename,solar_ms=slow_solar_ms2,solint_full_DI_selfcal=10000, \
				        solint_partial_DI_selfcal=100000, solint_full_DD_selfcal=1000000, solint_partial_DD_selfcal=10000)
        os.system("cp analysis.log analysis_test06.log")	
        ###############################################################################
        '''
        7. Calibration ms=True, bandpass table=False, DI_selfcal_full=True, DD_selfcal_full=True, fluxscaling=True, slow=True
        '''
        os.system("rm -rf caltables/"+slow_solar_ms2[:-3]+"*")
        imagename='sun_only_test07'
        lwa.image_ms(bcal=caltable_name,imagename=imagename,solar_ms=slow_solar_ms2,solint_full_DI_selfcal=1, \
                        solint_partial_DI_selfcal=1, solint_full_DD_selfcal=1, solint_partial_DD_selfcal=1,\
			            do_fluxscaling=True)
        os.system("cp analysis.log analysis_test07.log")	
        ###############################################################################
        '''
        8. Calibration ms=False, bandpass table=True, DI_selfcal_full=False, DI_selfcal_partial=False, DD_selfcal_full=False, DD_selfcal_partial=False, fluxscaling=True, slow=True
        '''
        os.system("rm -rf caltables/"+slow_solar_ms2[:-3]+"*")
        imagename='sun_only_test08'
        lwa.image_ms(bcal=caltable_name,imagename=imagename,solar_ms=slow_solar_ms2,solint_full_DI_selfcal=10000, \
				        solint_partial_DI_selfcal=100000, solint_full_DD_selfcal=1000000, solint_partial_DD_selfcal=10000,\
					        do_fluxscaling=True)
        os.system("cp analysis.log analysis_test08.log")			        
        ###############################################################################
        print ("Slow visibility tests successfully done")
        ###############################################################################

    if test_fast:

        '''
        9. Calibration ms=True, bandpass table=False, DI_caltable available, DD_caltable available, flux scaling=False, fast=True
        '''
        os.system("rm -rf caltables/"+slow_solar_ms2[:-3]+"*")
        imagename='sun_only_test09'
        lwa.image_ms(calib_ms=calib_ms, bcal=caltable_name,imagename=imagename,solar_ms=fast_solar_ms1, fast=True)
        os.system("cp analysis.log analysis_test09.log")	
        ###############################################################################

        '''
        10. Calibration ms=True, bandpass table=True, DI_caltable available, DD_caltable not available, flux scaling=False, fast=True
        '''
        os.system("rm -rf caltables/"+slow_solar_ms1[:-3]+"_selfcalibrated*")
        imagename='sun_only_test10'
        lwa.image_ms(calib_ms=calib_ms, bcal=caltable_name,imagename=imagename,solar_ms=fast_solar_ms1, fast=True)
        os.system("cp analysis.log analysis_test10.log")	
        ###############################################################################

        '''
        11. Calibration ms=True, bandpass table=True, DI_caltable not available, DD_caltable not available, flux scaling=False, fast=True
        '''
        os.system("rm -rf caltables/"+slow_solar_ms1[:-3]+"_*")
        imagename='sun_only_test11'
        lwa.image_ms(calib_ms=calib_ms, bcal=caltable_name,imagename=imagename,solar_ms=fast_solar_ms1, fast=True)
        os.system("cp analysis.log analysis_test11.log")	
        ###############################################################################

        '''
        12. Calibration ms=True, bandpass table=True, DI_caltable available, DD_caltable not available, flux scaling=True, flux caltable available, fast=True
        '''

