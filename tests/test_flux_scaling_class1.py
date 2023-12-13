from ovrolwasolar import flux_scaling_class1

vis='/home/surajit/Downloads/ovro_typeIII/quiet_sun/82MHz/pipeline_processing/20230309_191023_82MHz_calibrated.ms'
fc=flux_scaling_class1.flux_scaling(vis=vis)

fc.correct_flux_scaling()
