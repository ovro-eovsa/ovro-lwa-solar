===================
Fast Visibility
===================

Intro
-----

Fast Visibility data from LWA solar observation has 0.1s time resolution, with baselines from 48 atenna.
Designed for fast solar activities imaging.

Data processing
---------------

From pipeline the fastvis visibility files are calibrated with slow visibilities (which comes from slow pipeline)
Then stored as MS files.

Interferometry imaging
=======================

As MS files, fastvis can be imaged with WSCLEAN or CASA. 
The following is an example of imaging with WSCLEAN.

.. code:: bash
    
    wsclean -name fastvis -size 512 512 -scale 15asec -niter 2000 \
        -auto-threshold 1.0 -auto-mask 3.0 -mgain 0.85 -data-column DATA \
        -mem 85 -no-update-model-required -no-dirty -no-reorder -no-fit-beam \
        --intervals 1 100 --intervals-out 100 uvsub.ms


Single Component Gaussian Source Fitting
========================================
For bright and single soruce events, most of the case can be represented by single component Gaussian model.


an example: 
`notebook <./_static/fastvisfit.html>`_ 
