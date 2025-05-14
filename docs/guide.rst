Quick Guide for Imaging Data
============================

This guide provides a quick overview of how to use the OVRO-LWA Solar package to plot imaging data.

Preparation
-----------

Set up a Python 3.10 environment with the required packages. You can use ``conda`` or ``pip`` to create a new environment.

Then install the ``ovrolwasolar`` package using ``pip``:

.. code:: bash

   pip install git+https://github.com/ovro-eovsa/ovro-lwa-solar.git

Optionally, you can use our prebuilt Docker image. The image is available on Docker Hub and can be pulled using the following command (or use other container hosting services: Apptainer, Podman, Singularity, etc.):

.. code:: bash

   docker pull peijin/lwa-solar-pipehost
   docker run -it --rm -v /path/to/your/data:/data peijin/lwa-solar-pipehost

Docker image URL: `https://hub.docker.com/repository/docker/peijin/lwa-solar-pipehost <https://hub.docker.com/repository/docker/peijin/lwa-solar-pipehost>`_

Imaging Products
----------------

For imaging products, each file contains data from a single time slot, with images available across multiple frequencies.

We offer two file formats: **FITS** and **HDF5**. Both contain the same information, but the HDF5 version is compressed to reduce file size. The compression is frequency-dependent; the pixel size is based on the beam size at each frequency.

We recommend using the **HDF5** format due to its smaller size. It can be converted back to a standard FITS file using the following Python code:

.. code:: python

   from ovrolwasolar import utils as outils
   outils.recover_fits_from_h5('path/to/hdf/file.h5', 'path/to/fits/file.fits')

Plotting
--------

The following notebook demonstrates how to plot the imaging data:

`View Notebook Demo <./_static/demo_plot_img.html>`_
