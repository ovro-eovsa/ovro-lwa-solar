# Docker/Singularity for pipeline

Using contianer technology to run the pipeline is a good way to ensure that the pipeline is run in a consistent environment, and also simplifies the process to setup the enviroment.  This directory contains the files needed to build a container for the pipeline.

Based on Ubuntu 22.04, the container is built using the `Dockerfile` in this directory.

## Installed software/packages:

* Python 3.10
* casatools 6.6.0.20
* casatasks 6.6.0.20
* suncasa
* wsclean 3.0
* ipython 8.17.2
* jupyterlab 4.0.9

## Use with Docker

(A) Run interactive bash command line  
```bash
docker run --rm -i -v /mnt:/mnt -u root -t peijin/lwadata /bin/bash
```
Then you will get a shell env will all env dependencies

(B) Run a jupyter to start  
```bash
docker run --rm -i -p 8998:8998 -v /mnt:/mnt -u root -t peijin/lwadata /bin/bash -c "jupyter-lab --notebook-dir=/mnt --ip='*' --port=8998 --no-browser --allow-root"
```