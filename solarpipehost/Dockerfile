FROM ubuntu:22.04


ARG DEBIAN_FRONTEND=noninteractive
# get essential dependencies
RUN apt update -y && \
     apt install python3 python3-dev python3-pip python3-pyqt5 \
        bison \
        build-essential \
        cmake \
        flex \
        gfortran \
        git \
        libblas-dev \
        libboost-date-time-dev \
        libboost-filesystem-dev \
        libboost-numpy-dev \
        libboost-program-options-dev \
        libboost-python-dev \
        libboost-system-dev \
        libboost-test-dev \
        libcfitsio-dev \
        libfftw3-dev  \
        libfftw3-double3 \
        libfftw3-single3 \
        libglibmm-2.4-1v5 \
        libgomp1 \
        libgtkmm-3.0-1v5 \
        libhdf5-103 \
        libhdf5-cpp-103 \
        liblapack3 \
        liblua5.3-0 \
        libpangomm-1.4-1v5 \
        libpng16-16 \
        libsigc++-2.0-0v5 \
        libstdc++6 \
        wsclean-dev  -y

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1
RUN python -m pip install packaging

# install modular casa and suncasa
RUN python -m pip install casatools==6.6.0.20 casatasks==6.6.0.20 matplotlib==3.5.2 \
    ipython==8.17.2 jupyter notebook jupyterlab==4.0.9
RUN python -m pip install suncasa

