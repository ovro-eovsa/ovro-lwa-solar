FROM ubuntu:22.04

# get essential dependencies
RUN apt update -y && apt install python3 python3-dev python3-pip python3-pyqt5 git -y

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1
RUN python -m pip install packaging

# install modular casa and suncasa
RUN python -m pip install casatools casatasks matplotlib 
RUN python -m pip install suncasa

RUN apt install cmake -y

# install wsclean
RUN git clone --no-checkout \
        https://gitlab.com/aroffringa/wsclean.git && \
    mkdir wsclean/build && \
    cd wsclean && git checkout master && \
    cd build && \
    cmake .. -DPORTABLE=True && \
    make install
