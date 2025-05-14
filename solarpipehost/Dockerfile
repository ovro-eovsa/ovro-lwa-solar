FROM astronrd/linc:latest

ARG DEBIAN_FRONTEND=noninteractive

# Install Python and virtualenv support
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv

# Create and activate virtual environment
RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"

# Upgrade pip + install packages in virtualenv
RUN pip install --upgrade pip setuptools packaging

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# install modular casa and suncasa
RUN python -m pip install casatools casatasks

RUN python -m pip install git+https://github.com/peijin94/ovro-lwa-solar.git && \
    python -m pip install git+https://github.com/peijin94/suncasa-src.git

# Create a non-root user 'lwauser' and set permissions
#RUN useradd -ms /bin/bash lwauser
#USER lwauser


# Set working directory
#WORKDIR /home/lwauser