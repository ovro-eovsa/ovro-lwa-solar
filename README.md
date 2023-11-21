# ovro-lwa-solar
Calibration and imaging pipeline for the Sun based on data taken by the Owens Valley Long Wavelength Array (OVRO-LWA)

Dependencies:
- Modular CASA 6
- suncasa
- Astropy, matplotlib, numpy
- wsclean


## Installation

* Install python 3.8+, git

* Install dependency packages

```bash
pip install astropy numpy matplotlib==3.5

```

* update pip
```bash
python -m pip install --upgrade pip
```

* Install ovrolwasolar module
```bash
git clone https://github.com/binchensun/ovro-lwa-solar.git
cd ovro-lwa-solar
python -m pip install .
```

# run the pipeline

simple example:
```python

import ovrolwasolar.solar_pipeline as lwa
import os

os.chdir('../../data')
solar_ms='20230919_202113_55MHz.ms'
calib_ms='20230919_053329_55MHz.ms'

lwa.image_ms(solar_ms,calib_ms=calib_ms,logging_level='debug')

```