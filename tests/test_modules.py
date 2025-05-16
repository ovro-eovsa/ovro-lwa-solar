# pytest test_modules.py
# if ~/.casa/data/ is not a directory, create it
import os
os.makedirs(os.path.expanduser('~/.casa/data/'), exist_ok=True)

from ovrolwasolar import solar_pipeline as lwasp
import pytest
import os, sys, glob


@pytest.fixture
def prepare_data():
    url_data = 'https://ovsa.njit.edu/lwa-data/tests/testdata.tar'
    dir_tmp = './tmpdir/'
    # create a tmp directory
    os.makedirs(dir_tmp, exist_ok=True)
    os.chdir(dir_tmp)
    os.system('wget ' + url_data)
    os.system('tar -xvf testdata.tar')
    yield dir_tmp

@pytest.fixture
def prepare_data_local():
    dir_tmp = './tmpdir/'
    # create a tmp directory
    os.makedirs(dir_tmp, exist_ok=True)
    # copy caltables and ms files to the tmp directory
    os.chdir(dir_tmp)
    data_tar = '/data07/peijinz/tmp/testdata.tar'
    os.system('cp '+data_tar+' ./')
    os.system('tar -xvf testdata.tar')
    yield dir_tmp

@pytest.fixture
def run_wsclean(prepare_data):
    msfile = './testdata/slow/20240519_173002_55MHz.ms'
    imagename = msfile[:-3] + "_self" + str(0)
    a = lwasp.deconvolve.run_wsclean(msfile, imagename=imagename, niter=100,
            mgain=0.9, auto_mask=False, auto_threshold=False, pol="I", auto_pix_fov=True, quiet=False)
    yield a
    
def test_deconvolve(run_wsclean):
    expected_image = './testdata/slow/20240519_173002_55MHz_self0-image.fits'
    assert os.path.exists(expected_image)

"""
@pytest.fixture
def run_image_quick(prepare_data):
    dir_tmp = prepare_data_local
    lwasp.image_ms_quick( solar_ms=  './testdata/slow/20240519_173002_55MHz.ms', 
                        bcal='./testdata/caltables/20240517_100405_55MHz.bcal',
                        num_phase_cal=1, num_apcal=0 , logging_level='debug')

def test_image_quick(run_image_quick):
    generated_images = glob.glob('./testdata/slow/*image.fits')
    assert len(generated_images) == 1
"""

