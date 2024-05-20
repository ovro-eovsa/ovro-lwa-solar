from ovrolwasolar import solar_pipeline as lwasp
import pytest
import os, sys, glob
import wget


caltable = '/data07/peijinz/testdir/testdata/caltables/20240517_100405_55MHz.bcal'
sun_slow_ms = '/data07/peijinz/testdir/testdata/testdata/slow/20240519_173002_55MHz.ms'

@pytest.fixture
def prepare_data():
    url_data = 'https://ovsa.njit.edu/lwa-data/tests/testdata.tar'
    dir_tmp = tmp_path_factory.mktemp('testdir')
    os.chdir(dir_tmp)
    wget.download(url_data)
    os.system('tar -xvf testdata.tar')
    yield dir_tmp

@pytest.fixture
def prepare_data_local():
    dir_tmp = tmp_path_factory.mktemp('testdir')
    os.chdir(dir_tmp)
    # copy caltables and ms files to the tmp directory
    data_tar = '/data07/peijinz/testdir/testdata/testdata.tar'
    os.system(f'cp {data_tar} {dir_tmp}')
    os.system('tar -xvf testdata.tar')
    yield dir_tmp

@pytest.fixture
def run_image_quick(prepare_data_local):
    lwasp.image_ms_quick( solar_ms= sun_slow_ms , bcal=caltable, num_phase_cal=0, num_apcal=1 , logging_level='debug')


def test_image_quick(run_image_quick):
    generated_images = glob.glob('*.image.fits')
    assert len(generated_images) == 1
