from astropy.io import fits
import h5py
import numpy as np
from scipy.ndimage import zoom    
#from matplotlib import pyplot as plt

def compress_fits_to_h5(fitsfile, hdf5_file):
    """
    Compress a fits file to a h5 files
    
    :param fitsfile: the fits file to be compressed
    :param hdf5_file: the h5 file to be saved
    """

    hdul = fits.open(fitsfile)
    data = hdul[0].data
    header = hdul[0].header

    # ch_vals
    # [('cfreqs', '>f4'), ('cdelts', '>f4'), ('bmaj', '>f4'), ('bmin', '>f4'), ('bpa', '>f4')]
    ch_vals = []
    for ch_val in hdul[1].data.dtype.names:
        ch_vals.append(hdul[1].data[ch_val])
    ch_vals = np.array(ch_vals)

    downsize_ratio = (hdul[1].data['bmin']*3600)/ 3 / hdul[0].header['CDELT2']
    recover_data  = np.zeros_like(data)
    with h5py.File(hdf5_file, 'w') as f:
        # Create a dataset for the FITS data
        for pol in range(0, data.shape[0]):
            for ch_idx in range(0, len(downsize_ratio)):
                downsized_data = zoom(data[0,ch_idx,:,:], 1/downsize_ratio[ch_idx], order=5)
                dset = f.create_dataset('FITS_pol'+str(pol)+'ch'+str(ch_idx).rjust(4,'0') , data=downsized_data,compression="gzip", compression_opts=9)
                recover_data[0,ch_idx,:,:] = zoom(downsized_data, data.shape[-1]/downsized_data.shape[-1], order=5)
            
            # Add FITS header info as attributes
        dset = f.create_dataset('ch_vals', data=ch_vals)
        dset.attrs['arr_name'] = hdul[1].data.dtype.names
        dset.attrs['original_shape'] = data.shape
        for key, value in header.items():
            dset.attrs[key] = value


def recover_fits_from_h5(hdf5_file, fname_recover):
    """
    Recover a fits file from a h5 files
    
    :param hdf5_file: the h5 file to be read
    :param fname_recover: the fits file to be recovered
    """
    with h5py.File(hdf5_file, 'r') as f:
        
        # Read in the ch_vals
        ch_vals = f['ch_vals'][:]
        ch_vals_names = f['ch_vals'].attrs['arr_name']
        ch_vals = {ch_vals_names[i]:ch_vals[i] for i in range(len(ch_vals_names))}
        attaching_columns = []
        for key in ch_vals.keys():
            attaching_columns.append(fits.Column(name=key, format='E', array=ch_vals[key]))

        datashape = f['ch_vals'].attrs['original_shape']

        # Read in the compressed data
        recover_data = np.zeros(datashape)
        for pol in range(0, datashape[0]):
            for ch_idx in range(0, len(ch_vals['cfreqs'])):
                tmp_small=f['FITS_pol'+str(pol)+'ch'+str(ch_idx).rjust(4,'0')][:]
                recover_data[pol,ch_idx,:,:] = zoom(tmp_small, datashape[-1]/tmp_small.shape[-1], order=5)

        # Read in the header
        header = {}
        for key in f['ch_vals'].attrs.keys():
            header[key] = f['ch_vals'].attrs[key]
        
        header.pop('arr_name', None)
        header.pop('original_shape', None)

        # convert header to fits header obj
        header = fits.Header(header)

        # Write out the recovered FITS file 
        hdu_list = fits.HDUList([fits.PrimaryHDU(recover_data, header), fits.BinTableHDU.from_columns(attaching_columns)])
        hdu_list.writeto(fname_recover, overwrite=True)