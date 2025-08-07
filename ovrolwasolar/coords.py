import numpy as np
from astropy.io import fits
import os
from sunpy.coordinates import sun
from skimage.transform import rotate
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import get_body, EarthLocation
from math import cos, sin, tan, atan, atan2, radians, degrees, hypot

import logging

def angdist(ra1, de1, ra2, de2):
    """
    Calculate angular distance using Vincenty equation (in radians).
    
    Args:
        ra1, de1: Right ascension and declination of first point
        ra2, de2: Right ascension and declination of second point
        
    Returns:
        float: Angular distance in radians
    """
    num1 = cos(de2) * sin(ra2 - ra1)
    num2 = cos(de1) * sin(de2) - sin(de1) * cos(de2) * cos(ra2 - ra1)
    denominator = sin(de1) * sin(de2) + cos(de1) * cos(de2) * cos(ra2 - ra1)
    return atan2(hypot(num1, num2), denominator)

def radec2hpc(ra, de, sun_ra, sun_de, sun_P):
    """
    Convert RA/Dec to helioprojective coordinates (all values in radians).
    
    Args:
        ra, de: Target right ascension and declination
        sun_ra, sun_de: Solar right ascension and declination
        sun_P: Solar P angle
        
    Returns:
        tuple: (rho, hpc_x, hpc_y) coordinates
    """
    rho = angdist(ra, de, sun_ra, sun_de)
    theta = atan2(sin(ra - sun_ra), 
                tan(de) * cos(sun_de) - sin(sun_de) * cos(ra - sun_ra))
    hpc_x = atan(-tan(rho) * sin(theta - sun_P))
    hpc_y = atan(tan(rho) * cos(theta - sun_P))
    return rho, hpc_x, hpc_y

def getSunEphem(reftimestr='', verbose=False):
    """
    Calculate solar ephemeris data using sunpy's direct coordinate functions.
    
    Args:
        reftime: Reference time (default: current time)
        verbose: Print detailed information if True
        
    Returns:
        dict: Solar ephemeris data
    """
    from datetime import datetime
    
    if reftimestr == '':
        start_time = datetime.now()
    else:
        start_time = datetime.strptime(reftimestr, '%Y-%m-%dT%H:%M:%S.%f')
    
    # Convert to astropy time
    obstime = Time(start_time)
    
    # Get coordinates using astropy
    location = EarthLocation.of_site('OVRO')
    phasecentre = get_body('sun', obstime, location)
    ra = phasecentre.ra.to(u.rad).value
    dec = phasecentre.dec.to(u.rad).value
    
    # Get solar angles from sunpy
    P = sun.P(obstime).to(u.rad).value
    B0 = sun.B0(obstime).to(u.rad).value
    L0 = sun.L0(obstime).to(u.rad).value
    
    # Get Earth-Sun distance and apparent radius
    dsun = sun.earth_distance(obstime).to(u.AU).value
    rapp = sun.angular_radius(obstime).to(u.rad).value
    
    if verbose:
        print('Local Solar Ephemeris Calculation')
        print('Date: ', start_time.strftime('%Y-%m-%d %H:%M:%S'))
        print(f'RA: {degrees(ra):.6f}°, Dec: {degrees(dec):.6f}°')
        print(f'Distance: {dsun:.6f} AU')
        print(f'P: {degrees(P):.6f}°, B0: {degrees(B0):.6f}°, L0: {degrees(L0):.6f}°')
        print(f'Apparent radius: {degrees(rapp)*3600.0:.2f} arcsec')
    
    return {
        't': start_time.strftime('%Y-%m-%dT%H:%M:%S.%f'),
        'ra': ra,
        'dec': dec,
        'dsun': dsun,
        'rrate': 0.0,
        'rapp': rapp,
        'P': P,
        'B0': B0,
        'L0': L0
    }

def fitsj2000tohelio(in_fits, out_fits=None, reftime="", toK=True, 
        verbose=False, sclfactor=1.0, subregion=None):
    """
    Convert a FITS image from J2000 to helioprojective coordinates.
    
    Args:
        in_fits (str): Input FITS file path
        out_fits (str): Output FITS file path
        reftime (str): Reference time (default: from FITS header)
        toK (bool): Convert data from Jy/beam to Kelvin if True
        verbose (bool): Print detailed information if True
    """
    # Constants
    JPL_AU = 149597870700.0  # meters
    JPL_RSUN = 696000000.0   # meters
    
    if out_fits is None:
        out_fits = in_fits.replace('.fits', '.helio.fits')

    # Copy input file to output location
    os.system(f"cp {in_fits} {out_fits}")
    
    # Open the FITS file for updating
    hdul = fits.open(out_fits, mode="update")
    hdr = hdul[0].header
    data = hdul[0].data

    
    # Get observation time and solar ephemeris
    obstimestr = reftime if reftime else hdr["DATE-OBS"]
    ephemSun = getSunEphem(obstimestr, verbose=verbose)
    
    # Rotate image by solar P angle
    rotated_data = np.zeros_like(data)
    P_deg = np.degrees(ephemSun['P'])

    logging.debug(f'Rotating image by {P_deg:.2f} degrees')

    if len(data.shape) == 2:
        rotated_data = rotate(data.astype(np.float32), angle=P_deg, 
                            preserve_range=True, mode='constant', cval=np.nan)
    elif len(data.shape) == 4:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                rotated_data[i, j, :, :] = rotate(data[i, j, :, :].astype(np.float32),
                                                angle=P_deg, preserve_range=True,
                                                mode='constant', cval=np.nan)
    
    # Extract and convert coordinates
    crval1 = float(hdr['CRVAL1']) * u.Unit(hdr['CUNIT1'])
    crval2 = float(hdr['CRVAL2']) * u.Unit(hdr['CUNIT2'])
    cdelt1 = float(hdr['CDELT1']) * u.Unit(hdr['CUNIT1'])
    cdelt2 = float(hdr['CDELT2']) * u.Unit(hdr['CUNIT2'])
    
    # Convert to radians
    crval1 = crval1.to(u.rad).value
    crval2 = crval2.to(u.rad).value
    cdelt1 = cdelt1.to(u.rad).value
    cdelt2 = cdelt2.to(u.rad).value
    
    # Convert coordinates
    rho, crval1, crval2 = radec2hpc(crval1, crval2,
                                   sun_ra=ephemSun['ra'],
                                   sun_de=ephemSun['dec'],
                                   sun_P=ephemSun['P'])
    
    # Convert to appropriate units and prepare for header updates
    cdelt1 = -cdelt1  # Correct for different direction of RA and HGLN axes
    hpc_x = degrees(crval1) * 3600.0
    hpc_y = degrees(crval2) * 3600.0
    crota2 = 0.0  # Since we already rotated the image
    
    data = rotated_data

    # Apply subregion cropping if specified
    if subregion is not None:
        xmin, xmax, ymin, ymax = subregion
        if len(data.shape) == 2:
            data = data[ymin:ymax, xmin:xmax]
        elif len(data.shape) == 4:
            data = data[:, :, ymin:ymax, xmin:xmax]
        
        # Update reference pixel in header
        crpix1 = float(hdr.get('CRPIX1', 1))
        crpix2 = float(hdr.get('CRPIX2', 1))
        hdr['CRPIX1'] = crpix1 - xmin
        hdr['CRPIX2'] = crpix2 - ymin

    # update beam angle 
    if 'BPA' in hdr:
        bpa = hdr['BPA']
        logging.debug(f'Original beam position angle: {bpa}')
        if verbose:
            print(f'Updating beam position angle: {bpa}')
        bpa = (bpa - P_deg) % 360.0
        hdr['BPA'] = bpa

    logging.debug(f'Updated beam position angle: {bpa}')

    # Update header keywords
    header_updates = {
        'CRVAL1': hpc_x,
        'CRVAL2': hpc_y,
        'CUNIT1': 'arcsec',
        'CUNIT2': 'arcsec',
        'CDELT1': degrees(cdelt1) * 3600.0,
        'CDELT2': degrees(cdelt2) * 3600.0,
        'CTYPE1': 'HPLN-TAN',
        'CTYPE2': 'HPLT-TAN',
        'DSUN_REF': JPL_AU,
        'DSUN_OBS': ephemSun['dsun'] * JPL_AU,
        'RSUN_REF': JPL_RSUN,
        'RSUN_OBS': abs(degrees(ephemSun['rapp'])) * 3600.0,
        'HGLN_OBS': 0.0,
        'HGLT_OBS': degrees(ephemSun['B0']),
        'CRLN_OBS': degrees(ephemSun['L0']),
        'CRLT_OBS': degrees(ephemSun['B0']),
        'SOLAR_P': degrees(ephemSun['P']),
        'XCEN': hpc_x,
        'YCEN': hpc_y,
        'WCSNAME': 'Helioprojective-cartesian',
        'PC1_1': cos(crota2),
        'PC1_2': -sin(crota2) * cdelt2/cdelt1,
        'PC2_1': sin(crota2) * cdelt1/cdelt2,
        'PC2_2': cos(crota2),
    }
    
    # Update header
    for keyword, value in header_updates.items():
        if verbose:
            print(f'Updating {keyword}: {value}')
        hdr.set(keyword, value)
    
    # Update the data
    hdul[0].data = data
    
    # Convert from Jy/beam to Kelvin if requested
    if toK:
        if 'BUNIT' in hdr and hdr['BUNIT'] == 'K':
            if verbose:
                print('Data is already in Kelvin')
        else:
            if verbose:
                print('Converting data to Kelvin')
                
            if all(key in hdr for key in ['BMAJ', 'BMIN', 'CRVAL3']):
                bmaj = hdr['BMAJ'] * 3600.0  # Convert to arcsec
                bmin = hdr['BMIN'] * 3600.0  # Convert to arcsec
                freq = hdr['CRVAL3']/1e9  # Convert to GHz
                
                convJyb2K = 1.222e6/(bmaj*bmin*freq**2)
                
                if verbose:
                    print(f'Beam major axis: {bmaj:.2f} arcsec')
                    print(f'Beam minor axis: {bmin:.2f} arcsec')
                    print(f'Frequency: {freq:.2f} GHz')
                
                # Convert Jy/beam to K
                hdul[0].data = hdul[0].data * convJyb2K    
                hdr['BUNIT'] = 'K'
            else:
                print('Warning: Missing required header keywords for Jy/beam to K conversion')
    
    # multiply with scale factor
    hdul[0].data = hdul[0].data * sclfactor

    # Save and close
    hdul.flush()
    hdul.close()

    return out_fits

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python j2000_to_helio.py <input_fits> <output_fits> [reftime] [toK] [verbose]")
        sys.exit(1)
    
    in_fits = sys.argv[1]
    out_fits = sys.argv[2]
    reftime = sys.argv[3] if len(sys.argv) > 3 else ""
    toK = True if len(sys.argv) <= 4 or sys.argv[4].lower() == "true" else False
    verbose = True if len(sys.argv) > 5 and sys.argv[5].lower() == "true" else False
    
    fitsj2000tohelio(in_fits, out_fits, reftime, toK, verbose)