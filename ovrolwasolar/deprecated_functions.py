def correlation_based_leakage_correction(msname,imagename, slope=None,outms=None,sol_area=400,thresh=7,correlation_thresh=10):
    '''
    This function calculates the correlation of Q,U and V with I, after taking into account the leakage of the beam. Any
    residual correlation should come from leakage. We determine the correlation factor and then use the image based
    leakage correction method to apply this to the data.
    
    :param msname: Name of MS
    :param imagename: Prefix of the image which shall be used to calculate the correlation parameters
    :param slope: The linear correlation between I and various Stokes parameters. Should be in order of [Q,U,V]
    :param outms: Name of the MS which will be output after correction.
    :param sol_area: The area around Sun in arcminutes which will be considered for determining the correction factor
    :param thresh: thresh x rms is the threshold which will be used to ignore low SNR points during correlation 
                    calculation
    :param correlation_thresh: The correlation should be determined with at least this SNR for it to be considered
                                as real.
    :return outms: The MS after correction
    :rtype : str
    '''
    if slope is None:
        me=measures()
        m = utils.get_sun_pos(msname, str_output=False)
        solar_ra=m['m0']['value']*180/np.pi
        solar_dec=m['m1']['value']*180/np.pi
        
        head=fits.getheader(imagename+"-I-image.fits")

        freq=head['CRVAL3']*1e-6
        imwcs=WCS(head) 
        solar_xy = imwcs.all_world2pix(list(zip([solar_ra], [solar_dec],[head['CRVAL3']],[head['CRVAL4']])), 1)   
        
        if head['cunit1'] == 'deg':
            dx = np.abs(head['cdelt1'] * 60.)
        else:
            print(head['cunit1'] + ' not recognized as "deg". Model could be wrong.')
        if head['cunit2'] == 'deg':
            dy = np.abs(head['cdelt2'] * 60.)
        else:
            print(head['cunit2'] + ' not recognized as "deg". Model could be wrong.')                   
        sol_area_xpix = int(sol_area / dx)
        sol_area_ypix = int(sol_area / dy)
        
        solx=int(solar_xy[0][0])
        soly=int(solar_xy[0][1])
        
        ymin=soly - sol_area_ypix // 2
        ymax=soly + sol_area_ypix // 2 + 1
        xmin=solx - sol_area_xpix // 2
        xmax=solx + sol_area_xpix // 2 + 1
        

        
        Idata=fits.getdata(imagename+"-I-image.fits")
        Qdata=fits.getdata(imagename+"-Q-image.fits")[0,0,ymin:ymax,xmin:xmax]
        Udata=fits.getdata(imagename+"-U-image.fits")[0,0,ymin:ymax,xmin:xmax]
        Vdata=fits.getdata(imagename+"-V-image.fits")[0,0,ymin:ymax,xmin:xmax]
        
        rms=np.nanstd(Idata)
        pos=np.where(Idata<thresh*rms)
        Idata[pos]=np.nan
        
        Idata=Idata[0,0,ymin:ymax,xmin:xmax]
        
        solar_az,solar_el=utils.get_solar_azel(msname)        
        factors=get_beam_factors([solar_el],[solar_az],freq=freq)
        

        
        Qdata_true=Qdata-factors[0,1]*Idata
        Udata_true=Udata-factors[0,2]*Idata
        Vdata_true=Vdata-factors[0,3]*Idata
        
        slope=np.zeros(3)
        error=np.zeros(3)
        
        pos=np.where(np.isnan(Idata)==False)
        p,cov=np.polyfit(Idata[pos],Qdata_true[pos],deg=1,cov=True)
        
        if abs(p[0])>correlation_thresh*np.sqrt(cov[0,0]):
            slope[0]=p[0]
            error[0]=cov[0,0]
        
        p,cov=np.polyfit(Idata[pos],Udata_true[pos],deg=1,cov=True)
        if abs(p[0])>correlation_thresh*np.sqrt(cov[0,0]):
            slope[1]=p[0]
            error[1]=cov[0,0]
        
        
        p,cov=np.polyfit(Idata[pos],Vdata_true[pos],deg=1,cov=True)
        if abs(p[0])>correlation_thresh*np.sqrt(cov[0,0]):
            slope[2]=p[0]
            error[2]=cov[0,0]
        
        slope=np.array(slope)*(-1)
    
    if outms is None:
        outms=msname.replace(".ms","_correlation_leak_corrected.ms")

    outms=correct_image_leakage(msname,slope,outms=outms)
    return outms
    
#TODO This function needs rewritting. 
def image_based_leakage_correction(msname,stokes='Q,U,V',factor=None,outms=None):
    '''
    This function implements an image based leakage correction.
    However, note that this function does not determine the factor
    by which the correction is done. It expects that the factor will
    be provided. By default he factor is None, which means nothing will be
    done. The order of the stokes parameters should be same as the order in
    which the correction factors are provided. The factors should be determined
    as follows:
    
    Qfrac_beam-Qfrac_observed
    Ufrac_beam-Ufrac_observed
    Vfrac_beam-Vfrac_observed
    
    :param msname: MS which has the data to be corrected
    :type msname: str
    :param stokes: Stokes parameters for which image based leakage correction
                    will be done; example "Q,U,V"
    :type stokes: str
    :param factor: the correction factor. Should in units of I. Should be a list
    :type factor: float
    :param outms: Name of output MS; optional
    :type outms: str
    '''
    
    if outms is None:
        outms=msname.replace(".ms","_image_leak_corrected.ms")
    
    present=utils.check_corrected_data_present(msname)
    if present:
        datacolumn='CORRECTED'
    else:
        datacolumn='DATA'
    
    split(vis=msname,outputvis=outms,datacolumn=datacolumn)
    
    if factor is None:
        return outms
    
    ind_stokes=stokes.split(',') ### contains the individual stokes parameters
    num_stokes=len(ind_stokes)
    
    if num_stokes!=len(factor):
        logging.warning("The length of correction factor does not "+\
                        "match the number of stokes parameters given."+\
                        " I will assume all reamining correction factors "+\
                        "to be zero. Please proceed with caution")
    
    
    
    for j,pol in enumerate(ind_stokes):
        if pol=='Q':
            correct_Q_leakage(outms,factor[j],inplace=True)
        if pol=='U':
            correct_U_leakage(outms,factor[j])
        if pol=='V':
            correct_V_leakage(outms,factor[j])
    return
    
def get_img_correction_factor(imagename,stokes,msfile,imgcat=None,sun_only=False, sol_area=400.,src_area=200,thresh=5,limit_frac=0.05):
    '''
    This function determines the correction factor using an image based method. It calculates the difference between ratio of 
    predicted Q/I,U/I,V/I values and their observed values for each source detected. For all sources other than Sun,
    the model is essentially a point. To get the I flux, we search for the peak within a 200arcmin region around
    the detected source. For the Sun, we have put in a scaled copy of the I model. Hence we do a average of the
    ratios between a 400arcmin region around the Sun. Then it does a weighted average of
    the differences. The weighting is done by the Stokes I fluxes. Thus the difference of the brightest source gets the
    highest weight. If the resultant mean is small, then wo do no bother to do an image based correction. The way the
    correction fraction is determined implies that the corrected image should be
    Q_corrected=Q_obs+correction_factor*Iobs
    U_corrected=U_obs+correction_factor*Iobs
    V_corrected=V_obs+correction_factor*Iobs
    
    Please also ensure that the Stokes I and other parameters are correlated before running 
    this function for that Stokes parameter.
    
    :param imagename: Prefix of the image. This means imagename-I-image.fits, imagename-Q-image.fits etc should exist
    :type imagename: str
    :param stokes: Stokes of the image. Can be either Q,U,V
    :type stokes: str
    :param msfile: MS name, which is used to determine the solar loc
    :type msfile: str
    :param imgcat: Catalog of the source list returned by PyBDSF. Default is None. Should be specified if sun_only=False
    :type imgcat: str
    :param sol_area: Diameter of region around Sun which will be used to calculate correction factor
    :type sol_area: integer
    :param src_area: Diameter of region around other sources which will be used to calculate correction factor
    :type src_area: integer
    :param thresh: Threshold in terms of rms which will be used to determine a source detection in Stokes images
    :type thresh: float
    :param limit_frac: If absolute of the correction factor is less than this, the image plane correction will
                        not be done. This function will return 0 in that case
    :type limit_frac: float
    :param sun_only: Other sources will be ignored when finding the leakage correction factor.
    :type sun_only: Boolean. Default: False
    '''
    me=measures()
    m = utils.get_sun_pos(msfile, str_output=False)
    solar_ra=m['m0']['value']*180/np.pi
    solar_dec=m['m1']['value']*180/np.pi
    
    head=fits.getheader(imagename+"-I-image.fits")
    Idata=fits.getdata(imagename+"-I-image.fits")
    freq=head['CRVAL3']*1e-6
    imwcs=WCS(head) 
    solar_xy = imwcs.all_world2pix(list(zip([solar_ra], [solar_dec],[head['CRVAL3']],[head['CRVAL4']])), 1)   
    
    if head['cunit1'] == 'deg':
        dx = np.abs(head['cdelt1'] * 60.)
    else:
        print(head['cunit1'] + ' not recognized as "deg". Model could be wrong.')
    if head['cunit2'] == 'deg':
        dy = np.abs(head['cdelt2'] * 60.)
    else:
        print(head['cunit2'] + ' not recognized as "deg". Model could be wrong.')                   
    sol_area_xpix = int(sol_area / dx)
    sol_area_ypix = int(sol_area / dy)
    
    src_area_xpix = int(src_area / dx)
    src_area_ypix = int(src_area / dy)
    
    solx=int(solar_xy[0][0])
    soly=int(solar_xy[0][1])
    
    model=imagename+"-"+stokes+"-model.fits"
    img=imagename+"-"+stokes+"-image.fits"
    
    model_data=fits.getdata(model)
    img_data=fits.getdata(img)
    
    
    head=fits.getheader(imagename+"-I-image.fits")
    obstime=Time(head['DATE-OBS'])
    
    diff_frac=[]
    weights=[]
    
    rms=np.nanstd(img_data)
    pos=np.where(abs(img_data)<10*rms)
    rms=np.nanstd(img_data[pos])
    
    if not sun_only:
    
        img_ra,img_dec,img_flux=get_good_sources(imagename+"-I-image.fits",imgcat)
        
        num_sources=len(img_ra)
        
        img_coord=SkyCoord(img_ra*u.degree,img_dec*u.degree,frame='icrs')   
        
        
        
        alt,az=convert_to_altaz(img_coord,obstime)
        
        
        
        
        freq=head['CRVAL3']*1e-6
        
        factors=get_beam_factors(alt,az,freq=freq)
        
        if stokes=='Q':
            factors=factors[:,1]
        elif stokes=='U':
            factors=factors[:,2]
        elif stokes=='V':   
            factors=factors[:,3]
        
        img_coord=SkyCoord(img_ra*u.degree,img_dec*u.degree,frame='icrs')  
        
        img_xy = imwcs.all_world2pix(list(zip(img_ra, img_dec,[head['CRVAL3']]*num_sources,[head['CRVAL4']]*num_sources)), 1)
        
        
                
        for i in range(num_sources):
            x=int(img_xy[i,0])
            y=int(img_xy[i,1])


            if x>sol_area_xpix//2-solx and x<sol_area_xpix//2+solx and \
                y>sol_area_ypix//2-solx and y<sol_area_xpix//2+soly:
                continue
            Ival=np.nanmax(Idata[0,0,y-src_area_ypix//2:y+src_area_ypix//2,\
                                    x-src_area_xpix//2:x+src_area_xpix//2])

            model_val=model_data[0,0,y,x]
            

            
            img_val1=np.nanmax(img_data[0,0,y-src_area_ypix//2:y+src_area_ypix//2,\
                                    x-src_area_xpix//2:x+src_area_xpix//2])
            img_val2=np.nanmin(img_data[0,0,y-src_area_ypix//2:y+src_area_ypix//2,\
                                    x-src_area_xpix//2:x+src_area_xpix//2])
            
            
            if img_val1>abs(img_val2):
                img_val=img_val1
            else:
                img_val=img_val2
            

            if abs(img_val)>thresh*rms and (not (abs(weights-Ival)<1e-3).any()):
                diff_frac.append(factors[i]-img_val/Ival)
                weights.append(Ival)
            
   

    pos=np.where(abs(img_data)<thresh*rms)
    img_data[pos]=np.nan
   
    
    rms=np.nanstd(Idata)
    pos=np.where(abs(Idata)<thresh*rms)
    Idata[pos]=np.nan
    
    
    
    solar_az,solar_el=utils.get_solar_azel(msfile)        
    factors=get_beam_factors([solar_el],[solar_az],freq=freq)
    
    if stokes=='Q':
        solar_model_frac=factors[0,1]
    elif stokes=='U':
        solar_model_frac=factors[0,2]
    elif stokes=='V':   
        solar_model_frac=factors[0,3]
    
    
    solar_obs_frac=np.nanmedian(img_data[0, 0, soly - sol_area_ypix // 2:soly + sol_area_ypix // 2 + 1,
                                            solx - sol_area_xpix // 2:solx + sol_area_xpix // 2 + 1]/
                                            Idata[0, 0, soly - sol_area_ypix // 2:soly + sol_area_ypix // 2 + 1,
                                            solx - sol_area_xpix // 2:solx + sol_area_xpix // 2 + 1])

    if not np.isnan(solar_obs_frac):
        diff_frac.append(solar_model_frac-solar_obs_frac)
        weights.append(np.nanmax(Idata[0, 0, soly - sol_area_ypix // 2:soly + sol_area_ypix // 2 + 1,
                                            solx - sol_area_xpix // 2:solx + sol_area_xpix // 2 + 1]))
    
    diff_frac=np.array(diff_frac)
    weight=np.array(weights)
    print (diff_frac)
    print (weights)
    frac=np.average(diff_frac,weights=weights)
    if abs(frac)<limit_frac:
        frac=0.0
    return frac
    
def get_conv_kernel(head):
    
    bmaj=head['BMAJ']
    bmin=head['BMIN']
    bpa=head['BPA']
    dx=abs(head['CDELT1'])
    
    
    sigma_major=bmaj/(2*np.sqrt(2*np.log(2)))
    sigma_minor=bmin/(2*np.sqrt(2*np.log(2)))
    
    sigma_major_pix=sigma_major/dx
    sigma_minor_pix=sigma_minor/dx
    
    theta=90-bpa  ### CASA angle increases clockwise
    
    kernel=Gaussian2DKernel(sigma_major_pix,sigma_minor_pix,theta=theta)
    return kernel
    
def correct_crosshand_phase_self(msname,crosshand_phase=0.0, inplace=False, outms=None):
    '''
    Here I apply the crosshand phase by hand. The crosshand phase suppliedto this function
    should be exactly equal to the crosshand_phase supplied to crosshand_phase_optimising_func.
    The way caltable is modified by that function is same as what is applied here.
    '''
    if outms is None:
        outms=msname.replace(".ms","_img_leak_corrected.ms")
    
    present=utils.check_corrected_data_present(msname)
    if present:
        datacolumn='CORRECTED'
    else:
        datacolumn='DATA'
        
    if not inplace:
        split(vis=msname,outputvis=outms,datacolumn=datacolumn)
        datacolumn='DATA'
        msname=outms
    
    if datacolumn=='CORRECTED':
        datacolumn='CORRECTED_DATA'
    
    tb=table()
    tb.open(msname,nomodify=False)
    try:
        data=tb.getcol(datacolumn)
        data[1,...]*=(np.cos(crosshand_phase)-1j*np.sin(crosshand_phase))
        data[2,...]*=(np.cos(crosshand_phase)+1j*np.sin(crosshand_phase))
        tb.putcol(datacolumn,data)
        tb.flush()
        success=True
    except:
        pass
    finally:
        tb.close()
    
    if not success:
        logging.warning("Crosshand phase correction "+\
                        "was not successfull. Please proceed with caution.")
    return msname
