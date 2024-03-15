import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from casatools import table, msmetadata
from matplotlib import gridspec
import sunpy.map as smap
from astropy.coordinates import SkyCoord
from astropy import units as u 
from suncasa.utils import plot_mapX as pmX
from matplotlib.patches import Ellipse
import base64
import io
import matplotlib.image as mpimg


# the functions to plot the data

njit_logo_str="iVBORw0KGgoAAAANSUhEUgAAAHgAAAA3CAMAAADwtH5ZAAAAVFBMVEVHcEzuNCTuNCTuNCTuNCTuNCTuNCTuNCTuNCTuNCTuNCTuNCTuNCTuNCTuNCTuNCTuNCTuNCTuNCTuNCTuNCTuNCTuNCTuNCTuNCTuNCTuNCTuNCQLSl2nAAAAG3RSTlMA2xyijg4I9TgExewsRlFwFPqvz7x7IuVcZJtcMgeUAAADGklEQVRYw+XY2ZKzKhAAYJRFFgUFFJT3f8/TGJPRaP6aVGXIxemam8HlSwNiI0L/z5i7w78t5/DH2b6NQUObW3t02Xw4GfVGQ/M+eMfPrrBhL4uOBKV8PNyLRmgMSpr6CEesQlA20kNz16RG5Qhw8BYDFhdwqg6/h9nU6OezmE8pzaeLpyql5jkbMzhvMV7skJJbMMY+pOUqYzh8uHZMoT2dhlMaulMrlSmF6alxIb2AyJknkgU2YXINJ7eXzAu4uYar+umG4/To8rSBfDzDaycm2R7g6RLmv4IRRSeY1pdwc5Tn873egtEZvgomBxJA9m1p2CetVc55Kg+jOfe23+RYBoZr4ak1O1kPVV8A7l1eGMT4I+vBlYMR+5HLwncZ7tIVg8224mTZ1nBBKXhEe7nnjaQF4Lq6w5uMdePpGy+JD8Cb7IZSMEFHOZWHkbitJO/AF68ytC1EMG7iJQxFxL48WGXPfgnDzEzqFRzTP2Gu0mESZ9mKX87qKVw239/rlylsr4gu35FwcZCv4VNdIjhZH8B4OsAYpXUubYKe6mma6r7vKWVM7OpHM8fZ6P0PE7O5LJDk87LCtIkQs1nHQNC65TqakWBrvZTVkOdpU93DSentQoxuX3e/uJxFL2com3gciZVVaFbsEMPQqFA5Jy1eCBkhTy7eqPphMMPFWLIaSCyrAwgUSNIDNJqoO95CV1OWC8/3Ax6N50KxbyOxlfohB1U5j4mJmcrSB7Y5UOTj3dQXPTfYPUwQJSZzx6fPaPv9VYMfM2tFm81UlV9gq1TTD4NbvrMbb/mKKRKp7qYlkdfs77aTbFzWecVaY7dnJHgw+z80tzShHykfvbo9mHbspr82HxvhmzpktBZlUCRas6rKEz2VQmEOaxxyqnhuabmvFIwTN6TGEV2XSxWSjVZltetLfpIR7ejSUFqF1/QSUrXosiqi2jbKxqnwd69+lo0bOSvM1sYFW7qLV9aR4skCK6UpPbL5E6T/Qh9DuWlxV7yPoVDFXxha2E0QwkV5tjak+wLL9PiFsUWoNZp+gUU81ugrQcU31P8AFQ9Mc+zu8kIAAAAASUVORK5CYII="
nsf_logo = "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"
ovrolwa_logo="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"
caltech_logo="iVBORw0KGgoAAAANSUhEUgAAALQAAABNCAMAAAA4jr7RAAAAS1BMVEVHcEz/bh7/bh7/bh7/bh7/bh7/bh7/bh7/bh7/bh7/bh7/bh7/bh7/bh7/bh7/bh7/bh7/bh7/bh7/bh7/bh7/bh7/bh7/bh7/bh5CVYR9AAAAGHRSTlMA2OBx/CBS9hhAtIcDyA7v5jQpYQimmEmEErTxAAAEtElEQVRo3u1Zi5KrKhAMoIAIIiDq/3/pBQZfiUnWbKr2VF26TtWuLGI7zPQ0ntutoKCgoKCgoKCgoKCgoKDgX8PATdMY272bx2nANMCFtPLqc+D+7guMpSUCacZ0r0b6ekHKqmpGHC6Iv0y6iffj+vec7djPK5iir5jQKszJpAkTH5AO9/e/J03RfIBuux+RJnq+TFp+h7QMz75D1Q7vSUsSHv9XkXZsoar1+it5T5qGydUfRdrkdEYtNaYhCq6wfUu6jfP+JtKdgOob0zLyxluIdvsj0n8UaVdBEm8PJ2nkqSj9A5HOgfZb4ck8RJcZtTHG1t0PIj1wa07ak0zD/J70UMdR+QHpJilHbw7BD6KHfBoamlFhHYDV2AwnkcY+oAGxJyJODe1p2msPdx71YRjl4Uy6c2k28nS4TDo9eR4PY/VIDMTFeraq4KxH/kg6/mF2cUNavLUnv5bx4NBuuF5IYyqqZWF/NVWGlAqMPmmU6ijfKYnuSc+JdC0OM1Heuq5l+2FlM2mGd6PiohHh6EXR5eTWSCGQxYo+kK5i7rgbF4vUA8lKpV2R7V3XCvwa9tDK3EXTkdio81edWNo9wztuoWtGe3QkrUKVhlwCcridgtKjTTIpy27GewTpQDbSGuVYzOJSWkvDXugWsOL7i+6etN8aVKUgJyBTUNg9Dumlpk5KThJDxDNpNhrObQs6YK+LZn7yI2mMez3ulQ7zc51Ov6Ll0RbnPU9TUyJnu8AEheYSfILcYsGaT0ifR7rjdZ3kdeANSeF7IA06DRElxy0KbzseqlyOPooePHIpPijLa6SNfpdTvHGjwDkPn0Q6rVKJcYGCVOrUsWC6Tafnpfbq/jppmxPtWb8MjWFX7NWTSE/V/AjMa3xiYiA99NLN0pyLpCEY2jzZB1HdEzmNtDvhPGtr9TFr9m3c/oK09PPJwtSl0Dd4NdpoHKvnOX1Kmhkg7V66vE9I38j8KNQh/IrUi2D1IhhtDinwJNJpiLXkCA651750eR+Rhkpk+3DI1BKQS6a1EgaqdHoR6cSjouftdpOmOlmuL5AG87FpbFwHItyOhxPMK50GduMmnDzdJcXxDNQy38hvkIZeHdvxyhleo7cp3dWw194npKEy9Bpq7vu2XnNvOV80IVv0WH+D9OCzMJB6SN9/srNrb34fJ9qfkwa1BI+B3QAeXKT06qA1zoykimmyUH2D9M0uhheH7uDV5tIgTsJ0AzfgEU5IV945YnKSMUGoIwLmxoMF+KgqDufPQe1X0iMkyPZ1aZVlbJYanbVSaPkwEj3sSprs/LTBD4oXzQW/M+Szqr9E+jahh242PZjharFjK2mjd4eA6Y61BkNk1MPZ4Euk96cq2E6T62lriCodQVh4GbrkyfZSbk3ZZYX1c6DdN9VU7d8iLQcq+mr9/uiWVtMRSHGtCOcKB5C4L+Fnctnhz5ox1qMJDrBKw+eIXji+sy8ChuMiycKn++vloB6XRebTb72u9UL4kTTdbSe4U2htzsS3CD61rnk8+cefMKUzzdSsnwE6Q2MnnOzRNA6Wroskn7vdH1nHq+H2OaQ8Mdby2grng/Ff+X+LgoKCgoKCgoKCgoKCgv8Z/gMgTqct1e1j7QAAAABJRU5ErkJggg=="

def inspection_bl_flag(ms_file):
    """
    Function to inspect the baseline flagging of the data
    
    :param ms_file: str : path to the measurement set file
    """
    tb = table()
    msmd = msmetadata()
    tb.open(ms_file)

    # Extract DATA Column
    visibility_data = tb.getcol('DATA')

    # Extract UVW Column
    uvw_data = tb.getcol('UVW')

    # extract ant1 and2 columns
    ant1 = tb.getcol('ANTENNA1')
    ant2 = tb.getcol('ANTENNA2')

    # extract location of antennas
    antenna_positions = msmd.antennaposition()
    antenna_names = msmd.antennanames()

    u_col = uvw_data[0, :]
    v_col = uvw_data[1, :]
    w_col = uvw_data[2, :]

    stokes_I = 0.5 * (visibility_data[0,:,:] + visibility_data[3,:,:])

    # Extract FLAG Column
    flag_data = tb.getcol('FLAG')


    # Close the table
    tb.close()

    img_cross = np.zeros((352, 352))

    for idx in range(stokes_I.shape[1]):
        # img_cross[ant1[idx], ant2[idx]] = np.mean(np.abs(stokes_I[:,idx]), axis=0)
        # insert flag_data
        img_cross[ant1[idx], ant2[idx]] = np.mean(np.abs(flag_data[0,:,idx]), axis=0)
        img_cross[ant2[idx], ant1[idx]] = np.mean(np.abs(flag_data[0,:,idx]), axis=0)

    fig_plt  = plt.imshow((img_cross), cmap='viridis', origin='lower', norm=mcolors.PowerNorm(0.5))
    return fig_plt



def slow_pipeline_default_plot(fname, 
            freqs_plt = [34.1, 38.7, 43.2, 47.8, 52.4, 57.0, 61.6, 66.2, 70.8, 75.4, 80.0, 84.5],
            fov = 7998,add_logo=True, apply_refraction_corr=False):
    """
    Function to plot the default pipeline output

    :param fname: str : path to the pipeline output file
    :param freqs_plt: list : list of frequencies to plot
    :param fov: float : field of view (default -3999 to 3999 arcsec)
    :param add_logo: bool : add logo to the plot
    :param apply_refraction_corr: bool : apply refraction correction to the plot
    """
    # Load the data

    from suncasa.io import ndfits
    meta, rdata = ndfits.read(fname)

    fig = plt.figure(figsize=(8, 6.5))
    gs = gridspec.GridSpec(3, 4, left=0.07, right=0.98, top=0.94, bottom=0.10, wspace=0.02, hspace=0.02)

    if True:
                freqs_mhz = meta['ref_cfreqs']/1e6
                for i in range(12):
                    ax = fig.add_subplot(gs[i])
                    freq_plt = freqs_plt[i]
                    ax.set_facecolor('black')
                    plt.setp(ax,xlabel='Solar X [arcsec]', ylabel='Solar Y [arcsec]')
                    ax.text(0.02, 0.98, '{0:.0f} MHz'.format(freq_plt), color='w', ha='left', va='top', 
                            fontsize=11, transform=ax.transAxes)                
                    plt.setp(ax.yaxis.get_majorticklabels(),
                              rotation=90, ha="center", va="center", rotation_mode="anchor")

                    if np.min(np.abs(freqs_mhz - freq_plt)) < 2.:
                        bd = np.argmin(np.abs(freqs_mhz - freq_plt)) 
                        bmaj,bmin,bpa = meta['bmaj'][bd],meta['bmin'][bd],meta['bpa'][bd]
                        beam0 = Ellipse((-fov/2*0.75, -fov/2*0.75), bmaj*3600,
                                bmin*3600, angle=(-(90-bpa)),  fc='None', lw=2, ec='w')

                        rmap_plt_ = smap.Map(np.squeeze(rdata[0, bd, :, :]/1e6), meta['header'])
                        rmap_plt = pmX.Sunmap(rmap_plt_)

                        if apply_refraction_corr:
                            # check if keyword is present
                            if 'refra_shift_x' in meta.keys():
                                com_x_corr = meta["refra_shift_x"][bd]
                                com_y_corr = meta["refra_shift_y"][bd]
                                rmap_plt.xrange = rmap_plt.xrange - com_x_corr*u.arcsec
                                rmap_plt.yrange = rmap_plt.yrange - com_y_corr*u.arcsec
                                 
                        vmaxplt = np.percentile(rdata[0, bd, :, :]/1e6, 99.9)
                        if np.isnan(vmaxplt):
                             vmaxplt = np.inf
                        im = rmap_plt.imshow(axes=ax, cmap='hinodexrt', vmin=0, vmax=vmaxplt)
                        # set background black

                        rmap_plt.draw_limb(ls='-', color='w', alpha=0.8)
                        ax.add_artist(beam0)

                        freq_mhz = meta['ref_cfreqs'][bd]/1e6
                        ax.text(0.99, 0.02, r"$T_B^{\rm max}=$"+ str(np.round(vmaxplt,2))+'MK', color='w', ha='right', va='bottom',
                                    fontsize=10, transform=ax.transAxes)
                        
                    else:
                        ax.text(0.5, 0.5, 'No Data', color='w', 
                                ha='center', va='center', fontsize=18, transform=ax.transAxes)
                    ax.set_xlim([-fov/2, fov/2])
                    ax.set_ylim([-fov/2, fov/2])
                    
                    

                    if i not in [8,9,10,11]: 
                        ax.set_xlabel('')
                        ax.get_xaxis().set_ticks([])
                    if i not in [0, 4, 8]:
                        ax.set_ylabel('')
                        ax.get_yaxis().set_ticks([])
                        

                    # add logo
                    if add_logo:
                        img1 = base64.b64decode(njit_logo_str)
                        img1 = io.BytesIO(img1)
                        img1 = mpimg.imread(img1, format='png')
                        img2 = base64.b64decode(caltech_logo)
                        img2 = io.BytesIO(img2)
                        img2 = mpimg.imread(img2, format='png')

                        ax_logo1 = fig.add_axes([0.015, 0.027, 0.07, 0.07])
                        ax_logo1.imshow(img1)
                        ax_logo1.axis('off')
                        ax_logo2 = fig.add_axes([0.005,-0.003, 0.09, 0.08])
                        ax_logo2.imshow(img2)
                        ax_logo2.axis('off')

                    # add figure title
                    fig.suptitle('OVRO-LWA '+ str(meta['header']['date-obs'])[0:22], fontsize=12)

    return fig