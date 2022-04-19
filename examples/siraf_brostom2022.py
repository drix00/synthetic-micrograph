# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 15:09:02 2019
This code can be used to assess the spatial resolution of SEM images (and 
potentially also for STEM, HAADF STEM, TEM, and optical images), using a 
Fourier based algorithm. The procedure used in the script is described
in detail in the paper: "Assessing Resolution from Single SEM Images" written
by Anders Brostrøm and Kristian Mølhave from DTU, Denmark.

The code is commented throughout, describing the function of either lines or 
segments of code, in order to make it more readable.

The code requires numpy, matplotlib, scipy, and the opencv (CV2) packages.

All user inputs are given in the "Controls" section, which include image 
filename and directory.

When running the code, both the fitted Gaussian FWHM and sigma values are 
printed in the consol, and the fits are plotted for user assessment. The code
can also produce an interactive plot, where users may adjust the sigma, beta,
and c values manually in order to optimize the fit.

Please acknowledge and cite the paper titled: 
    "Assessing Resolution from Single SEM Images"
when using this code.
 
@author: Anders Brostrøm
"""

###############################################################################
##################################### Controls ################################
###############################################################################
""" 
This section holds all the inputs for the algorithm. Users should change
the inputs to specify where and which file is to be used for the analysis. 
Users should not alter the code in any other section as it might affect the 
outcome  
"""

# Specify the directory of the image
direct = r"../data"

# Specify the name of the image file with its extension
# Accepted extensions: 
#    - Windows bitmap (bmp)
#    - Portable image formats (pbm, pgm, ppm)
#    - Sun raster (sr, ras)
#    - JPEG (jpeg, jpg, jpe)
#    - JPEG 2000 (jp2)
#    - TIFF files (tiff, tif)
#    - Portable network graphics (png)
img_name = "Standard_Img.png"

# Set to 1 if a Hanning window is to be applied in the algorithm. This window
# can be used to elimate the cross in FFT images
Hanning_window = 1

# Set to 1 if the algorithm should make a plot of the individual steps, including
# Hanning corrected windows (if opted), amplitude spectrum,     and fit to image data
Plot_please = 1

# Specify whether the radial averaged image data, should be fitted with weights.
# This can account for fluctuations in the individual data points during the fitting
# procedure. The data can either be unweighted, weighted by the std, or the
# standard error of the mean by specifying:
# "std" for standard deviation
# "std_err" for standard error of the mean
# or anything else for no weights
# The errors used will be displayed on the radial mean plot, if plot_please is 
# set to 1
weight_err = "std"

# Specify if the fitting window is to be limited to frequencies between 0.01 
# and 0.5. This can be advantageous in cases where a strong Gibbs effect is
# present, causing a sharp drop at frequencies higher than 0.5. Or cases with
# limited low frequency responses, meaning that the added frequencies from the
# Hanning window can influence the results significantly.
# Set to 1 for on or 0 for off
Hanning_limit = 0
Gibbs_limit = 0

# If the automated fitting procedures are unsuccesfull at fitting to the image
# data, then setting interactive to 1 will allow you to do a manual fit. 
interactive = 1

# By default, the limits of the three variables in the manual fitting procedure
# is 0 and 10. If it is not possible to make a reasonable fit within these limits,
# you can set new ones below.  
c_lim = (0,10)
beta_lim = (0,10)
sigma_lim = (0,10)

###############################################################################
##################################### Packages ################################
###############################################################################
""" 
Load needed packages to run the script
"""
import matplotlib.pyplot as plt
import cv2 as cv2
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import sem
from matplotlib.widgets import Slider, Button
import time

plt.close("all")
plt.rcParams['xtick.labelsize'] = 30
plt.rcParams['ytick.labelsize'] = 30

###############################################################################
################################# Functions ###################################
###############################################################################
"""
Section containing the needed function to run the algorithm.
"""
#----------------------------------------------------------------------------#

def dist_matrix_Elip(test_img):
    """
    Function used to generate a matrix containing distance measures for all
    coordinates (x,y) relative to the image center. This is used when binning 
    pixels according to distance from the center of the amplitude spectrum 

    Parameters
    ----------
    test_img : TYPE - numpy.ndarray
               DESCRIPTION - Image from which to generate a distance map

    Returns
    -------
    distances : TYPE - numpy.ndarray
                DESCRIPTION - Distance map with same dimensions as input image

    """
    height, width = test_img.shape
    maxs = ((height-1),(width-1)) 
    grid_x, grid_y = np.mgrid[0:height, 0:width]
    grid_x = grid_x/maxs[0] - 0.5
    grid_y = grid_y/maxs[1] - 0.5
    distances = np.hypot(grid_x, grid_y)
    return distances

#----------------------------------------------------------------------------#

def func_fit(x,sigma,b,c): 
    """
    Function derived from the convolution of a step function and a Gaussian 
    function in Fourier space. The function is used to fit radial averaged image 
    data

    Parameters
    ----------
    x : TYPE - numpy.ndarray
        DESCRIPTION - Radii of the radial image data.
    sigma : TYPE - float
            DESCRIPTION - Initial guess for sigma value of Gaussian function
    b : TYPE - float 
        DESCRIPTION - Initial guess for parameter b.
    c : TYPE - float
        DESCRIPTION - Initial guess for parameter c.

    Returns
    -------
    Fitted_Values : TYPE - numpy.ndarray
                    DESCRIPTION - the fitted values

    """
    Fitted_Values = c+ np.log(1+(b/(x*np.sqrt(2*np.pi)))*np.e**(-2*(np.pi*sigma*x)**2))
    return Fitted_Values

#----------------------------------------------------------------------------#

def Img_alg(img1,hanning=1,plots=0,weight_err=None,Han_lim = 1,Gibb_lim = 1):
    """
    This function is the main function of the algorithm. It treats the
    image data to give the radially averaged data necessary for the fitting
    procedure, and performs the fit as well using the scipy.optimize function.
    
    Parameters
    ----------
    img1 : TYPE - numpy.ndarray, dtype=numpy.uint8
           DESCRIPTION - Image to analyse.
    hanning : TYPE - Boolean, optional
              DESCRIPTION - Set whether to apply a hanning filter to the image 
                            to minimize horisontal and vertical white lines in 
                            the image fft.  
                            The default is 1.
    plots : TYPE - Boolean, optional
            DESCRIPTION - Set whether to display image, amplitude spectrum and 
                          radial average plot. 
                          The default is 0.
    weight_err : TYPE - str, optional
                 DESCRIPTION - Set whether to use weights in the fitting 
                               procedure. Can be set to:
                               "std"    : Std of each radial mean will be used 
                                          as weight when fitting
                               "std_err": Standard erro of the mean for each 
                                          radial mean will be used as weight 
                                          when fitting

    Returns
    -------
    popt : TYPE - numpy.ndarray
           DESCRIPTION - Fitted parameters sigma, b, and c.
    errs : TYPE - numpy.ndarray
           DESCRIPTION - Estimated errors of sigma, b, and c, determined from covariance.
    img2 : TYPE - numpy.ndarray
           DESCRIPTION - The original image
    Amplitude_spectrum : TYPE - numpy.ndarray
                         DESCRIPTION - The determined amplitude spectrum.
    Means : TYPE - numpy.ndarray
            DESCRIPTION - Radial means of amplitude spectrum
    Radii : TYPE - numpy.ndarray
            DESCRIPTION - Radii of each radial mean circle
    
    """
    # Generate a matrix of distances from image center for all coordinates (x,y)
    distances = dist_matrix_Elip(img1)   
    
    # Set the step width in the radial binning procedure 
    Step = 1/min(img1.shape)    
    
    # Generate an array of radii for the radial distance binning
    Radii = np.arange(Step,distances.max(),Step)
    
    # Get image shape
    n,n1 = img1.shape
    
    # Generate a Hanning window if opted by the user
    if hanning:
        h = np.hanning(n)    
        h1 = np.hanning(n1)    
        han2d = np.sqrt(np.outer(h,h1))
        
        # Apply Hanning window
        img2 = img1*han2d    
    else:
        # If a Hanning window is not opted, use the original image
        img2 = img1

    # Determine the shifted amplitude spectrum via FFT analysis. 
    # The log(1+..) step is used to scale the image data, resulting in more robust
    # fits
    Amplitude_spectrum = np.fft.fftshift(np.log(1+np.abs(np.fft.fft2(img2,norm="ortho"))))
    
    # Calculate the radial means of the image using the distance bins
    Means = np.array([np.nanmean(Amplitude_spectrum[(distances >= r) & (distances < r+Step)]) for r in Radii])
    
    # To account for potential errors while averaging, NaN values are set to 0
    Means[np.isnan(Means)]=0
    
    # Setup initiation values and parameter bounds for the fitting procedure
    alpha_init = Means[0]/2
    beta_init = 3
    if Gibb_lim:
        c_init = Means[(Radii>0.4) & (Radii<0.5)].mean()
    else:
        c_init = Means[(Radii>0.6) & (Radii<0.7)].mean()
    param_bounds=([0,-np.inf,c_init/2],[max([n,n1]),np.inf,c_init*2])
    
    # Calculate the standard deviation (Std) and the standard error of the mean (Std_err)
    Std = np.array([np.std(Amplitude_spectrum[(distances >= r) & (distances < r+Step)]) for r in Radii])
    Std_err = np.array([sem(Amplitude_spectrum[(distances >= r) & (distances < r+Step)]) for r in Radii])
    
    # If specified, set limits for the fitting window to exclude potential
    # Gibbs lines or influence from the Hanning window at low requencies.    
    if Gibb_lim:
        High_lim = 0.5
    else: 
        High_lim = 0.5
    if Han_lim:
        Low_lim = int(20/min(img1.shape))
    else:
        Low_lim = 0
    
    # Fit the image data using the function derived from the convolution of a 
    # step and Gaussian function in FFT space. The image data can fitted to 
    # unweighted data or to data, which is weighted by the std or the standard 
    # error of the mean, depending on user specification
    
    if weight_err=="std":
        popt, pcov = curve_fit(func_fit,Radii[(Radii>Low_lim) & (Radii<High_lim)],Means[(Radii>Low_lim) & (Radii<High_lim)],
                   p0=np.array([alpha_init,beta_init,c_init]),bounds=param_bounds,sigma=Std[(Radii>Low_lim) & (Radii<High_lim)])
    elif weight_err=="std_err":
        popt, pcov = curve_fit(func_fit,Radii[(Radii>Low_lim) & (Radii<High_lim)],Means[(Radii>Low_lim) & (Radii<High_lim)],
                   p0=np.array([alpha_init,beta_init,c_init]),bounds=param_bounds,sigma=Std_err[(Radii>Low_lim) & (Radii<High_lim)])
    else:
        popt, pcov = curve_fit(func_fit,Radii[(Radii>Low_lim) & (Radii<High_lim)],Means[(Radii>Low_lim) & (Radii<High_lim)],
                   p0=np.array([alpha_init,beta_init,c_init]),bounds=param_bounds)
    
    # Calculate errors of the fitting procedure
    errs = np.sqrt(np.diag(pcov))
    
    # If specified by the user, plots are generated to illustrate the process
    # and the results
    if plots:
        # Generate figures for plotting
        fig = plt.figure(tight_layout=True)
        gs = fig.add_gridspec(2, 2)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1,:])
        
        # Display Image
        ax1.imshow(img2,cmap="gray")
        ax1.set_axis_off()
        
        # Display amplitude spectrum
        ax2.imshow(Amplitude_spectrum,cmap="gray")
        ax2.set_axis_off()
        
        # Display image data as radial average plot
        ax3.plot(Radii,Means,color="blue",label="Image data",lw=4)
        
        # Display error weights if any
        if weight_err=="std":
            ax3.fill_between(Radii,Means-Std,Means+Std,color="blue",alpha=0.5,label="Standard deviation")
        elif weight_err=="std_err":
            ax3.fill_between(Radii,Means-Std_err,Means+Std_err,color="blue",alpha=0.5,label="Standard error of the mean")
        
        # Plot the fit
        ax3.plot(Radii,func_fit(Radii,popt[0],popt[1],popt[2]),color="red",ls="--",label="Fit: FWHM = ${:.2f} \pm {:.2f}$".format(2.355*popt[0],2.355*errs[0]),lw=4)
        
        if Gibb_lim:
            ax3.axvspan(0.5,0.707,color="k",alpha=0.5)
        if Han_lim:
            ax3.axvspan(0,0.01,color="k",alpha=0.5)
        
        # Set axis labels, legend, and axis limits
        ax3.legend(fontsize=25)
        ax3.grid(which="both",axis="both")
        ax3.set_xlim(0,0.707)
        ax3.set_xlabel("Normalized Frequency",fontsize=25)
        ax3.set_ylabel("Amplitude",fontsize=25)
        
    # Return results from the algorithm.
    return popt, errs, img2, Amplitude_spectrum, Means, Radii

###############################################################################
#################################### Code #####################################
###############################################################################
"""
The image is loaded and the algorithm is called and executed
"""
t1 = time.time()

# Load the image specified by the user
img = cv2.imread(direct+"\\"+img_name,0) # Load the image

# Call the algorithm and perform the analysis
fitparams, fit_errs, img2, Amplitude, Means, Radii = Img_alg(img,Hanning_window,Plot_please,weight_err,Hanning_limit,Gibbs_limit) # Run the algorithm

# Print the determined sigma and FWHM of the Gaussian Point Spread function 
print("-----------------------------")
print("Sigma: {:.2f} +/- {:.2f} pixels".format(fitparams[0],fit_errs[0]))
print("FWHM: {:.2f} +/- {:.2f} pixels".format(2.355*fitparams[0],2.355*fit_errs[0]))
print("-----------------------------")

t2 = time.time()
print("SIRAF algorithm Brostom 2022 took " + str(t2 - t1) + " seconds")

###############################################################################
############################# Interactive Code ################################
###############################################################################
"""
Code related to the interactive plot, which allows for a manual fit
"""

if interactive:
    
    # Create the figure with a title, axis labels, and a grid
    fig, ax = plt.subplots()
    plt.title(r"$ c+log \left( 1+ \frac{ \sigma \beta }{ k \sqrt{ 2 \pi } } e^{ -2( \pi \sigma k)^{ 2 } } \right) $", fontsize = 20)
    plt.grid(which="both",axis="both")
    ax.set_xlabel('Normalized frequency',fontsize=25)
    ax.set_ylabel('Amplitude',fontsize=25)
    
    # Plot the image data we are fitting to, and the initial fit produced from
    # the automated fitting procedures
    plt.plot(Radii,Means,color="blue",label="Image data",lw=3)
    line, = plt.plot(Radii, func_fit(Radii, fitparams[0], fitparams[1], fitparams[2]), lw=3,label="Fit",color="red",ls="--")
    
    # Make the legend and set the margin to 0
    plt.legend(fontsize=20)
    ax.margins(x=0)
    
    # adjust the plot window to make room for the sliders
    plt.subplots_adjust(bottom=0.35)
    
    # Make the slider to control the sigma value in the fit.
    axsigma = plt.axes([0.1, 0.05, 0.65, 0.03], facecolor="white")
    sigma_slider = Slider(
        ax=axsigma,
        label=r"$\sigma$",
        valmin=sigma_lim[0],
        valmax=sigma_lim[1],
        valinit=fitparams[0],
    )
    
    # Make the slider to control the beta value in the fit.
    axbeta = plt.axes([0.1, 0.1, 0.65, 0.03], facecolor="white")
    beta_slider = Slider(
        ax=axbeta,
        label=r"$\beta$",
        valmin=beta_lim[0],
        valmax=beta_lim[1],
        valinit=fitparams[1],
    )
    
    # Make the slider to control the c value in the fit.
    axasym = plt.axes([0.1, 0.15, 0.65, 0.03], facecolor="white")
    asym_slider = Slider(
        ax=axasym,
        label="c",
        valmin=c_lim[0],
        valmax=c_lim[1],
        valinit=fitparams[2],
    )
    
    # The function to be called anytime a slider's value changes
    def update(val):
        line.set_ydata(func_fit(Radii, sigma_slider.val, beta_slider.val, asym_slider.val))
        fig.canvas.draw_idle()
    
    
    # register the update function with each slider
    sigma_slider.on_changed(update)
    beta_slider.on_changed(update)
    asym_slider.on_changed(update)
    
    # Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', color="white", hovercolor='0.975')
    
    # Define the reset function
    def reset(event):
        sigma_slider.reset()
        beta_slider.reset()
        asym_slider.reset()
        
    # register the reset function to the button
    button.on_clicked(reset)
    
    # Show the plot
    plt.show()  
