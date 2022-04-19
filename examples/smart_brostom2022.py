# -*- coding: utf-8 -*-
"""
Created on Wed May 20 11:54:37 2020
This script is a python version of the SMART algorithm, which was originally 
described in the paper:
    Joy, D. C. (2002). SMART - A program to measure SEM resolution and imaging 
    performance. Journal of Microscopy, 208(1), 24–34.

All user inputs are given in the "Controls" section, which includes image 
filename and directory as well as the size of the region of interest. The image
should be square. If it is not square, a square region should be selected as
ROI e.g. via img[500:800,500:800].   

When running the script, the user is prompted with an image window containing 
a slider. The user should adjust the slider to an optimal value, which segments 
the magnitude spectrum into a noise and signal region. Typically this means
identifying and estimating the size of the white circle or ellipse at the image
centre. Once the slider has been set, the user should press "q" in order to 
proceed. 
The algorithm will then finish and report the resolution and eccentricity based
on the dimensions of an ellipse fitted to the segmented magnitude spectrum.
Images of the results are displayed automatically.

The algorithm requires the standard packages numpy and matplotlib as well as 
the opencv package, which is freely available for download.

Please acknowledge and cite the paper titled: 
    "Assessing Resolution from Single SEM Images"
when using this code.

@author: Anders Brostrøm
"""

###############################################################################
##################################### Packages ################################
###############################################################################
""" 
Load the relevant packages needed to run the scripts 
"""

import numpy as np 
import matplotlib.pyplot as plt
import cv2 as cv2

###############################################################################
##################################### Controls ################################
###############################################################################
""" 
This section holds all the inputs for the SMART algorithm. Users should change
the inputs to specify where and which file is to be used for the analysis. 
Users should not alter the code in any other sections as it might affect the 
outcome.  
"""
# Directory containign the image file
direct = r"../data"

# Name of the image. Most extensions are supported
img_name = "Standard_Img.png"

# Set a size of the region of interest. It can be set to the full image size 
# or a sub region to optimize performance. Subregions are chosen around the 
# image centre
ROI_size = 800

###############################################################################
################################# Functions ###################################
###############################################################################
"""
Section containing the needed function to run the algorithm.
"""

def on_trackbar(val):
    """
    Function to generate and update the threshold slider in the displayed window
    """
    global dst1
    _,dst1 = cv2.threshold(Normed_mag,val,255,cv2.THRESH_BINARY)
    src2 = cv2.cvtColor(Normed_mag, cv2.COLOR_GRAY2RGB)
    dst2 = np.zeros_like(src2)
    dst2[:,:,2] = dst1
    dst = cv2.addWeighted(dst2, 0.25, src2, 0.75, 0.0)
    cv2.imshow("Threshold - press 'q' to proceed", dst)
    return val

###############################################################################
#################################### Code #####################################
###############################################################################
"""
This section contains the algorithm steps
"""
# Load image
img = cv2.imread(direct+"\\"+img_name,0)
    
# Cut out a ROI at the image centre, based on the specified size    
wid,hei = img.shape
centX = int(wid/2)
centY = int(hei/2)
ROI_size = int(ROI_size/2)
ROI = img[centX-ROI_size:centX+ROI_size,centY-ROI_size:centY+ROI_size]

# Perform FFT and shift high frequency components to the centre. 
f = np.fft.fft2(ROI)
fshift = np.fft.fftshift(f)

# Determine the magnitude spectrum, and normalize the image to a range from 0:255.
magnitude_spectrum = 20*np.log(np.abs(fshift))
Normed_mag = 255.*magnitude_spectrum/magnitude_spectrum.max()
Normed_mag = Normed_mag.astype(np.uint8)

# Code for the interactive threshold
cv2.namedWindow("Threshold - press 'q' to proceed")
cv2.imshow("Threshold - press 'q' to proceed", Normed_mag.astype(np.uint8))
cv2.createTrackbar("Threshold Val", "Threshold - press 'q' to proceed" , 0, 255, on_trackbar)

# Wait until user press some key
if cv2.waitKey() & 0xFF == ord('q'):
    cv2.destroyAllWindows()

# Opening and closing steps after thresholding. Necessary to remove unwanted noise
kernel = np.ones((2,2),np.uint8)
processed = dst1.copy()
processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)

# Find contours in the resulting binary image and select the largest, if several are present
contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_TC89_L1)
if len(contours)>0:
    areas = []
    for c in contours:
        areas = areas + [float(cv2.contourArea(c))]
        ix = areas.index(max(areas))
    contour = contours[ix]
else:
    contour = contours

# Fit an ellipse to the larget contour in the image and display it in red
ellip = cv2.fitEllipse(contour)
final = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
final = cv2.ellipse(final, ellip, (255,0,0), 2)

# Calculate the image resolution and eccentricity of the fitted ellipse
img_res = float(ROI_size)/np.mean(ellip[1])
eccentricity = (max(ellip[1])-min(ellip[1]))/max(ellip[1])

###############################################################################
################################ Plotting #####################################
###############################################################################
"""
This section plots and prints the results of the algorithm
"""

# Display the results
fig, axs = plt.subplots(nrows=2,ncols=2,sharex=True,sharey=True,figsize=(15,15))
axs[0,0].imshow(ROI,cmap="gray")
axs[0,0].set_title("ROI")
axs[0,1].imshow(Normed_mag,cmap="gray")
axs[0,1].set_title("Magnitude Spectrum")
axs[1,0].imshow(dst1,cmap="gray")
axs[1,0].set_title("Binary Magnitude spectrum")
axs[1,1].imshow(final,cmap="gray")
axs[1,1].set_title("Fitted ellipse")
plt.tight_layout()

# Print the results to the console
print("The image resolution is {0} pixels".format(img_res))
print("The stigmatic error is {0}".format(eccentricity))







