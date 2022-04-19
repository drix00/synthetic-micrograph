# -*- coding: utf-8 -*-
"""
Created on Tue May 19 09:45:56 2020
This code can be used to simulate a SEM image with specified parameters. The
procedure used in the script was based on the work of Cizmar et al. (2008)
"Simulated SEm Image for Resolution Measurement".

The code is commented throughout, describing the function of either lines or
segments of code in order to guide potential readers.

In order to use the script, only parameters in the "Controls" section should
be changed.

Please acknowledge and cite the paper titled:
    "Assessing Resolution from Single SEM Images"
when using this code.

@author: Anders Brostrøm
"""

###############################################################################
##################################### Packages ################################
###############################################################################
""" 
Load needed packages to run the script
"""
import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv2
import time

plt.close("all")

###############################################################################
##################################### Controls ################################
###############################################################################
""" 
This section holds all the inputs for the algorithm. Users should change
the inputs to specify how the simulated image should look like. Users should 
not alter the code in any other section as it might affect the outcome.  
"""

img_size = (800, 800)  # Specify the desired size of the simulated image. Default = 800x800

""" Particle variables """
number_of_particles = 10000  # Approximate number - some will be removed due to overlap. Default = 10000

# Individual particle sizes are chosen at random between specified limits
Max_particle_size = 40  # Largest possible particle size in pixels. Default = 40
Min_particle_size = 2  # Smallest possible particle size in pixels. It cannot be
# smaller than 2 and it must be smaller than Max_particle_size. Default = 2

# Individual particle intensities are chosen randomely as: Particle_intensity +/- Particle_intesnity_Var
Particle_intensity = 150  # Average intensity of particles. Default = 150
Particle_intesnity_Var = 50  # Variation in particle intensity. Default = 50

# Intensity structures of the substrate and particles are set via reverse FFT.
# The roughness of the structure is controlled here. A value of the image size
# gives small detailed structure, while values close to 0 gives smooth
# intensity structures
Particle_structure_size = 300  # Internal particle structure. Cannot exceed image size and must be positive. Default = 300
Structure_intensity_P = 50  # Intensity range in particle structures (Particle_intensity +/- Structure_intensity_P). Default = 50.
Structure_degree_P = 0.5  # Clarity of particle structure. Scales from 0:1. Default = 0.5
Edge_On = 1  # Set to 1 to turn edge effect on or 0 for off. Default = 1
Edge_intensity = 240  # Intensity at particle edge. Default = 240
Edge_width = 4  # Width of particle edge. Default is 4
Edge_steepness = 0.9  # Steepness of the white edge effect. Typically varies from 1 to -5. Default = 0.9

""" Background settings """
Background_intensity = 50  # Average background intensity. Default = 50
Background_structure = 500  # Background structures. Cannot exceed imgsize and must be positive. Default = 500
Structure_intensity_B = 30  # Intensity range in background structures (Background_intensity +/- Structure_intensity_B). Default = 30
Structure_degree_B = 1  # Weight background structure vs average intensity. Has to be between 0 and 1. Default = 1

""" Focus and astigmatism parameters """
sigma = 1  # Defocus/blurring aka sigma of Gaussian PSF. Default = 1
astigmatism = 1  # Astigmatism in the image. 1 is no astigmatism. Default = 1
angle = 0  # Direction of astigmatism. Default = 0

""" Noise level """
Noise_degree = 0.4  # Sets the degree of Poisson noise, scaling from 0:1. Default = 0.4

""" Drift and Vibration effects """
Vibration = 1  # Set to one to turn vibration effects on or 0 for off
Max_vibration_shiftX = 2  # Largest pixel shift in x direction. Default = 2
Max_vibration_shiftY = 0.5  # Largest pixel shift in y direction. Default = 0.5
Shift_OccurenceX = 10  # Shift occurence in x direction. Low values gives few shifts. Default = 10
Shift_OccurenceY = 10  # Shift occurence in y direction. Low values gives few shifts. Default = 10

""" Display images """
show_steps = 1  # Set to 1 if images from each step should be displayed

""" Save Image and/or Process """
Save_img = True
Direct = r"../data"
ImgName = "Standard_Img.png"

Save_Process = True
Direct1 = r"../data"
ImgName1 = "Process.png"

rng = np.random.default_rng(12345)

###############################################################################
################################# Functions ###################################
###############################################################################
"""
Section containing the needed function to simulate the image.
"""


def makeParticle(x0, y0, r, a1, a2, f1, f2):
    """
    Function that determines particle coordinates from an initial center
    coordinate and a set of deformation parameters. x0 and y0 are the center
    coordinates of the initial circle, r is the radius of the initial circle.
    The other parameters are deformation parameters.
    Requirements:
    0 < a1 < 0.45
    0 < a2 < 0.1
    0 < f1, f2, delta < 2pi
    """
    if r > 50:
        angles = np.arange(0, 2 * np.pi, 0.001)
    else:
        angles = np.arange(0, 2 * np.pi, 0.01)
    delta = 1 + a1 * np.sin(2 * angles + f1) + a2 * np.sin(3 * angles + f2)
    x = np.array([x0 + r * delta * np.cos(angles)])
    y = np.array([y0 + r * delta * np.sin(angles)])
    x = np.round(x).astype(int)[0]
    y = np.round(y).astype(int)[0]
    return x, y


def Blurr_Filter(x, y, s, phi, sig):
    """
    Filter used to produce blur and astigmatism in the image.
    """
    x1 = s * (x * np.cos(phi) + y * np.sin(phi))
    y1 = (1 / s) * (-x * np.sin(phi) + y * np.cos(phi))
    P = (1 / (2 * np.pi * sig ** 2)) * np.e ** (-(x1 ** 2 + y1 ** 2) / (2 * sig ** 2))
    return P


def pad_with(vector, pad_width, iaxis, kwargs):
    """
    Function to pad the image, which is used when producing random structures
    in the background and on particles.
    """
    pad_value = kwargs.get('padder', 10)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value


def image_convolve_mask(image, params, P_inten, Edge, E_steep, E_width):
    """
    Function to apply edge effects on existing particles.
    """
    for param in params:
        E_inten = P_inten + Edge
        if E_inten > 255:
            E_inten = 255
        x = param[0]
        y = param[1]
        r = param[2]
        a1 = param[3]
        a2 = param[4]
        f1 = param[5]
        f2 = param[6]
        for i in range(E_width):
            # Find the edge coordinates of existing particles
            x1, y1 = makeParticle(x, y, r - i, a1, a2, f1, f2)
            # Set the color of particle edges to E_inten and calculate new
            # intensity values for particle pixels when moving inwards untill E_width is reached
            if (max(y1) - min(y1)) == 0:
                y2 = y1
            else:
                y2 = (y1 - min(y1)) / (max(y1) - min(y1))
            initial_part = (P_inten - E_inten) / (np.e ** (-E_steep * E_width) - 1)
            C = initial_part * (np.e ** (-E_steep * i) - 1) + E_inten
            image[x1, y1] = C * y2 + P_inten * (1 - y2)
    return image


def Structure_Generation(Structure_Size, Mean_intensity, degree, struct_inten, contours_P=0, var=0, grain_params=0):
    """
    Function to apply structure inside particles and in the background. The
    structure is generated from a noisy image which is FFT transformed, padded,
    and then reverse transformed to real space. This produces a realistic
    structure, which be controlled from the size of the intial noise image
    relative to the size of the padding.
    """
    # Produce image of random noise with specified size
    Grain_Noise_Matrix = rng.integers(0, 255, size=(Structure_Size, Structure_Size))

    # FFT convert the noise image
    fft = np.fft.fftshift(np.fft.fft2(Grain_Noise_Matrix))

    # pad the image
    pad_width = int((w - Structure_Size) / 2)
    pad_height = int((h - Structure_Size) / 2)
    padded = np.pad(fft, ((pad_height, pad_height), (pad_width, pad_width)), pad_with, padder=0)

    # Reverse FFT
    back = np.abs(np.fft.ifft2(np.fft.ifftshift(padded)))

    # Normalize the resulting image to ensure intensities between 0-255
    structure = cv2.normalize(back, _, Mean_intensity - struct_inten, Mean_intensity + struct_inten,
                              cv2.NORM_MINMAX).astype(np.uint8)
    structure[structure > 255] = 255

    # Apply the calculated structure to particles or background
    if contours_P:
        Average_Map = np.zeros_like(structure, dtype=np.uint8)
        for j, i in enumerate(contours_P):
            if var:
                intensity = rng.integers(Mean_intensity - var, Mean_intensity + var, endpoint=True)
                grain_params[j] = grain_params[j] + [intensity]
            else:
                intensity = Mean_intensity
            if intensity < 0:
                intensity = 0
            elif intensity > 255:
                intensity = 255
            Average_Map = cv2.drawContours(Average_Map, [i], -1, int(intensity), -1)
    else:
        Average_Map = np.ones_like(structure, dtype=np.uint8) * Mean_intensity
    Final_structure = cv2.addWeighted(Average_Map, 1 - degree, structure, degree, 0)
    return Final_structure, grain_params


###############################################################################
#################################### Code #####################################
###############################################################################
"""
In this section the algorithm is called and carried out
"""
print("------------ Simulating ------------")
t1 = time.time()
# Generate a black image, which is slightly larger then the specifed. The image
# will be cut later so particles can also lie on the edge
h = img_size[0] + 40
w = img_size[1] + 40
img = np.zeros([h, w], dtype=np.uint8)

print("Generating Particles")
# Generate random coordinates and deformation parameters to be used when making particles

rand_x = [rng.integers(0, h - 1) for i in range(number_of_particles)]
rand_y = [rng.integers(0, w - 1) for i in range(number_of_particles)]
rand_r = [rng.integers(Min_particle_size, Max_particle_size) for i in range(number_of_particles)]
rand_a1 = [rng.uniform(0, 0.45) for i in range(number_of_particles)]
rand_a2 = [rng.uniform(0, 0.1) for i in range(number_of_particles)]
rand_f1 = [rng.uniform(0, 2 * np.pi) for i in range(number_of_particles)]
rand_f2 = [rng.uniform(0, 2 * np.pi) for i in range(number_of_particles)]
coords = zip(rand_x, rand_y, rand_r, rand_a1, rand_a2, rand_f1, rand_f2)

# Make lists to contain relevant parameters
Grain_params = []
x_mins = []
x_maxs = []
y_mins = []
y_maxs = []
n = 0
k = 0

print("Checking Particle Parameters")
# Loop to generate particle contours on the black image
for x, y, r, a1, a2, f1, f2 in coords:
    # Generate particle contours
    x1, y1 = makeParticle(x, y, r, a1, a2, f1, f2)
    # In case edge effect is turned on, the required distance between particles is increased.
    if Edge_width:
        Edge_distance = Edge_width
    else:
        Edge_distance = 1
    # If new particle contours are on the edge of the image, they are not drawn
    if np.any(x1 < Edge_distance) or np.any(y1 < Edge_distance) or np.any(x1 > h - (Edge_distance + 1)) or np.any(
            y1 > w - (Edge_distance + 1)):
        continue
    # If the edge of new particle contours touch existing contours they are not drawn
    elif (255 in img[x1, y1]) or (255 in img[x1 - 1, y1]) or (255 in img[x1 + 1, y1]) or (255 in img[x1, y1 + 1]) or (
            255 in img[x1, y1 - 1]):
        continue
    # If new particle contours are inside existing contours they are not drawn
    elif n != 0:
        k = 0
        for j in range(len(x_mins)):
            if (x1.min() >= x_mins[j]) & (x1.max() <= x_maxs[j]) & (y1.max() <= y_maxs[j]) & (y1.min() >= y_mins[j]):
                k += 1
            elif (x1.min() <= x_mins[j]) & (x1.max() >= x_maxs[j]) & (y1.max() >= y_maxs[j]) & (y1.min() <= y_mins[j]):
                k += 1
    if k != 0:
        continue
    # Approved particle contours are drawn and their parameters are
    # stored for later use.
    img[x1, y1] = 255
    Grain_params = Grain_params + [[x, y, r, a1, a2, f1, f2]]
    x_mins = x_mins + [x1.min()]
    x_maxs = x_maxs + [x1.max()]
    y_mins = y_mins + [y1.min()]
    y_maxs = y_maxs + [y1.max()]
    n += 1

print("Drawing Particles")
# The drawn particle contours are located and pixels inside are set to 255
contours1, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
img1 = cv2.drawContours(img.copy(), contours1, -1, 255, -1)

# Particle contours are located again after they have been filled
contours, _ = cv2.findContours(img1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# Generate masks for particles and background
Particle_mask = img1.astype(np.uint8)
Background_mask = ~Particle_mask

# Generate intensity structure inside particles
Particle_structures, Grain_params = Structure_Generation(Particle_structure_size, Particle_intensity,
                                                         Structure_degree_P, Structure_intensity_P, contours,
                                                         Particle_intesnity_Var, Grain_params)
# Generate intensity structure in the background
Background_structures, _ = Structure_Generation(Background_structure, Background_intensity, Structure_degree_B,
                                                Structure_intensity_B)

# Apply the particle structure to the image
img2 = img1.copy()
img2[Particle_mask == 255] = Particle_structures[Particle_mask == 255]

# Edge effects is applied if specified
if Edge_On:
    print("Applying Edge Effect")
    img3 = image_convolve_mask(img2.copy(), Grain_params, Particle_intensity, Edge_intensity, Edge_steepness,
                               Edge_width)
else:
    img3 = img2.copy()

print("Applying Background Structure")
# Background structure is applied
img3[Background_mask == 255] = Background_structures[Background_mask == 255]

# Generate kernel to add blur and astigmatism
if sigma:
    kern_size = int(sigma * 5)
    kern = np.zeros((kern_size, kern_size))
    for j, i in enumerate(kern):
        for k, l in enumerate(i):
            kern[j, k] = Blurr_Filter(j - ((kern_size - 1) / 2), k - ((kern_size - 1) / 2), astigmatism, angle, sigma)

    # Apply blur and astigmatism
    img4 = cv2.filter2D(img3, -1, kern).astype(np.uint8)
else:
    img4 = img3.copy()

print("Applying Vibrational Effects")
# Apply vibration and drift artefacts
img5 = img4.copy()
if Vibration:
    Npixels = img5.ravel().shape[0]
    for i in range(10, h - 10):
        Ax = Max_vibration_shiftX * rng.random(1)
        for j in range(10, w - 10):
            pixel_id = i * h + j
            Ay = Max_vibration_shiftY * rng.random(1)
            # Produce shifts in x direction
            xv = Ax * np.sin(Shift_OccurenceX * (pixel_id / Npixels))
            # Produce shifts in y direction
            yv = Ay * np.sin(Shift_OccurenceY * (pixel_id / Npixels))
            # Apply shifts to the image
            img5[i, j] = img4[i + int(xv), j + int(yv)]

print("Adding Noise")
# Calculate poisson noise of the image
img6 = rng.poisson(img5)
# In case some intensities are above 255 they are lowered to 255
img6[img6 > 255] = 255
img6 = img6.astype(np.uint8)
# The noise is added to the image based on specified weights
img7 = cv2.addWeighted(img6, Noise_degree, img5, 1 - Noise_degree, 0)

# Crop image so particles can be cut at the frame
img8 = img7[20:h - 20, 20:w - 20]

# If specified the individual steps of the algorithm are displayed and saved
if show_steps:
    fig, ax = plt.subplots(frameon=False)
    ax.set_axis_off()
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)

    fig1, axs = plt.subplots(nrows=2, ncols=4)
    axs[0, 0].imshow(img1, cmap="gray", vmin=0, vmax=255)
    axs[0, 0].set_title("Draw particles", fontsize=15)
    axs[0, 1].imshow(img2, cmap="gray", vmin=0, vmax=255)
    axs[0, 1].set_title("Add particle structure", fontsize=15)
    axs[0, 2].imshow(img3, cmap="gray", vmin=0, vmax=255)
    axs[0, 2].set_title("Apply edge effect and background structure", fontsize=15)
    axs[0, 3].imshow(img4, cmap="gray", vmin=0, vmax=255)
    axs[0, 3].set_title("Add blur and astigmatism", fontsize=15)
    axs[1, 0].imshow(img5, cmap="gray", vmin=0, vmax=255)
    axs[1, 0].set_title("Add vibration effects", fontsize=15)
    axs[1, 1].imshow(img6, cmap="gray", vmin=0, vmax=255)
    axs[1, 1].set_title("Calculate Poisson noise", fontsize=15)
    axs[1, 2].imshow(img7, cmap="gray", vmin=0, vmax=255)
    axs[1, 2].set_title("Apply Poisson noise", fontsize=15)
    axs[1, 3].imshow(img8, cmap="gray", vmin=0, vmax=255)
    axs[1, 3].set_title("Crop image", fontsize=15)
    for i in axs.flatten():
        i.axis("off")
    plt.tight_layout()
    if Save_Process:
        plt.subplots_adjust(hspace=-0.25, wspace=0.05)
        fig1.set_size_inches(25, 15)
        fig1.savefig(Direct1 + "\\" + ImgName1, dpi=100, bbox_inches="tight")

# The simulated image is cropped to remove image axes and displayed
fig, ax = plt.subplots(frameon=False)
ax.set_axis_off()
ax.imshow(img8, cmap="gray", vmin=0, vmax=255)

# If specified, the simulated image is saved
if Save_img:
    print("Saving Image")
    cv2.imwrite(Direct + "\\" + ImgName, img8)

t2 = time.time()
print("Simulate image Brostom 2022 took " + str(t2 - t1) + " seconds")
print("------------ Done ------------")
plt.show()
