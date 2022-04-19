#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
.. py:currentmodule:: micrograph.simulation
.. moduleauthor:: Hendrix Demers <hendrix.demers@mail.mcgill.ca>

Create a simulated micrograph  based on Cizmar (2008) and Brostrom (2022).
"""

###############################################################################
# Copyright 2022 Hendrix Demers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################

# Standard library modules.
import os.path
from dataclasses import dataclass

# Third party modules.
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Local modules.

# Project modules.
from micrograph import time_fn

# Globals and constants variables.


@dataclass
class InputParameters:
    image_size: tuple[int] = (800, 800)

    number_of_particles: int = 10000
    particle_size_max: int = 40
    particle_size_min: int = 2
    particle_intensity: int = 150
    particle_intensity_variation: int = 50
    particle_structure_size: int = 300
    particle_structure_intensity: int = 50
    particle_structure_degree: float = 0.5

    edge_on: bool = True
    edge_intensity: int = 240
    edge_width: int = 4
    edge_steepness: float = 0.9

    background_intensity: int = 50
    background_structure: int = 500
    background_structure_intensity: int = 30
    background_structure_degree: int = 1

    focus_sigma: float = 1.0
    astigmatism: float = 1.0
    angle: int = 0

    noise_degree: float = 0.4

    vibration: bool = True
    max_vibration_shift_x: int = 2
    max_vibration_shift_y: float = 0.5
    shift_occurrence_x: int = 10
    shift_occurrence_y: int = 10


class Simulation:
    def __init__(self, parameters: InputParameters):
        self.show_steps = True
        self.save_image = True

        self.rng = np.random.default_rng(12345)

        self.parameters = parameters

        self.image = self._create_micrograph()
        self.images = []

        self._process_figure = None

    @time_fn
    def process(self):
        print("------------ Simulating ------------")

        image = self._create_micrograph()
        coords = self._generate_particles()
        grain_params = self._check_particles_parameters(coords, image)
        image1, image2, grain_params, background_structures, background_mask = self._draw_particles(image, grain_params)
        image3 = self._apply_edge_effect(image2, grain_params)
        image4 = self._apply_background_structure(image3, background_structures, background_mask)
        image5 = self._apply_vibration_effects_np(image4)
        image6, image7 = self._add_noise(image5)
        image8 = self._crop(image7)

        self.image = image8
        self.images = [image, image1, image2, image3, image4, image5, image6, image7, image8]

    def display(self):
        self.display_steps()

        # The simulated image is cropped to remove image axes and displayed
        fig, ax = plt.subplots(frameon=False)
        ax.set_axis_off()
        ax.imshow(self.image, cmap="gray", vmin=0, vmax=255)

    @time_fn
    def save_micrograph(self, file_path):
        if self.save_image:
            cv2.imwrite(file_path, self.image)
            if self.show_steps and self._process_figure is not None:
                file_path_process, extension = os.path.splitext(file_path)
                file_path_process = file_path_process + "_process" + extension
                plt.subplots_adjust(hspace=-0.25, wspace=0.05)
                self._process_figure.set_size_inches(25, 15)
                self._process_figure.savefig(file_path_process, dpi=100, bbox_inches="tight")

    @time_fn
    def _create_micrograph(self):
        h = self.parameters.image_size[0] + 40
        w = self.parameters.image_size[1] + 40
        image = np.zeros([h, w], dtype=np.uint8)

        return image

    @time_fn
    def _generate_particles(self):
        print("Generating Particles")
        h = self.parameters.image_size[0] + 40
        w = self.parameters.image_size[1] + 40
        # Generate random coordinates and deformation parameters to be used when making particles
        number_of_particles = self.parameters.number_of_particles
        min_particle_size = self.parameters.particle_size_min
        max_particle_size = self.parameters.particle_size_max

        rand_x = self.rng.integers(0, h - 1, size=number_of_particles)
        rand_y = self.rng.integers(0, w - 1, size=number_of_particles)
        rand_r = self.rng.integers(min_particle_size, max_particle_size, size=number_of_particles)
        rand_a1 = self.rng.uniform(0, 0.45, size=number_of_particles)
        rand_a2 = self.rng.uniform(0, 0.1, size=number_of_particles)
        rand_f1 = self.rng.uniform(0, 2 * np.pi, size=number_of_particles)
        rand_f2 = self.rng.uniform(0, 2 * np.pi, size=number_of_particles)
        coords = zip(rand_x, rand_y, rand_r, rand_a1, rand_a2, rand_f1, rand_f2)

        return coords

    @time_fn
    def _check_particles_parameters(self, coords, image):
        print("Checking Particle Parameters")

        edge_width = self.parameters.edge_width
        h = self.parameters.image_size[0] + 40
        w = self.parameters.image_size[1] + 40

        # Make lists to contain relevant parameters
        grain_params = []
        x_minimums = []
        x_maximums = []
        y_minimums = []
        y_maximums = []
        n = 0
        k = 0

        # Loop to generate particle contours on the black image
        for x, y, r, a1, a2, f1, f2 in coords:
            # Generate particle contours
            x1, y1 = make_particle(x, y, r, a1, a2, f1, f2)
            # In case edge effect is turned on, the required distance between particles is increased.
            if edge_width:
                edge_distance = edge_width
            else:
                edge_distance = 1
            # If new particle contours are on the edge of the image, they are not drawn
            if np.any(x1 < edge_distance) or np.any(y1 < edge_distance) or np.any(
                    x1 > h - (edge_distance + 1)) or np.any(
                    y1 > w - (edge_distance + 1)):
                continue
            # If the edge of new particle contours touch existing contours they are not drawn
            elif (255 in image[x1, y1]) or (255 in image[x1 - 1, y1]) or (255 in image[x1 + 1, y1]) or (
                    255 in image[x1, y1 + 1]) or (
                    255 in image[x1, y1 - 1]):
                continue
            # If new particle contours are inside existing contours they are not drawn
            elif n != 0:
                k = 0
                for j in range(len(x_minimums)):
                    if (x1.min() >= x_minimums[j]) & (x1.max() <= x_maximums[j]) & (y1.max() <= y_maximums[j]) & (
                            y1.min() >= y_minimums[j]):
                        k += 1
                    elif (x1.min() <= x_minimums[j]) & (x1.max() >= x_maximums[j]) & (y1.max() >= y_maximums[j]) & (
                            y1.min() <= y_minimums[j]):
                        k += 1
            if k != 0:
                continue
            # Approved particle contours are drawn and their parameters are
            # stored for later use.
            image[x1, y1] = 255
            grain_params.append([x, y, r, a1, a2, f1, f2])
            x_minimums.append(x1.min())
            x_maximums.append(x1.max())
            y_minimums.append(y1.min())
            y_maximums.append(y1.max())
            n += 1

        return grain_params

    @time_fn
    def _draw_particles(self, image, grain_params):
        print("Drawing Particles")
        # The drawn particle contours are located and pixels inside are set to 255
        contours1, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
        image1 = cv2.drawContours(image.copy(), contours1, -1, 255, -1)

        # Particle contours are located again after they have been filled
        contours, hierarchy = cv2.findContours(image1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Generate masks for particles and background
        particle_mask = image1.astype(np.uint8)
        background_mask = ~particle_mask

        # Generate intensity structure inside particles
        particle_structure_size = self.parameters.particle_structure_size
        particle_intensity = self.parameters.particle_intensity
        structure_degree_p = self.parameters.particle_structure_degree
        structure_intensity_p = self.parameters.particle_structure_intensity
        particle_intensity_var = self.parameters.particle_intensity_variation
        particle_structures, grain_params = structure_generation(self.rng, self.parameters, hierarchy,
                                                                 particle_structure_size, particle_intensity,
                                                                 structure_degree_p, structure_intensity_p, contours,
                                                                 particle_intensity_var, grain_params)
        # Generate intensity structure in the background
        background_structure = self.parameters.background_structure
        background_intensity = self.parameters.background_intensity
        structure_degree_b = self.parameters.background_structure_degree
        structure_intensity_b = self.parameters.background_structure_intensity
        background_structures, _ = structure_generation(self.rng, self.parameters, hierarchy, background_structure,
                                                        background_intensity, structure_degree_b,
                                                        structure_intensity_b)

        # Apply the particle structure to the image
        image2 = image1.copy()
        image2[particle_mask == 255] = particle_structures[particle_mask == 255]

        return image1, image2, grain_params, background_structures, background_mask

    @time_fn
    def _apply_edge_effect(self, image2, grain_params):
        # Edge effects is applied if specified
        if self.parameters.edge_on:
            print("Applying Edge Effect")
            particle_intensity = self.parameters.particle_intensity
            edge_intensity = self.parameters.edge_intensity
            edge_steepness = self.parameters.edge_steepness
            edge_width = self.parameters.edge_width
            image3 = image_convolve_mask(image2.copy(), grain_params, particle_intensity, edge_intensity,
                                         edge_steepness, edge_width)
        else:
            image3 = image2.copy()

        return image3

    @time_fn
    def _apply_background_structure(self, image3, background_structures, background_mask):
        print("Applying Background Structure")
        # Background structure is applied
        image3[background_mask == 255] = background_structures[background_mask == 255]

        # Generate kernel to add blur and astigmatism
        sigma = self.parameters.focus_sigma
        astigmatism = self.parameters.astigmatism
        angle = self.parameters.angle
        if sigma:
            kern_size = int(sigma * 5)
            kern = np.zeros((kern_size, kern_size))
            for j, i in enumerate(kern):
                for k, l in enumerate(i):
                    kern[j, k] = blurr_filter(j - ((kern_size - 1) / 2), k - ((kern_size - 1) / 2), astigmatism, angle,
                                              sigma)

            # Apply blur and astigmatism
            image4 = cv2.filter2D(image3, -1, kern).astype(np.uint8)
        else:
            image4 = image3.copy()

        return image4

    @time_fn
    def _apply_vibration_effects(self, image4):
        print("Applying Vibrational Effects")
        # Apply vibration and drift artefacts
        image5 = image4.copy()
        vibration = self.parameters.vibration
        max_vibration_shift_x = self.parameters.max_vibration_shift_x
        max_vibration_shift_y = self.parameters.max_vibration_shift_y
        shift_occurrence_x = self.parameters.shift_occurrence_x
        shift_occurrence_y = self.parameters.shift_occurrence_y
        h = self.parameters.image_size[0] + 40
        w = self.parameters.image_size[1] + 40

        if vibration:
            number_pixels = image5.ravel().shape[0]
            for i in range(10, h - 10):
                ax = max_vibration_shift_x * self.rng.random(1)
                for j in range(10, w - 10):
                    time = i * h + j
                    ay = max_vibration_shift_y * self.rng.random(1)
                    # Produce shifts in x direction
                    xv = ax * np.sin(shift_occurrence_x * (time / number_pixels))
                    # Produce shifts in y direction
                    yv = ay * np.sin(shift_occurrence_y * (time / number_pixels))
                    # Apply shifts to the image
                    image5[i, j] = image4[i + int(xv), j + int(yv)]

        return image5

    @time_fn
    def _apply_vibration_effects_np(self, image4):
        print("Applying Vibrational Effects")
        # Apply vibration and drift artefacts
        image5 = image4.copy()
        vibration = self.parameters.vibration
        max_vibration_shift_x = self.parameters.max_vibration_shift_x
        max_vibration_shift_y = self.parameters.max_vibration_shift_y
        shift_occurrence_x = self.parameters.shift_occurrence_x
        shift_occurrence_y = self.parameters.shift_occurrence_y
        h = self.parameters.image_size[0] + 40
        w = self.parameters.image_size[1] + 40

        if vibration:
            number_pixels = image5.ravel().shape[0]
            indices_x, indices_y = np.indices((h, w))
            indices_x = indices_x[10:h - 10]
            indices_y = indices_y[10:h - 10]

            ax = max_vibration_shift_x * self.rng.random(indices_x.shape)
            ay = max_vibration_shift_y * self.rng.random(indices_y.shape)
            time = indices_x * h + indices_y
            # Produce shifts in x direction
            xv = ax * np.sin(shift_occurrence_x * (time / number_pixels))
            xv = xv.astype(int)
            # Produce shifts in y direction
            yv = ay * np.sin(shift_occurrence_y * (time / number_pixels))
            yv = yv.astype(int)

            image5[indices_x, indices_y] = image4[indices_x + xv, indices_y + yv]

        return image5

    @time_fn
    def _add_noise(self, image5):
        print("Adding Noise")
        # Calculate poisson noise of the image
        image6 = self.rng.poisson(image5)
        # In case some intensities are above 255 they are lowered to 255
        image6[image6 > 255] = 255
        image6 = image6.astype(np.uint8)
        # The noise is added to the image based on specified weights
        noise_degree = self.parameters.noise_degree
        image7 = cv2.addWeighted(image6, noise_degree, image5, 1 - noise_degree, 0)

        return image6, image7

    @time_fn
    def _crop(self, image7):
        h = self.parameters.image_size[0] + 40
        w = self.parameters.image_size[1] + 40
        image8 = image7[20:h - 20, 20:w - 20]
        return image8

    def display_steps(self):
        image0, image1, image2, image3, image4, image5, image6, image7, image8 = self.images
        # If specified the individual steps of the algorithm are displayed and saved
        if self.show_steps:
            self._process_figure, axs = plt.subplots(nrows=2, ncols=5)
            axs[0, 0].imshow(image0, cmap="gray", vmin=0, vmax=255)
            axs[0, 0].set_title("Particles contour", fontsize=15)
            axs[0, 1].imshow(image1, cmap="gray", vmin=0, vmax=255)
            axs[0, 1].set_title("Draw particles", fontsize=15)
            axs[0, 2].imshow(image2, cmap="gray", vmin=0, vmax=255)
            axs[0, 2].set_title("Add particle structure", fontsize=15)
            axs[0, 3].imshow(image3, cmap="gray", vmin=0, vmax=255)
            axs[0, 3].set_title("Apply edge effect and background structure", fontsize=15)
            axs[0, 4].imshow(image4, cmap="gray", vmin=0, vmax=255)
            axs[0, 4].set_title("Add blur and astigmatism", fontsize=15)
            axs[1, 0].imshow(image5, cmap="gray", vmin=0, vmax=255)
            axs[1, 0].set_title("Add vibration effects", fontsize=15)
            axs[1, 1].imshow(image6, cmap="gray", vmin=0, vmax=255)
            axs[1, 1].set_title("Calculate Poisson noise", fontsize=15)
            axs[1, 2].imshow(image7, cmap="gray", vmin=0, vmax=255)
            axs[1, 2].set_title("Apply Poisson noise", fontsize=15)
            axs[1, 3].imshow(image8, cmap="gray", vmin=0, vmax=255)
            axs[1, 3].set_title("Crop image", fontsize=15)
            axs[1, 4].imshow(self.image, cmap="gray", vmin=0, vmax=255)
            axs[1, 4].set_title("Final micrograph", fontsize=15)
            for i in axs.flatten():
                i.axis("off")
            plt.tight_layout()


def make_particle(x0, y0, r, a1, a2, f1, f2):
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


def structure_generation(rng, parameters, hierarchy, structure_size, mean_intensity, degree, struct_intensity,
                         contours_p=0, var=0, grain_params=0):
    """
    Function to apply structure inside particles and in the background. The
    structure is generated from a noisy image which is FFT transformed, padded,
    and then reverse transformed to real space. This produces a realistic
    structure, which be controlled from the size of the initial noise image
    relative to the size of the padding.
    """
    # Produce image of random noise with specified size
    grain_noise_matrix = rng.integers(0, 255, size=(structure_size, structure_size))

    # FFT convert the noise image
    fft = np.fft.fftshift(np.fft.fft2(grain_noise_matrix))

    # pad the image
    h = parameters.image_size[0] + 40
    w = parameters.image_size[1] + 40
    pad_width = int((w - structure_size) / 2)
    pad_height = int((h - structure_size) / 2)
    padded = np.pad(fft, ((pad_height, pad_height), (pad_width, pad_width)), pad_with, padder=0)

    # Reverse FFT
    back = np.abs(np.fft.ifft2(np.fft.ifftshift(padded)))

    # Normalize the resulting image to ensure intensities between 0-255
    structure = cv2.normalize(back, hierarchy, mean_intensity - struct_intensity, mean_intensity + struct_intensity,
                              cv2.NORM_MINMAX).astype(np.uint8)
    structure[structure > 255] = 255

    # Apply the calculated structure to particles or background
    if contours_p:
        average_map = np.zeros_like(structure, dtype=np.uint8)
        for j, i in enumerate(contours_p):
            if var:
                intensity = rng.integers(mean_intensity - var, mean_intensity + var)
                grain_params[j] = grain_params[j] + [intensity]
            else:
                intensity = mean_intensity
            if intensity < 0:
                intensity = 0
            elif intensity > 255:
                intensity = 255
            average_map = cv2.drawContours(average_map, [i], -1, int(intensity), -1)
    else:
        average_map = np.ones_like(structure, dtype=np.uint8) * mean_intensity
    final_structure = cv2.addWeighted(average_map, 1 - degree, structure, degree, 0)
    return final_structure, grain_params


def pad_with(vector, pad_width, iaxis, kwargs):
    """
    Function to pad the image, which is used when producing random structures
    in the background and on particles.
    """
    pad_value = kwargs.get('padder', 10)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value


def image_convolve_mask(image, params, p_intensity, edge, edge_steepness, edge_width):
    """
    Function to apply edge effects on existing particles.
    """
    for param in params:
        edge_intensity = p_intensity + edge
        if edge_intensity > 255:
            edge_intensity = 255
        x = param[0]
        y = param[1]
        r = param[2]
        a1 = param[3]
        a2 = param[4]
        f1 = param[5]
        f2 = param[6]
        for i in range(edge_width):
            # Find the edge coordinates of existing particles
            x1, y1 = make_particle(x, y, r - i, a1, a2, f1, f2)
            # Set the color of particle edges to edge_intensity and calculate new
            # intensity values for particle pixels when moving inwards until edge_width is reached
            if (max(y1) - min(y1)) == 0:
                y2 = y1
            else:
                y2 = (y1 - min(y1)) / (max(y1) - min(y1))
            initial_part = (p_intensity - edge_intensity) / (np.e ** (-edge_steepness * edge_width) - 1)
            color = initial_part * (np.e ** (-edge_steepness * i) - 1) + edge_intensity
            image[x1, y1] = color * y2 + p_intensity * (1 - y2)
    return image


def blurr_filter(x, y, s, phi, sigma):
    """
    Filter used to produce blur and astigmatism in the image.
    """
    x1 = s * (x * np.cos(phi) + y * np.sin(phi))
    y1 = (1 / s) * (-x * np.sin(phi) + y * np.cos(phi))
    p = (1 / (2 * np.pi * sigma ** 2)) * np.e ** (-(x1 ** 2 + y1 ** 2) / (2 * sigma ** 2))
    return p
