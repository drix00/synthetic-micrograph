#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
.. py:currentmodule:: micrograph.resolution.smart
.. moduleauthor:: Hendrix Demers <hendrix.demers@mail.mcgill.ca>

Description
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
import time

# Third party modules.
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv2

# Local modules.

# Project modules.
from micrograph.timing import Timing

# Globals and constants variables.


class SmartWindow:
    def __init__(self, normed_magnitude):
        self.normed_magnitude = normed_magnitude
        self.dst1 = None

    def on_trackbar(self, threshold_value):
        """
        Function to generate and update the threshold slider in the displayed window
        """
        _, self.dst1 = cv2.threshold(self.normed_magnitude, threshold_value, 255, cv2.THRESH_BINARY)
        src2 = cv2.cvtColor(self.normed_magnitude, cv2.COLOR_GRAY2RGB)
        dst2 = np.zeros_like(src2)
        dst2[:, :, 2] = self.dst1
        dst = cv2.addWeighted(dst2, 0.25, src2, 0.75, 0.0)

        ellipse, processed = compute_ellipse(self.dst1)

        dst = cv2.ellipse(dst, ellipse, color=(255, 0, 0), thickness=2)
        cv2.imshow("Threshold - press 'q' to proceed", dst)

        return threshold_value

    def display(self):
        # Code for the interactive threshold
        cv2.namedWindow("Threshold - press 'q' to proceed")
        cv2.imshow("Threshold - press 'q' to proceed", self.normed_magnitude.astype(np.uint8))
        cv2.createTrackbar("Threshold Val", "Threshold - press 'q' to proceed", 130, 255, self.on_trackbar)

        # Wait until user press some key
        if cv2.waitKey() & 0xFF == ord('q'):
            cv2.destroyAllWindows()


def compute_ellipse(image_data):
    # Opening and closing steps after thresholding. Necessary to remove unwanted noise
    kernel = np.ones((2, 2), np.uint8)
    processed = image_data.copy()
    processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
    processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
    # Find contours in the resulting binary image and select the largest, if several are present
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    if len(contours) > 0:
        areas = []
        for c in contours:
            areas = areas + [float(cv2.contourArea(c))]

        ix = areas.index(max(areas))
        contour = contours[ix]
    else:
        contour = contours
    # Fit an ellipse to the largest contour in the image and display it in red
    ellipse = cv2.fitEllipse(contour)
    return ellipse, processed


class Smart:
    def __init__(self):
        self.roi_size = 800
        self.t2 = None
        self.t3 = None

        self.roi = None
        self.normed_magnitude = None
        self.dst1 = None
        self.final = None

    def compute(self, micrograph_data):
        roi_size = self.copmute_roi(micrograph_data)

        self.compute_normed_magnitude_spectrum()

        self.t2 = time.time()

        smart_window = SmartWindow(self.normed_magnitude)
        smart_window.display()

        self.t3 = time.time()
        self.dst1 = smart_window.dst1
        ellipse, processed = compute_ellipse(smart_window.dst1)

        self.final = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
        self.final = cv2.ellipse(self.final, ellipse, color=(255, 0, 0), thickness=2)

        eccentricity, img_res = self.compute_results(ellipse, roi_size)

        return img_res, eccentricity

    def compute_results(self, ellipse, roi_size):
        # Calculate the image resolution and eccentricity of the fitted ellipse
        img_res = float(roi_size) / np.mean(ellipse[1])
        eccentricity = (max(ellipse[1]) - min(ellipse[1])) / max(ellipse[1])
        return eccentricity, img_res

    def compute_normed_magnitude_spectrum(self):
        # Perform FFT and shift high frequency components to the centre.
        f = np.fft.fft2(self.roi)
        fft_shift = np.fft.fftshift(f)
        # Determine the magnitude spectrum, and normalize the image to a range from 0:255.
        magnitude_spectrum = 20 * np.log(np.abs(fft_shift))
        self.normed_magnitude = 255. * magnitude_spectrum / magnitude_spectrum.max()
        self.normed_magnitude = self.normed_magnitude.astype(np.uint8)

    def copmute_roi(self, micrograph_data):
        width, height = micrograph_data.shape
        center_x = int(width / 2)
        center_y = int(height / 2)
        roi_size = int(self.roi_size / 2)
        self.roi = micrograph_data[center_x - roi_size:center_x + roi_size, center_y - roi_size:center_y + roi_size]
        return roi_size

    def display(self):
        # Display the results
        fig, axs = plt.subplots(nrows=2, ncols=2, sharex="all", sharey="all", figsize=(15, 15))
        axs[0, 0].imshow(self.roi, cmap="gray")
        axs[0, 0].set_title("ROI")
        axs[0, 1].imshow(self.normed_magnitude, cmap="gray")
        axs[0, 1].set_title("Magnitude Spectrum")
        axs[1, 0].imshow(self.dst1, cmap="gray")
        axs[1, 0].set_title("Binary Magnitude spectrum")
        axs[1, 1].imshow(self.final, cmap="gray")
        axs[1, 1].set_title("Fitted ellipse")
        plt.tight_layout()


def main():
    timing = Timing()
    timing.start()

    direct = r"../../data"
    img_name = "Standard_Img.png"

    t1 = time.time()

    """
    This section contains the algorithm steps
    """
    # Load image
    img = cv2.imread(direct + "\\" + img_name, 0)

    smart = Smart()

    img_res, eccentricity = smart.compute(img)

    # Print the results to the console
    print("The image resolution is {0} pixels".format(img_res))
    print("The stigmatic error is {0}".format(eccentricity))

    t4 = time.time()
    elapse_time = (smart.t2 - t1) + (t4 - smart.t3)
    print("SMART algorithm Demers took " + str(elapse_time) + " seconds")

    elapsed_time_s = timing.elapsed_time_s()
    print("SMART algorithm Demers took total " + str(elapsed_time_s) + " seconds")

    smart.display()


if __name__ == '__main__':
    main()

    plt.show()
