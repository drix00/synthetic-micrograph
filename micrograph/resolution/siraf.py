#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
.. py:currentmodule:: micrograph.resolution.siraf
.. moduleauthor:: Hendrix Demers <hendrix.demers@mail.mcgill.ca>

Implementation of SIRAF algorithm for spatial resolution calculation of micrograph.
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
from enum import Enum

# Third party modules.
import matplotlib.pyplot as plt
import cv2 as cv2
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import sem

# Local modules.

# Project modules.
from micrograph.timing import Timing

# Globals and constants variables.


class WeightError(Enum):
    NO = 0
    STD = 1
    STD_ERROR = 2


class SirafError(Exception):
    """A custom exception used to report errors in use of Siraf class"""


class Siraf:
    def __init__(self):
        """
        hanning_window : TYPE - Boolean, optional
          DESCRIPTION - Set whether to apply a hanning filter to the image
                        to minimize horisontal and vertical white lines in
                        the image fft.
                        The default is 1.
        weight_error : TYPE - str, optional
                     DESCRIPTION - Set whether to use weights in the fitting
                                   procedure. Can be set to:
                                   "std"    : Std of each radial mean will be used
                                              as weight when fitting
                                   "std_err": Standard erro of the mean for each
                                              radial mean will be used as weight
                                              when fitting

        """
        self.hanning_window = True
        self.weight_error = WeightError.STD
        self.hanning_limit = False
        self.gibbs_limit = False

    def compute_fwhm(self, micrograph_data):

        # Call the algorithm and perform the analysis
        results = self.image_algorithm(micrograph_data)

        return results

    def image_algorithm(self, micrograph_data):
        """
        This function is the main function of the algorithm. It treats the
        image data to give the radially averaged data necessary for the fitting
        procedure, and performs the fit as well using the scipy.optimize function.

        Parameters
        ----------
        micrograph_data : TYPE - numpy.ndarray, dtype=numpy.uint8
               DESCRIPTION - Image to analyse.

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
        distances = distance_matrix_ellipse(micrograph_data.shape)

        # Set the step width in the radial binning procedure
        step = 1 / min(micrograph_data.shape)

        # Generate an array of radii for the radial distance binning
        radii = np.arange(step, distances.max(), step)

        # Get image shape
        n, n1 = micrograph_data.shape

        img2 = self.compute_hanning_filter(micrograph_data, n, n1)

        amplitude_spectrum = self.compute_amplitude_spectrum(img2)

        means = self.compute_means(amplitude_spectrum, distances, radii, step)

        alpha_init, beta_init, c_init, param_bounds = self.initialize_fit(means, n, n1, radii)

        std, std_error = self.compute_weight_error(amplitude_spectrum, distances, radii, step)

        high_limit, low_limit = self.compute_limit(micrograph_data)

        pcov, popt = self.compute_fit(alpha_init, beta_init, c_init, high_limit, low_limit, means, param_bounds, radii,
                                      std, std_error)

        errs = self.compute_fit_errors(pcov)

        # Return results from the algorithm.
        results = SirafResults()
        results.fit_parameters = popt
        results.fit_errs = errs
        results.image2 = img2
        results.amplitude_spectrum = amplitude_spectrum
        results.means = means
        results.radii = radii
        results.std = std
        results.std_error = std_error

        return results

    def compute_fit_errors(self, pcov):
        # Calculate errors of the fitting procedure
        errs = np.sqrt(np.diag(pcov))
        return errs

    def compute_fit(self, alpha_init, beta_init, c_init, high_limit, low_limit, means, param_bounds, radii, std,
                    std_error):
        # Fit the image data using the function derived from the convolution of a
        # step and Gaussian function in FFT space. The image data can fitted to
        # unweighted data or to data, which is weighted by the std or the standard
        # error of the mean, depending on user specification
        x_data = radii[(radii > low_limit) & (radii < high_limit)]
        y_data = means[(radii > low_limit) & (radii < high_limit)]
        p0 = np.array([alpha_init, beta_init, c_init])
        sigma = None
        if self.weight_error == WeightError.STD:
            sigma = std[(radii > low_limit) & (radii < high_limit)]
        elif self.weight_error == WeightError.STD_ERROR:
            sigma = std_error[(radii > low_limit) & (radii < high_limit)]
        elif self.weight_error == WeightError.NO:
            sigma = None
        else:
            raise SirafError("Unknown weight error option")
        popt, pcov = curve_fit(func_fit, x_data, y_data, p0=p0, bounds=param_bounds, sigma=sigma)
        return pcov, popt

    def compute_limit(self, micrograph_data):
        # If specified, set limits for the fitting window to exclude potential
        # Gibbs lines or influence from the Hanning window at low frequencies.
        if self.gibbs_limit:
            high_limit = 0.5
        else:
            high_limit = 0.5
        if self.hanning_limit:
            low_limit = int(20 / min(micrograph_data.shape))
        else:
            low_limit = 0
        return high_limit, low_limit

    def compute_weight_error(self, amplitude_spectrum, distances, radii, step):
        # Calculate the standard deviation (Std) and the standard error of the mean (Std_err)
        std = np.array([np.std(amplitude_spectrum[(distances >= r) & (distances < r + step)]) for r in radii])
        std_error = np.array([sem(amplitude_spectrum[(distances >= r) & (distances < r + step)]) for r in radii])

        return std, std_error

    def initialize_fit(self, means, n, n1, radii):
        # Setup initiation values and parameter bounds for the fitting procedure
        alpha_init = means[0] / 2
        beta_init = 3
        if self.gibbs_limit:
            c_init = means[(radii > 0.4) & (radii < 0.5)].mean()
        else:
            c_init = means[(radii > 0.6) & (radii < 0.7)].mean()
        param_bounds = ([0, -np.inf, c_init / 2], [max([n, n1]), np.inf, c_init * 2])
        return alpha_init, beta_init, c_init, param_bounds

    def compute_means(self, amplitude_spectrum, distances, radii, step):
        # Calculate the radial means of the image using the distance bins
        means = np.array([np.nanmean(amplitude_spectrum[(distances >= r) & (distances < r + step)]) for r in radii])
        # To account for potential errors while averaging, NaN values are set to 0
        means[np.isnan(means)] = 0
        return means

    def compute_amplitude_spectrum(self, img2):
        # Determine the shifted amplitude spectrum via FFT analysis.
        # The log(1+..) step is used to scale the image data, resulting in more robust
        # fits
        amplitude_spectrum = np.fft.fftshift(np.log(1 + np.abs(np.fft.fft2(img2, norm="ortho"))))
        return amplitude_spectrum

    def compute_hanning_filter(self, micrograph_data, n, n1):
        # Generate a Hanning window if opted by the user
        if self.hanning_window:
            h = np.hanning(n)
            h1 = np.hanning(n1)
            han2d = np.sqrt(np.outer(h, h1))

            # Apply Hanning window
            img2 = micrograph_data * han2d
        else:
            # If a Hanning window is not opted, use the original image
            img2 = micrograph_data
        return img2

    def plot(self, results):
        """Display image, amplitude spectrum and radial average plot."""
        # If specified by the user, plots are generated to illustrate the process
        # and the results

        # Generate figures for plotting
        fig = plt.figure(tight_layout=True)
        gs = fig.add_gridspec(2, 2)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, :])

        # Display Image
        ax1.imshow(results.image2, cmap="gray")
        ax1.set_axis_off()

        # Display amplitude spectrum
        ax2.imshow(results.amplitude_spectrum, cmap="gray")
        ax2.set_axis_off()

        # Display image data as radial average plot
        ax3.plot(results.radii, results.means, color="blue", label="Image data", lw=4)

        # Display error weights if any
        if self.weight_error == WeightError.STD:
            ax3.fill_between(results.radii, results.means - results.std, results.means + results.std, color="blue",
                             alpha=0.5, label="Standard deviation")
        elif self.weight_error == WeightError.STD_ERROR:
            ax3.fill_between(results.radii, results.means - results.std_error, results.means + results.std_error,
                             color="blue", alpha=0.5,
                             label="Standard error of the mean")

        # Plot the fit
        label = "Fit: FWHM = ${:.2f} \pm {:.2f}$".format(2.355 * results.fit_parameters[0], 2.355 * results.fit_errs[0])
        y = func_fit(results.radii, results.fit_parameters[0], results.fit_parameters[1], results.fit_parameters[2])
        ax3.plot(results.radii, y, color="red", ls="--", label=label, lw=4)

        if self.gibbs_limit:
            ax3.axvspan(0.5, 0.707, color="k", alpha=0.5)
        if self.hanning_limit:
            ax3.axvspan(0, 0.01, color="k", alpha=0.5)

        # Set axis labels, legend, and axis limits
        ax3.legend(fontsize=25)
        ax3.grid(which="both", axis="both")
        ax3.set_xlim(0, 0.707)
        ax3.set_xlabel("Normalized Frequency", fontsize=25)
        ax3.set_ylabel("Amplitude", fontsize=25)


def distance_matrix_ellipse(shape):
    """
    Function used to generate a matrix containing distance measures for all
    coordinates (x,y) relative to the image center. This is used when binning
    pixels according to distance from the center of the amplitude spectrum

    Parameters
    ----------
    shape : TYPE - numpy.ndarray
               DESCRIPTION - Shape of the image from which to generate a distance map

    Returns
    -------
    distances : TYPE - numpy.ndarray
                DESCRIPTION - Distance map with same dimensions as input image

    """
    height, width = shape
    maximums = ((height - 1), (width - 1))
    grid_x, grid_y = np.mgrid[0:height, 0:width]
    grid_x = grid_x / maximums[0] - 0.5
    grid_y = grid_y / maximums[1] - 0.5
    distances = np.hypot(grid_x, grid_y)
    return distances


def func_fit(x, sigma, b, c):
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
    fitted_values = c + np.log(1 + (b / (x * np.sqrt(2 * np.pi))) * np.e ** (-2 * (np.pi * sigma * x) ** 2))
    return fitted_values


class SirafResults:
    def __init__(self):
        self.fit_parameters = None
        self.fit_errs = None
        self.image2 = None
        self.amplitude_spectrum = None
        self.means = None
        self.radii = None

        self.std = None
        self.std_error = None


def main():
    timing = Timing()
    timing.start()

    direct = r"../../data"
    img_name = "Standard_Img.png"

    img = cv2.imread(direct + "\\" + img_name, 0)

    siraf = Siraf()

    results = siraf.compute_fwhm(img)

    # Print the determined sigma and FWHM of the Gaussian Point Spread function
    print("-----------------------------")
    print("Sigma: {:.2f} +/- {:.2f} pixels".format(results.fit_parameters[0], results.fit_errs[0]))
    print("FWHM: {:.2f} +/- {:.2f} pixels".format(2.355 * results.fit_parameters[0], 2.355 * results.fit_errs[0]))
    print("-----------------------------")

    elapsed_time_s = timing.elapsed_time_s()
    print("SIRAF algorithm Demers took " + str(elapsed_time_s) + " seconds")

    # siraf.plot(results)


if __name__ == '__main__':
    main()

    # plt.show()
