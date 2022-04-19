#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
.. py:currentmodule:: generate_micrograph
.. moduleauthor:: Hendrix Demers <hendrix.demers@mail.mcgill.ca>

Generate a micrograph.

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

# Third party modules.
import matplotlib.pyplot as plt

# Local modules.

# Project modules.
from micrograph.simulation import InputParameters, Simulation
from micrograph import get_current_module_path
from micrograph import time_fn

# Globals and constants variables.


@time_fn
def main():
    parameters = InputParameters()

    simulation = Simulation(parameters)

    simulation.process()

    file_path = get_current_module_path(__file__, "../data/micrograph_hd.png")

    simulation.display()

    simulation.save_micrograph(file_path)


if __name__ == '__main__':
    main()
    plt.show()
