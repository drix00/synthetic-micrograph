#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
.. py:currentmodule:: micrograph.timing
.. moduleauthor:: Hendrix Demers <hendrix.demers@mail.mcgill.ca>

Various utilities for timing code execution.
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
from functools import wraps
# Third party modules.

# Local modules.

# Project modules.

# Globals and constants variables.


def time_fn(fn):
    @wraps(fn)
    def measure_time(*args, **kwargs):
        timing = Timing()
        timing.start()

        results = fn(*args, **kwargs)

        elapsed_time_s = timing.elapsed_time_s()
        print(f"@time_fn: {fn.func_name} took {str(elapsed_time_s)} seconds")
        return results
    return measure_time


class TimingError(Exception):
    """A custom exception used to report errors in use of Timing class"""


class Timing:
    def __init__(self):
        self._start_time = None

    def start(self):
        """Start a new timer"""
        if self._start_time is not None:
            raise TimingError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def elapsed_time_s(self):
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimingError(f"Timer is not running. Use .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None

        return elapsed_time
