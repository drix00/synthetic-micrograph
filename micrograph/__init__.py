#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
.. py:currentmodule:: trim.montecarlo

.. moduleauthor:: Hendrix Demers <hendrix.demers@mail.mcgill.ca>


"""

# Copyright 2019 Hendrix Demers
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

# Standard library modules.
import os.path
from logging import getLogger
import logging.config
import os
from functools import wraps
import time
from pathlib import Path

# Third party modules.

# Local modules.

# Project modules.

# Globals and constants variables.
__author__ = 'Hendrix Demers'
__email__ = 'hendrix.demers@mail.mcgill.ca'
__version__ = '0.1.0'
__copyright__ = '2019, ' + __author__
__project_name__ = 'Synthetic-Micrograph'

logger = getLogger(__name__)


def get_current_module_path(module_path: str, relative_path: str = "") -> Path:
    """
    Extract the current module path by using :py:obj:`__file__` special module variable
    and combine it with the relative path and return it.

    An example of usage::

        module_path = get_current_module_path(__file__)

    :param str module_path: Pass the `__file__` python keyword for this parameter
    :param str relative_path: The relative path to combine with the module path
    :return: The path obtained when combine the module path and relative path
    :rtype: Path
    """
    base_path = Path(module_path).parent
    file_path = base_path.joinpath(relative_path)
    file_path = file_path.resolve()

    return file_path


def get_log_file_path():
    path = get_current_module_path(__file__, "../../logs")
    logger.debug("log_file_path: %s", path)
    if not os.path.isdir(path):  # pragma: no cover
        os.makedirs(path)

    return path


def setup_logger():
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)

    # noinspection SpellCheckingInspection
    log_format = '%(asctime)s : %(name)-40s : %(levelname)-10s : %(message)s'
    formatter = logging.Formatter(log_format)

    ch.setFormatter(formatter)

    root_logger.addHandler(ch)

    path = get_log_file_path()
    log_file_path = os.path.join(path, "{}.log".format("synthetic_micrograph"))
    fh = logging.FileHandler(log_file_path)
    fh.setFormatter(formatter)
    fh.setLevel(logging.DEBUG)
    root_logger.addHandler(fh)


def time_fn(fn):
    @wraps(fn)
    def measure_time(*args, **kargs):
        t1 = time.time()
        results = fn(*args, **kargs)
        t2 = time.time()
        print("@time_fn:" + fn.__name__ + " took " + str(t2 - t1) + " seconds")
        return results
    return measure_time
