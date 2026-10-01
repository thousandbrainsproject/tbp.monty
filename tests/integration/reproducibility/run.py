# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.


from omegaconf import DictConfig

from tbp.monty.frameworks.run import main as run_serial
from tbp.monty.frameworks.run_parallel import main as run_parallel


def serial_run(config: DictConfig):
    """Executes the experiment in serial mode."""
    run_serial(config)


def parallel_run(config: DictConfig):
    """Executes the experiment in parallel mode."""
    run_parallel(config)
