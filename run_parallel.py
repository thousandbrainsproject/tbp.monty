#!/usr/bin/env python
# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
import sys
from contextlib import redirect_stdout

from tbp.monty.frameworks.run_env import setup_env

with redirect_stdout(sys.stderr):
    setup_env()

from tbp.monty.frameworks.run_parallel import main

if __name__ == "__main__":
    main()
