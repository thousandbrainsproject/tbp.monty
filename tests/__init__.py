# Copyright 2025-2026 Thousand Brains Project
# Copyright 2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from pathlib import Path

from tbp.monty.hydra import register_resolvers

# Initialize Hydra resolvers for test configs
register_resolvers()

# Root directory for Hydra configs
HYDRA_ROOT = Path(__file__).parents[1] / "src" / "tbp" / "monty" / "conf"
