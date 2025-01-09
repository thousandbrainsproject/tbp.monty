# Copyright 2025 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import unittest


class DependenciesTest(unittest.TestCase):
    def test_torch_geometric(self):
        import torch_geometric
        import torch_geometric.data
        import torch_geometric.typing  # noqa: F401
