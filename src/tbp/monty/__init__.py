# Copyright 2025-2026 Thousand Brains Project
# Copyright 2021-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

__version__ = "0.30.0"

import sys

# Things break if we try to import some of these classes in the Conda version
if sys.version_info >= (3, 10):
    import torch
    from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
    from torch_geometric.data.storage import GlobalStorage

    from tbp.monty.frameworks.models.object_model import GraphObjectModel

    # PyTorch 2.6 changes the default value of `weights_only` in `torch.load` from False
    # to True, as a result it now gives an error when we try to load things without
    # specifying `weights_only=False`.
    #
    # Rather than change all the call sites, the error message suggests an alternative
    # of allow-listing the expected global classes we use.
    # See https://docs.pytorch.org/docs/stable/notes/serialization.html#weights-only-allowlist
    torch.serialization.add_safe_globals(
        [GraphObjectModel, DataEdgeAttr, DataTensorAttr, GlobalStorage]
    )
