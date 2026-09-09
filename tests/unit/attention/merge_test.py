# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from tbp.monty.attention.merge import InhibitionFlipsGrid, Union
from tbp.monty.attention.voxel_grid import VOXEL_LEVELS, VoxelGrid
from tbp.monty.cmp import MIN_ATTENTION_WEIGHT

VOXEL_SIZE = 0.01


def grid_of(
    weights_by_voxel: dict[tuple[int, int, int], float], inhibit_all: bool = False
) -> VoxelGrid:
    """Build a grid holding the given voxels and weights.

    Returns:
        The grid.

    """
    frame = pd.DataFrame(
        {"weight": np.asarray(list(weights_by_voxel.values()), dtype=float)},
        index=pd.MultiIndex.from_tuples(weights_by_voxel.keys(), names=VOXEL_LEVELS),
    )
    return VoxelGrid(VOXEL_SIZE, frame, inhibit_all)


def as_dict(grid: VoxelGrid) -> dict[tuple[int, int, int], float]:
    """Map each of the grid's voxels to its weight.

    Returns:
        Voxel coordinate to weight.

    """
    data = grid.to_pandas()
    voxels = [tuple(int(c) for c in index) for index in data.index]
    return dict(zip(voxels, data["weight"].to_numpy().tolist()))


class UnionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.merge = Union()

    def test_a_proposed_voxel_takes_the_fresh_weight(self) -> None:
        merged = self.merge(grid_of({(0, 0, 0): 4}), grid_of({(0, 0, 0): 2}))
        self.assertEqual(as_dict(merged), {(0, 0, 0): 2})

    def test_unproposed_voxels_are_carried_over(self) -> None:
        merged = self.merge(grid_of({(0, 0, 0): 4}), grid_of({(1, 0, 0): 2}))
        self.assertEqual(as_dict(merged), {(0, 0, 0): 4, (1, 0, 0): 2})

    def test_an_empty_grid_a_yields_grid_b(self) -> None:
        grid_b = grid_of({(0, 0, 0): 2})
        merged = self.merge(VoxelGrid(VOXEL_SIZE), grid_b)
        self.assertEqual(as_dict(merged), {(0, 0, 0): 2})

    def test_an_empty_grid_b_yields_grid_a(self) -> None:
        grid_a = grid_of({(0, 0, 0): 4})
        merged = self.merge(grid_a, VoxelGrid(VOXEL_SIZE))
        self.assertEqual(as_dict(merged), {(0, 0, 0): 4})

    def test_mismatched_voxel_sizes_are_rejected(self) -> None:
        with self.assertRaises(AssertionError):
            self.merge(VoxelGrid(0.01), VoxelGrid(0.02))

    def test_partial_overlap_takes_b_where_they_meet(self) -> None:
        grid_a = grid_of({(0, 0, 0): 1, (1, 0, 0): 2})
        grid_b = grid_of({(1, 0, 0): 9, (2, 0, 0): 3})
        merged = self.merge(grid_a, grid_b)
        self.assertEqual(as_dict(merged), {(0, 0, 0): 1, (1, 0, 0): 9, (2, 0, 0): 3})

    def test_the_input_grids_are_not_mutated(self) -> None:
        grid_a = grid_of({(0, 0, 0): 1, (1, 0, 0): 2})
        grid_b = grid_of({(1, 0, 0): 9})
        self.merge(grid_a, grid_b)
        self.assertEqual(as_dict(grid_a), {(0, 0, 0): 1, (1, 0, 0): 2})
        self.assertEqual(as_dict(grid_b), {(1, 0, 0): 9})

    def test_merging_into_an_empty_grid_stays_numeric(self) -> None:
        # Regression: concat with an ill-typed empty frame used to poison the
        # merged weights into object dtype.
        merged = self.merge(VoxelGrid(VOXEL_SIZE), grid_of({(0, 0, 0): 2}))
        weights = merged["weight"].to_numpy()
        self.assertTrue(np.issubdtype(weights.dtype, np.floating))


class InhibitionFlipsGridTest(unittest.TestCase):
    def setUp(self) -> None:
        self.merge = InhibitionFlipsGrid()

    def test_without_the_signal_it_is_a_union(self) -> None:
        grid_a = grid_of({(0, 0, 0): 1, (1, 0, 0): 2})
        grid_b = grid_of({(1, 0, 0): 9, (2, 0, 0): 3})
        merged = self.merge(grid_a, grid_b)
        self.assertEqual(as_dict(merged), {(0, 0, 0): 1, (1, 0, 0): 9, (2, 0, 0): 3})

    def test_a_negative_weight_alone_does_not_flip_the_grid(self) -> None:
        grid_a = grid_of({(0, 0, 0): 1, (1, 0, 0): 0.5})
        grid_b = grid_of({(1, 0, 0): 0.9, (2, 0, 0): -0.2})
        merged = self.merge(grid_a, grid_b)
        self.assertEqual(
            as_dict(merged), {(0, 0, 0): 1, (1, 0, 0): 0.9, (2, 0, 0): -0.2}
        )

    def test_the_signal_flips_every_voxel(self) -> None:
        grid_a = grid_of({(0, 0, 0): 1, (1, 0, 0): 0.5})
        grid_b = grid_of({(1, 0, 0): 0.9, (2, 0, 0): 0.2}, inhibit_all=True)
        merged = self.merge(grid_a, grid_b)
        self.assertEqual(
            as_dict(merged),
            {(0, 0, 0): -1, (1, 0, 0): -1, (2, 0, 0): -1},
        )
        self.assertTrue((merged["weight"] == MIN_ATTENTION_WEIGHT).all())

    def test_an_inhibited_voxel_cannot_be_excited_again(self) -> None:
        grid_a = grid_of({(0, 0, 0): -0.4, (1, 0, 0): 0.5})
        grid_b = grid_of({(0, 0, 0): 1.0, (1, 0, 0): 1.0, (2, 0, 0): 1.0})
        merged = self.merge(grid_a, grid_b)
        self.assertEqual(
            as_dict(merged), {(0, 0, 0): -0.4, (1, 0, 0): 1.0, (2, 0, 0): 1.0}
        )

    def test_a_zero_weight_does_not_excite_an_inhibited_voxel(self) -> None:
        merged = self.merge(grid_of({(0, 0, 0): -1}), grid_of({(0, 0, 0): 0.0}))
        self.assertEqual(as_dict(merged), {(0, 0, 0): -1})

    def test_a_fresh_negative_weight_still_replaces_an_inhibited_one(self) -> None:
        merged = self.merge(grid_of({(0, 0, 0): -1}), grid_of({(0, 0, 0): -0.2}))
        self.assertEqual(as_dict(merged), {(0, 0, 0): -0.2})

    def test_inhibition_sticks_after_a_flip(self) -> None:
        flipped = self.merge(
            grid_of({(0, 0, 0): 1, (1, 0, 0): 1}),
            VoxelGrid(VOXEL_SIZE, inhibit_all=True),
        )
        merged = self.merge(flipped, grid_of({(1, 0, 0): 1, (2, 0, 0): 1}))
        self.assertEqual(as_dict(merged), {(0, 0, 0): -1, (1, 0, 0): -1, (2, 0, 0): 1})

    def test_an_empty_signalling_proposal_inhibits_the_whole_grid(self) -> None:
        grid_a = grid_of({(0, 0, 0): 0.7, (1, 0, 0): -0.3})
        merged = self.merge(grid_a, VoxelGrid(VOXEL_SIZE, inhibit_all=True))
        self.assertEqual(as_dict(merged), {(0, 0, 0): -1, (1, 0, 0): -1})

    def test_the_result_does_not_carry_the_signal(self) -> None:
        merged = self.merge(
            grid_of({(0, 0, 0): 0.7}), VoxelGrid(VOXEL_SIZE, inhibit_all=True)
        )
        self.assertFalse(merged.inhibit_all)

    def test_an_empty_proposal_leaves_the_grid_as_is(self) -> None:
        grid_a = grid_of({(0, 0, 0): 0.7})
        merged = self.merge(grid_a, VoxelGrid(VOXEL_SIZE))
        self.assertEqual(as_dict(merged), {(0, 0, 0): 0.7})

    def test_the_input_grids_are_not_mutated(self) -> None:
        grid_a = grid_of({(0, 0, 0): 1})
        grid_b = grid_of({(1, 0, 0): 0.5}, inhibit_all=True)
        self.merge(grid_a, grid_b)
        self.assertEqual(as_dict(grid_a), {(0, 0, 0): 1})
        self.assertEqual(as_dict(grid_b), {(1, 0, 0): 0.5})
        self.assertTrue(grid_b.inhibit_all)
