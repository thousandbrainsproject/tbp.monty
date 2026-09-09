# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import copy
import pickle
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch, sentinel

import numpy as np
import numpy.testing as nptest
import orjson
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from tbp.monty.cmp import AttentionRegion, Goal, encode_goal
from tbp.monty.frameworks.experiments.mode import ExperimentMode
from tbp.monty.frameworks.loggers.monty_handlers import MontyHandler
from tbp.monty.frameworks.loggers.npz_handler import (
    ARRAY_REFERENCE,
    EXPERIMENT_SUBDIRECTORY,
    INDEX_KEY,
    NpzHandler,
    PathFilter,
    _dumps,
    _join_path,
    _map_arrays,
    load_episode,
    materialize,
    resolve,
)

episode_ids = st.integers(0, 10_000)


# -- NpzHandler ---------------------------------------------------------

# The logger passes local (per-mode) episode numbers; the handler maps them
# to global ids through the kwargs. Episode 2 of eval is global episode 5.
LOCAL_EPISODE = 2
GLOBAL_EPISODE = 5
MODE = ExperimentMode.EVAL


def logger_data() -> dict:
    # The shape DetailedGraphMatchingLogger hands to report_episode: the
    # episode's stats row in the BASIC pool under its local episode number
    # and its module blocks in the DETAILED pool under its global id.
    return {
        "BASIC": {
            f"{MODE}_stats": {
                LOCAL_EPISODE: {
                    "LM_0": {"performance": "correct"},
                    "target": {"primary_target_object": "mug"},
                }
            }
        },
        "DETAILED": {
            GLOBAL_EPISODE: {
                "LM_0": {"evidences": [np.arange(4.0)]},
                "SM_1": {"raw_observations": [{"rgba": np.zeros((2, 2, 4))}]},
                "attention_system": {"voxel_size": 0.005},
                "motor_system": {"action_sequence": []},
            }
        },
    }


# The keys of the merged episode: the BASIC row's, then the DETAILED-only ones.
MERGED_KEYS = ["LM_0", "target", "SM_1", "attention_system", "motor_system"]


def report(handler: NpzHandler, output_dir: str) -> None:
    handler.report_episode(
        logger_data(),
        output_dir,
        LOCAL_EPISODE,
        MODE,
        eval_episodes_to_total={LOCAL_EPISODE: GLOBAL_EPISODE},
    )


class NpzHandlerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.output_dir = self.tmp.name

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_is_a_monty_handler_at_the_detailed_level(self) -> None:
        self.assertIsInstance(NpzHandler(), MontyHandler)
        self.assertEqual(NpzHandler.log_level(), "DETAILED")

    def test_close_is_a_no_op(self) -> None:
        self.assertIsNone(NpzHandler().close())

    def test_writes_the_pruned_episode_at_its_path(self) -> None:
        handler = NpzHandler(exclude=["SM_*"])

        with patch.object(handler, "write", return_value=[]) as write_mock:
            report(handler, self.output_dir)

        write_mock.assert_called_once()
        episode_id, episode, path = write_mock.call_args.args
        self.assertEqual(episode_id, GLOBAL_EPISODE)
        self.assertEqual(
            path,
            Path(self.output_dir)
            / EXPERIMENT_SUBDIRECTORY
            / f"episode_{GLOBAL_EPISODE:06d}.npz",
        )
        self.assertEqual(
            list(episode), ["LM_0", "target", "attention_system", "motor_system"]
        )
        nptest.assert_array_equal(episode["LM_0"]["evidences"], [np.arange(4.0)])

    def test_prunes_the_episode_with_its_path_filter(self) -> None:
        handler = NpzHandler()
        with patch.object(
            handler.path_filter, "prune", return_value=sentinel.pruned
        ) as prune_mock, patch.object(handler, "write", return_value=[]) as write_mock:
            report(handler, self.output_dir)

        # The episode holds arrays, so compare structure rather than ==.
        prune_mock.assert_called_once()
        (episode,) = prune_mock.call_args.args
        self.assertEqual(list(episode), MERGED_KEYS)
        self.assertIs(write_mock.call_args.args[1], sentinel.pruned)

    def test_episode_stats_merges_detailed_blocks_over_the_basic_row(self) -> None:
        stats = NpzHandler.episode_stats(
            logger_data(), GLOBAL_EPISODE, LOCAL_EPISODE, MODE
        )

        # The detailed LM block replaces the BASIC performance row, as in
        # DetailedJSONHandler.
        self.assertEqual(list(stats), MERGED_KEYS)
        self.assertEqual(list(stats["LM_0"]), ["evidences"])
        self.assertEqual(stats["target"], {"primary_target_object": "mug"})

    def test_episode_stats_prefers_the_local_episode_in_the_detailed_pool(
        self,
    ) -> None:
        data = logger_data()
        data["DETAILED"][LOCAL_EPISODE] = {"LM_0": {"evidences": "local"}}

        stats = NpzHandler.episode_stats(data, GLOBAL_EPISODE, LOCAL_EPISODE, MODE)

        self.assertEqual(stats["LM_0"], {"evidences": "local"})

    def test_creates_the_episodes_directory(self) -> None:
        handler = NpzHandler()

        with patch.object(handler, "write", return_value=[]):
            report(handler, self.output_dir)

        self.assertTrue((Path(self.output_dir) / EXPERIMENT_SUBDIRECTORY).is_dir())

    def test_moves_a_previous_runs_directory_aside_before_the_first_write(
        self,
    ) -> None:
        episodes_dir = Path(self.output_dir) / EXPERIMENT_SUBDIRECTORY
        episodes_dir.mkdir()
        (episodes_dir / "episode_000000.npz").write_text("previous run")
        old_dir = episodes_dir.with_name(f"{EXPERIMENT_SUBDIRECTORY}_old")
        old_dir.mkdir()
        (old_dir / "stale.npz").write_text("older run")
        handler = NpzHandler()

        report(handler, self.output_dir)

        self.assertEqual(
            sorted(path.name for path in episodes_dir.iterdir()),
            [f"episode_{GLOBAL_EPISODE:06d}.npz"],
        )
        self.assertEqual(
            [path.name for path in old_dir.iterdir()], ["episode_000000.npz"]
        )

    def test_moves_the_directory_aside_only_once_per_run(self) -> None:
        episodes_dir = Path(self.output_dir) / EXPERIMENT_SUBDIRECTORY
        handler = NpzHandler()

        report(handler, self.output_dir)
        with patch(
            "tbp.monty.frameworks.loggers.npz_handler.maybe_rename_existing_dir"
        ) as maybe_rename_existing_dir_mock:
            report(handler, self.output_dir)

        maybe_rename_existing_dir_mock.assert_not_called()
        self.assertEqual(
            sorted(path.name for path in episodes_dir.iterdir()),
            [f"episode_{GLOBAL_EPISODE:06d}.npz"],
        )

    def test_skips_episodes_not_requested(self) -> None:
        handler = NpzHandler(episodes=[GLOBAL_EPISODE + 1])

        with patch.object(handler, "write") as write_mock:
            report(handler, self.output_dir)

        write_mock.assert_not_called()

    def test_writes_an_episode_the_reader_loads_back(self) -> None:
        handler = NpzHandler(include=["target", "LM_0/evidences"])

        report(handler, self.output_dir)

        loaded = load_episode(
            Path(self.output_dir)
            / EXPERIMENT_SUBDIRECTORY
            / f"episode_{GLOBAL_EPISODE:06d}.npz"
        )
        episode = loaded[str(GLOBAL_EPISODE)]
        self.assertEqual(list(loaded), [str(GLOBAL_EPISODE)])
        self.assertEqual(episode["target"], {"primary_target_object": "mug"})
        nptest.assert_array_equal(episode["LM_0"]["evidences"], [[0.0, 1, 2, 3]])


class NpzHandlerEpisodesTest(unittest.TestCase):
    """Which episodes the ``episodes`` argument lets through to ``write``."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.output_dir = self.tmp.name

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def writes(self, handler: NpzHandler, episode_id: int) -> bool:
        data = logger_data()
        data["DETAILED"] = {episode_id: data["DETAILED"][GLOBAL_EPISODE]}
        with patch.object(handler, "write", return_value=[]) as write_mock:
            handler.report_episode(
                data,
                self.output_dir,
                LOCAL_EPISODE,
                MODE,
                eval_episodes_to_total={LOCAL_EPISODE: episode_id},
            )
        return write_mock.called

    @given(episode_id=episode_ids, episodes=st.sampled_from([None, [], ()]))
    def test_writes_every_episode_given_none_or_empty(
        self,
        episode_id: int,
        episodes: list[int] | None,
    ) -> None:
        handler = NpzHandler(episodes)

        self.assertTrue(self.writes(handler, episode_id))

    @given(episode_id=episode_ids, others=st.sets(episode_ids, min_size=1))
    def test_writes_only_the_episodes_listed(
        self,
        episode_id: int,
        others: set[int],
    ) -> None:
        # `others` is non-empty so the unlisted case is not the save-all case.
        unlisted = NpzHandler(episodes=(others - {episode_id}) or {-1})
        listed = NpzHandler(episodes=others | {episode_id})

        self.assertFalse(self.writes(unlisted, episode_id))
        self.assertTrue(self.writes(listed, episode_id))


# -- paths: PathFilter, prune --------------------------------------------------

# Bounds on the generated paths: how many segments a path has and how long
# each segment is. Segments avoid the separator and glob metacharacters so a
# segment used verbatim as a pattern matches only itself.
MAX_SEGMENTS = 5
MAX_SEGMENT_LENGTH = 8
SEGMENT_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789_"

segments = st.text(SEGMENT_ALPHABET, min_size=1, max_size=MAX_SEGMENT_LENGTH)
segment_lists = st.lists(segments, min_size=1, max_size=MAX_SEGMENTS)
paths = segment_lists.map("/".join)


@st.composite
def paths_with_an_ancestor(draw: st.DrawFn) -> tuple[str, str]:
    # A path and one of its ancestors-or-self, i.e. a prefix of its segments.
    path_segments = draw(segment_lists)
    depth = draw(st.integers(1, len(path_segments)))
    return "/".join(path_segments), "/".join(path_segments[:depth])


@st.composite
def paths_with_a_glob(draw: st.DrawFn) -> tuple[str, str]:
    # A path and a pattern that matches it with one segment replaced by "*".
    path_segments = draw(segment_lists)
    index = draw(st.integers(0, len(path_segments) - 1))
    pattern_segments = [*path_segments]
    pattern_segments[index] = "*"
    return "/".join(path_segments), "/".join(pattern_segments)


class JoinPathTest(unittest.TestCase):
    @given(key=segments)
    def test_returns_the_key_given_the_root_path(self, key: str) -> None:
        self.assertEqual(_join_path("", key), key)

    @given(path=paths, key=segments)
    def test_returns_the_path_and_key_separated(self, path: str, key: str) -> None:
        self.assertEqual(_join_path(path, key), f"{path}/{key}")

    @given(path=paths, key=st.integers(0, 100))
    def test_returns_integer_keys_as_text(self, path: str, key: int) -> None:
        self.assertEqual(_join_path(path, key), f"{path}/{key}")


class PathFilterMatchesTest(unittest.TestCase):
    """A pattern matches the exact path it names, literally or by glob."""

    @given(path=paths)
    def test_include_and_exclude_match_nothing_without_patterns(
        self,
        path: str,
    ) -> None:
        self.assertFalse(PathFilter().matches_include(path))
        self.assertFalse(PathFilter().matches_exclude(path))

    @given(path=paths)
    def test_a_literal_pattern_matches_its_own_path(self, path: str) -> None:
        self.assertTrue(PathFilter(include=[path]).matches_include(path))
        self.assertTrue(PathFilter(exclude=[path]).matches_exclude(path))

    @given(path_and_pattern=paths_with_a_glob())
    def test_a_glob_matches_the_path(self, path_and_pattern: tuple[str, str]) -> None:
        path, pattern = path_and_pattern

        self.assertTrue(PathFilter(include=[pattern]).matches_include(path))

    @given(path_and_ancestor=paths_with_an_ancestor())
    def test_a_pattern_does_not_match_a_descendant_of_its_path(
        self,
        path_and_ancestor: tuple[str, str],
    ) -> None:
        """Ancestor matches are prune's job, carried down the walk."""
        path, ancestor = path_and_ancestor
        if ancestor == path:
            return

        self.assertFalse(PathFilter(include=[ancestor]).matches_include(path))


# Scalar-leaved trees: dicts and lists of JSON scalars, compared with
# assertEqual. Arrays and objects get their own tests below.
MAX_TREE_LEAVES = 20

scalars = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(-1000, 1000),
    st.floats(allow_nan=False, allow_infinity=False),
    st.text(SEGMENT_ALPHABET, max_size=MAX_SEGMENT_LENGTH),
)
trees = st.recursive(
    scalars,
    lambda children: st.one_of(
        st.dictionaries(segments, children, max_size=4),
        st.lists(children, max_size=4),
    ),
    max_leaves=MAX_TREE_LEAVES,
)
# Trees with at least one leaf, so the filter cannot empty them by accident.
nonempty_trees = st.dictionaries(segments, trees, min_size=1, max_size=4)


class PruneTest(unittest.TestCase):
    @given(tree=nonempty_trees)
    def test_returns_the_tree_given_no_patterns(
        self,
        tree: dict[str, Any],
    ) -> None:
        self.assertEqual(PathFilter().prune(tree), tree)

    @given(kept=nonempty_trees, dropped=nonempty_trees)
    def test_drops_the_excluded_subtree(
        self,
        kept: dict[str, Any],
        dropped: dict[str, Any],
    ) -> None:
        tree = {"kept": kept, "dropped": dropped}

        self.assertEqual(PathFilter(exclude=["dropped"]).prune(tree), {"kept": kept})

    @given(kept=nonempty_trees, dropped=nonempty_trees)
    def test_drops_a_subtree_excluded_through_an_ancestor_path(
        self,
        kept: dict[str, Any],
        dropped: dict[str, Any],
    ) -> None:
        tree = {"a": {"b": dropped, "c": kept}}

        self.assertEqual(PathFilter(exclude=["a/b"]).prune(tree), {"a": {"c": kept}})

    @given(kept=nonempty_trees, dropped=nonempty_trees)
    def test_drops_containers_the_filter_emptied(
        self,
        kept: dict[str, Any],
        dropped: dict[str, Any],
    ) -> None:
        tree = {"a": {"b": {"c": dropped}}, "d": kept}

        self.assertEqual(PathFilter(exclude=["a/b/c"]).prune(tree), {"d": kept})

    @given(kept=nonempty_trees, dropped=nonempty_trees)
    def test_keeps_only_the_included_paths(
        self,
        kept: dict[str, Any],
        dropped: dict[str, Any],
    ) -> None:
        tree = {"a": {"b": kept, "c": dropped}, "d": dropped}

        self.assertEqual(PathFilter(include=["a/b"]).prune(tree), {"a": {"b": kept}})

    @given(tree=nonempty_trees)
    def test_returns_an_empty_dict_given_everything_excluded(
        self,
        tree: dict[str, Any],
    ) -> None:
        self.assertEqual(PathFilter(exclude=["*"]).prune(tree), {})

    def test_keeps_included_empty_containers(self) -> None:
        tree = {"a": {}, "b": [], "c": ()}

        self.assertEqual(PathFilter().prune(tree), tree)

    @given(kept=nonempty_trees, dropped=nonempty_trees)
    def test_addresses_list_items_by_index(
        self,
        kept: dict[str, Any],
        dropped: dict[str, Any],
    ) -> None:
        tree = {"modules": [kept, dropped, kept]}

        self.assertEqual(
            PathFilter(exclude=["modules/1"]).prune(tree),
            {"modules": [kept, None, kept]},
        )

    @given(kept=nonempty_trees, dropped=nonempty_trees)
    def test_a_star_reaches_the_same_key_in_every_list_item(
        self,
        kept: dict[str, Any],
        dropped: dict[str, Any],
    ) -> None:
        tree = {"steps": [{"kept": kept, "dropped": dropped}] * 3}

        self.assertEqual(
            PathFilter(exclude=["steps/*/dropped"]).prune(tree),
            {"steps": [{"kept": kept}] * 3},
        )

    @given(kept=nonempty_trees, dropped=nonempty_trees)
    def test_substitutes_none_for_a_dropped_list_item(
        self,
        kept: dict[str, Any],
        dropped: dict[str, Any],
    ) -> None:
        tree = {"steps": [{"dropped": dropped}, {"kept": kept}]}

        self.assertEqual(
            PathFilter(exclude=["steps/*/dropped"]).prune(tree),
            {"steps": [None, {"kept": kept}]},
        )

    @given(kept=nonempty_trees, dropped=nonempty_trees)
    def test_drops_a_list_whose_items_were_all_dropped(
        self,
        kept: dict[str, Any],
        dropped: dict[str, Any],
    ) -> None:
        tree = {"steps": [{"dropped": dropped}] * 2, "kept": kept}

        self.assertEqual(
            PathFilter(exclude=["steps/*/dropped"]).prune(tree), {"kept": kept}
        )

    def test_keeps_arrays_as_arrays(self) -> None:
        array = np.arange(6).reshape(2, 3)

        pruned = PathFilter().prune({"a": array})

        self.assertIs(pruned["a"], array)

    def test_keeps_a_list_of_arrays_as_a_leaf(self) -> None:
        maps = [np.zeros((2, 2)), None, np.ones((2, 2))]

        pruned = PathFilter().prune({"maps": maps})

        self.assertIs(pruned["maps"], maps)

    def test_expands_dataclasses_so_the_filter_can_see_inside(self) -> None:
        region = AttentionRegion.uniform([[0.0, 0, 0]], 1.0, sender_id="SM_0")

        pruned = PathFilter(exclude=["regions/*/*/weights"]).prune(
            {"regions": [[region]]}
        )

        self.assertEqual(
            list(pruned["regions"][0][0]), ["locations", "sender_id", "inhibit_all"]
        )
        nptest.assert_array_equal(
            pruned["regions"][0][0]["locations"], region.locations
        )
        self.assertEqual(pruned["regions"][0][0]["sender_id"], "SM_0")

    def test_expands_goals_through_their_registered_encoder(self) -> None:
        goal = Goal(
            location=np.zeros(3),
            morphological_features=None,
            non_morphological_features=None,
            confidence=1.0,
            pass_message=False,
            sender_id="SM_0",
            sender_type="SM",
            process_features_in_lm=False,
            goal_tolerances=None,
            info={"passed_attention_filter": True},
        )
        expected = encode_goal(goal)
        del expected["info"]

        pruned = PathFilter(exclude=["goals/*/info"]).prune({"goals": [goal]})

        self.assertEqual(list(pruned["goals"][0]), list(expected))
        nptest.assert_array_equal(pruned["goals"][0].pop("location"), np.zeros(3))
        expected.pop("location")
        self.assertEqual(pruned["goals"][0], expected)

    def test_calls_buffer_encoder_default_given_an_unknown_object(self) -> None:
        with patch(
            "tbp.monty.frameworks.loggers.npz_handler.BufferEncoder.default",
            return_value={"expanded": 1},
        ) as default_mock:
            PathFilter().prune({"a": sentinel.unknown})

        default_mock.assert_called_once_with(sentinel.unknown)

    def test_keeps_an_object_nothing_can_expand(self) -> None:
        pruned = PathFilter().prune({"a": sentinel.opaque})

        self.assertIs(pruned["a"], sentinel.opaque)


# -- storage: dumps, map_arrays, NpzHandler.write ----------------------

# Bounds on the arrays the storage tests write: every array has at most this
# many elements, and the npz-member threshold sits inside that range so a draw
# can land on either side of it.
MAX_ARRAY_LENGTH = 64
# Finite values that survive a float32 cast; JSON has no inf or nan.
finite_floats = st.floats(-1e6, 1e6, width=64, allow_nan=False)

float_arrays = arrays(
    np.float64, st.integers(1, MAX_ARRAY_LENGTH), elements=finite_floats
)
int_arrays = arrays(np.int64, st.integers(1, MAX_ARRAY_LENGTH))


class DumpsTest(unittest.TestCase):
    @given(array=float_arrays)
    def test_serializes_arrays_as_json_lists(self, array: np.ndarray) -> None:
        self.assertEqual(orjson.loads(_dumps({"a": array})), {"a": array.tolist()})

    def test_serializes_integer_keys_as_strings(self) -> None:
        self.assertEqual(_dumps({7: 1}), b'{"7":1}')

    def test_serializes_nan_as_null(self) -> None:
        self.assertEqual(_dumps([float("nan"), np.float64("nan")]), b"[null,null]")

    def test_serializes_dataclasses_through_the_buffer_encoder(self) -> None:
        region = AttentionRegion.uniform([[0.0, 0, 0]], 1.0, sender_id="SM_0")

        self.assertEqual(
            orjson.loads(_dumps(region)),
            {
                "locations": [[0.0, 0, 0]],
                "weights": [1.0],
                "sender_id": "SM_0",
                "inhibit_all": False,
            },
        )


class MapArraysTest(unittest.TestCase):
    def test_calls_fn_with_each_array_and_its_list_indexed_path(self) -> None:
        tree = {"a": [np.zeros(1), {"b": np.ones(1)}], "c": np.full(1, 2.0)}
        calls = []

        _map_arrays(tree, lambda array, path: calls.append((path, array)))

        self.assertEqual([path for path, _ in calls], ["a/0", "a/1/b", "c"])
        nptest.assert_array_equal([array[0] for _, array in calls], [0.0, 1.0, 2.0])

    def test_returns_the_tree_with_every_array_replaced(self) -> None:
        tree = {"a": [np.zeros(1), {"b": np.ones(1)}], "c": np.full(1, 2.0)}

        mapped = _map_arrays(tree, lambda _, path: path)

        self.assertEqual(mapped, {"a": ["a/0", {"b": "a/1/b"}], "c": "c"})

    def test_returns_scalar_leaves_unchanged(self) -> None:
        tree = {"a": [1, "two", None, {"b": 3.0}], "c": sentinel.opaque}

        self.assertEqual(_map_arrays(tree, lambda *_: sentinel.array), tree)


class NpzHandlerWriteTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "episode_000000.npz"
        # float_dtype=None keeps the arrays as drawn; the default is tested
        # on its own.
        self.handler = NpzHandler(float_dtype=None)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def read_index(self) -> dict:
        with np.load(self.path) as npz:
            return orjson.loads(npz[INDEX_KEY].tobytes())

    def array_keys(self) -> list[str]:
        with np.load(self.path) as npz:
            return sorted(key for key in npz.files if key != INDEX_KEY)

    def test_writes_the_json_index_as_a_uint8_member(self) -> None:
        self.handler.write(0, {"a": 1}, self.path)

        with np.load(self.path) as npz:
            self.assertEqual(npz.files, [INDEX_KEY])
            self.assertEqual(npz[INDEX_KEY].dtype, np.uint8)
            self.assertEqual(orjson.loads(npz[INDEX_KEY].tobytes()), {"0": {"a": 1}})

    @given(episode_id=episode_ids)
    def test_keys_the_stats_by_episode_id(self, episode_id: int) -> None:
        self.handler.write(episode_id, {"name": "sensor"}, self.path)

        self.assertEqual(self.read_index(), {str(episode_id): {"name": "sensor"}})

    @given(episode_id=episode_ids, array=float_arrays)
    def test_replaces_arrays_with_a_reference_to_their_member(
        self,
        episode_id: int,
        array: np.ndarray,
    ) -> None:
        self.handler.write(episode_id, {"SM_0": {"maps": [array]}}, self.path)

        self.assertEqual(
            self.read_index(),
            {
                str(episode_id): {
                    "SM_0": {
                        "maps": [
                            {
                                ARRAY_REFERENCE: {
                                    "key": "SM_0/maps/0",
                                    "dtype": "<f8",
                                    "shape": [len(array)],
                                }
                            }
                        ]
                    }
                }
            },
        )
        with np.load(self.path) as npz:
            nptest.assert_array_equal(npz["SM_0/maps/0"], array)

    def test_references_even_single_element_arrays(self) -> None:
        self.handler.write(0, {"a": np.zeros(1)}, self.path)

        self.assertEqual(self.array_keys(), ["a"])
        self.assertEqual(self.read_index()["0"]["a"][ARRAY_REFERENCE]["shape"], [1])

    @given(array=float_arrays)
    def test_casts_float_arrays_to_float_dtype(self, array: np.ndarray) -> None:
        handler = NpzHandler(float_dtype="float32")

        handler.write(0, {"a": array}, self.path)

        self.assertEqual(self.read_index()["0"]["a"][ARRAY_REFERENCE]["dtype"], "<f4")
        with np.load(self.path) as npz:
            nptest.assert_array_equal(npz["a"], array.astype(np.float32))

    @given(array=float_arrays)
    def test_casts_float_arrays_to_float32_by_default(self, array: np.ndarray) -> None:
        NpzHandler().write(0, {"a": array}, self.path)

        with np.load(self.path) as npz:
            self.assertEqual(npz["a"].dtype, np.float32)

    def test_keeps_the_array_dtype(self) -> None:
        array = np.arange(8, dtype=np.uint8)

        self.handler.write(0, {"rgba": array}, self.path)

        with np.load(self.path) as npz:
            self.assertEqual(npz["rgba"].dtype, np.uint8)

    def test_overwrites_an_existing_file_in_place(self) -> None:
        self.handler.write(0, {"a": np.zeros(8)}, self.path)

        self.handler.write(0, {"a": np.ones(8)}, self.path)

        self.assertEqual(
            [path.name for path in self.path.parent.iterdir()], ["episode_000000.npz"]
        )
        with np.load(self.path) as npz:
            nptest.assert_array_equal(npz["a"], np.ones(8))

    def test_leaves_scalars_and_strings_in_the_json(self) -> None:
        stats = {"a": {"n": 1, "s": "text", "none": None, "flags": [True, False]}}

        self.handler.write(0, stats, self.path)

        self.assertEqual(self.read_index(), {"0": stats})


# -- reader: load_episode, resolve ---------------------------------------------


def reference(key: str) -> dict:
    # The dtype and shape are informational; the reader takes them from the file.
    return {ARRAY_REFERENCE: {"key": key, "dtype": "<f8", "shape": [1]}}


class LoadEpisodeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "episode_000000.npz"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_loads_an_episode_the_handler_wrote(self) -> None:
        stats = {"SM_0": {"maps": [np.arange(16.0), np.arange(16.0) * 2], "name": "s"}}
        NpzHandler().write(7, stats, self.path)

        loaded = load_episode(self.path)

        self.assertEqual(list(loaded), ["7"])
        self.assertEqual(loaded["7"]["SM_0"]["name"], "s")
        nptest.assert_array_equal(loaded["7"]["SM_0"]["maps"], stats["SM_0"]["maps"])

    def test_loads_every_array_before_closing_the_file(self) -> None:
        NpzHandler().write(7, {"a": np.arange(4.0)}, self.path)

        loaded = load_episode(self.path)

        self.path.unlink()
        nptest.assert_array_equal(loaded["7"]["a"], np.arange(4.0))


class LazyLoadTest(unittest.TestCase):
    """``load_episode(lazy=True)`` reads members on access and acts like dicts."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "episode_000000.npz"
        self.stats = {
            "SM_0": {
                "raw_observations": [
                    {"rgba": np.full((4, 4), step)} for step in range(3)
                ],
                "maps": [np.arange(16.0), np.arange(16.0) * 2],
                "name": "s",
            },
            "target": {"object": "mug", "position": [0.0, 1.0, 2.0]},
        }
        NpzHandler(float_dtype=None).write(0, self.stats, self.path)
        self.lazy = load_episode(self.path, lazy=True)["0"]

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_reads_members_on_access_only(self) -> None:
        reads: list[str] = []
        original = np.lib.npyio.NpzFile.__getitem__

        def counting(npz: np.lib.npyio.NpzFile, key: str) -> np.ndarray:
            reads.append(key)
            return original(npz, key)

        with patch.object(np.lib.npyio.NpzFile, "__getitem__", counting):
            lazy = load_episode(self.path, lazy=True)["0"]
            self.assertEqual(reads, [INDEX_KEY])
            lazy["target"]
            self.assertEqual(reads, [INDEX_KEY])
            lazy["SM_0"]["raw_observations"][1]["rgba"]
            lazy["SM_0"]["raw_observations"][1]["rgba"]

        self.assertEqual(reads, [INDEX_KEY, "SM_0/raw_observations/1/rgba"])

    def test_equals_the_eager_tree_everywhere(self) -> None:
        eager = load_episode(self.path)["0"]

        self.assertIsInstance(self.lazy, dict)
        self.assertEqual(self.lazy["target"], eager["target"])
        nptest.assert_array_equal(
            self.lazy["SM_0"]["raw_observations"][2]["rgba"],
            eager["SM_0"]["raw_observations"][2]["rgba"],
        )
        nptest.assert_array_equal(self.lazy["SM_0"]["maps"], eager["SM_0"]["maps"])

    def test_whole_dict_access_resolves(self) -> None:
        block = self.lazy["SM_0"]

        self.assertEqual(list(block), ["raw_observations", "maps", "name"])
        self.assertEqual(block.get("name"), "s")
        self.assertIsNone(block.get("missing"))
        values = dict(block.items())
        nptest.assert_array_equal(values["maps"][0], np.arange(16.0))
        nptest.assert_array_equal(dict(block)["maps"][1], np.arange(16.0) * 2)
        nptest.assert_array_equal({**block}["maps"][0], np.arange(16.0))

    def test_lists_iterate_slice_and_convert(self) -> None:
        observations = self.lazy["SM_0"]["raw_observations"]

        self.assertEqual(len(observations), 3)
        self.assertEqual([o["rgba"][0, 0] for o in observations], [0, 1, 2])
        self.assertEqual(observations[1:][0]["rgba"][0, 0], 1)
        nptest.assert_array_equal(np.stack(self.lazy["SM_0"]["maps"]).shape, (2, 16))

    def test_copies_and_pickles_as_plain_loaded_dicts(self) -> None:
        copied = copy.deepcopy(self.lazy)
        unpickled = pickle.loads(pickle.dumps(self.lazy))  # noqa: S301

        for tree in (copied, unpickled):
            self.assertIs(type(tree), dict)
            self.assertIs(type(tree["SM_0"]["raw_observations"]), list)
            nptest.assert_array_equal(
                tree["SM_0"]["raw_observations"][0]["rgba"], np.zeros((4, 4))
            )

    def test_materialize_gives_the_eager_tree(self) -> None:
        plain = materialize(self.lazy)

        self.assertIs(type(plain), dict)
        self.assertIs(type(plain["SM_0"]["raw_observations"][0]), dict)
        nptest.assert_array_equal(plain["SM_0"]["maps"][0], np.arange(16.0))


class ResolveTest(unittest.TestCase):
    def setUp(self) -> None:
        self.arrays = {"x": np.arange(3), "y": np.ones(2), "z": np.zeros(2)}

    def test_replaces_each_reference_with_the_array_it_names(self) -> None:
        tree = {"x": reference("x"), "nested": [reference("z"), 1]}

        resolved = resolve(tree, self.arrays)

        nptest.assert_array_equal(resolved["x"], np.arange(3))
        nptest.assert_array_equal(resolved["nested"][0], np.zeros(2))
        self.assertEqual(resolved["nested"][1], 1)

    def test_returns_scalar_leaves_unchanged(self) -> None:
        tree = {"a": [1, "two", None, {"b": 3.0}], ARRAY_REFERENCE: 1, "c": {}}

        self.assertEqual(resolve(tree, self.arrays), tree)
