# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import numpy.testing as nptest
import torch
from hypothesis import given
from hypothesis import strategies as st

from analysis.models import (
    CURRENT_CHAIN,
    LearnedObject,
    decode_object_ids,
    default_model,
    load_learned_objects,
    object_id_feature,
    resolve_model_path,
)
from analysis.views import Points

object_names = st.text(
    alphabet=st.characters(min_codepoint=48, max_codepoint=122), min_size=1, max_size=12
)


def fake_graph(n: int, mapping: dict[str, list[int]], width: int) -> SimpleNamespace:
    # The attributes analysis.models reads off a GraphObjectModel.
    return SimpleNamespace(
        pos=torch.arange(n * 3, dtype=torch.float32).reshape(n, 3),
        x=torch.arange(n * width, dtype=torch.float32).reshape(n, width),
        feature_mapping=mapping,
    )


def save_fake_model(path: Path, lm_dict: dict) -> Path:
    torch.save({"lm_dict": lm_dict}, path)
    return path


class ObjectIdFeatureTest(unittest.TestCase):
    @given(name=object_names)
    def test_is_the_sum_of_character_codes(self, name: str) -> None:
        self.assertEqual(object_id_feature(name), sum(ord(c) for c in name))


class DecodeObjectIdsTest(unittest.TestCase):
    @given(names=st.lists(object_names, min_size=1, max_size=6, unique=True))
    def test_names_every_candidate_from_its_own_feature(self, names: list[str]) -> None:
        values = np.array([object_id_feature(n) for n in names], dtype=float)

        decoded = decode_object_ids(values, names)

        for name, found in zip(names, decoded):
            self.assertIn(name, found.split("|"))

    def test_joins_colliding_candidates(self) -> None:
        # Anagrams share a feature value.
        self.assertEqual(
            decode_object_ids(np.array([object_id_feature("ab")]), ["ab", "ba"]),
            ["ab|ba"],
        )

    def test_falls_back_to_the_number_for_an_unknown_value(self) -> None:
        self.assertEqual(decode_object_ids(np.array([7.0]), ["cube"]), ["7"])


class ResolveModelPathTest(unittest.TestCase):
    def test_accepts_a_model_file_a_run_dir_and_a_pretrained_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "run" / "pretrained" / "model.pt"
            model.parent.mkdir(parents=True)
            model.touch()
            for given in (model, model.parent, model.parent.parent):
                self.assertEqual(resolve_model_path(given), model)

    def test_raises_for_a_missing_model(self) -> None:
        with self.assertRaises(FileNotFoundError):
            resolve_model_path("/nowhere/at/all")


class DefaultModelTest(unittest.TestCase):
    def test_is_the_newest_trained_stage_of_the_current_chain(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            models_dir = Path(tmp)
            for run_name in CURRENT_CHAIN[:2]:
                model = models_dir / run_name / "pretrained" / "model.pt"
                model.parent.mkdir(parents=True)
                model.touch()

            with patch("analysis.models.MODELS_DIR", models_dir):
                found = default_model()

            self.assertEqual(
                found, models_dir / CURRENT_CHAIN[1] / "pretrained" / "model.pt"
            )

    def test_prefers_the_last_stage_when_every_stage_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            models_dir = Path(tmp)
            for run_name in CURRENT_CHAIN:
                model = models_dir / run_name / "pretrained" / "model.pt"
                model.parent.mkdir(parents=True)
                model.touch()

            with patch("analysis.models.MODELS_DIR", models_dir):
                found = default_model()

            self.assertEqual(
                found, models_dir / CURRENT_CHAIN[-1] / "pretrained" / "model.pt"
            )

    def test_raises_before_any_stage_is_trained(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch(
            "analysis.models.MODELS_DIR", Path(tmp)
        ):
            with self.assertRaises(FileNotFoundError):
                default_model()


class LoadLearnedObjectsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        mapping = {"pose_vectors": [0, 9], "on_object": [9, 10], "hsv": [10, 13]}
        self.model = save_fake_model(
            Path(self.tmp.name) / "model.pt",
            {
                0: {"graph_memory": {"cube": {"patch_0": fake_graph(4, mapping, 13)}}},
                2: {
                    "graph_memory": {
                        "cube_tbp": {
                            "patch_2": fake_graph(2, mapping, 13),
                            "learning_module_0": fake_graph(
                                3, {"pose_vectors": [0, 9], "object_id": [9, 10]}, 10
                            ),
                        }
                    }
                },
            },
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_nests_by_module_object_and_channel(self) -> None:
        learned = load_learned_objects(self.model)

        self.assertEqual(list(learned), ["LM_0", "LM_2"])
        self.assertEqual(
            list(learned["LM_2"]["cube_tbp"]), ["patch_2", "learning_module_0"]
        )
        obj = learned["LM_2"]["cube_tbp"]["learning_module_0"]
        self.assertEqual(
            (obj.lm, obj.object_id, obj.channel),
            ("LM_2", "cube_tbp", "learning_module_0"),
        )

    def test_splits_features_by_their_column_spans(self) -> None:
        obj = load_learned_objects(self.model)["LM_0"]["cube"]["patch_0"]

        self.assertEqual(len(obj), 4)
        self.assertEqual(obj.points["pose_vectors"].shape, (4, 9))
        self.assertEqual(obj.points["hsv"].shape, (4, 3))
        # Single-column features come back flat.
        self.assertEqual(obj.points["on_object"].shape, (4,))
        nptest.assert_array_equal(obj.points["on_object"], [9, 22, 35, 48])

    def test_normals_are_the_first_pose_vector(self) -> None:
        obj = load_learned_objects(self.model)["LM_0"]["cube"]["patch_0"]
        nptest.assert_array_equal(obj.normals, obj.points["pose_vectors"][:, :3])


class LearnedObjectMorphologyTest(unittest.TestCase):
    def setUp(self) -> None:
        # Pose vectors: normal (0..2), then two tangent directions.
        self.pose_vectors = np.array([[1.0, 0, 0, 0, 1, 0, 0, 0, 1]])

    def test_is_the_normal_for_a_3d_channel(self) -> None:
        obj = LearnedObject(
            "LM_0",
            "cube",
            "patch_0",
            Points(np.zeros((1, 3)), {"pose_vectors": self.pose_vectors}),
        )
        self.assertFalse(obj.is_2d)
        nptest.assert_array_equal(obj.morphology, [[1.0, 0, 0]])

    def test_is_the_edge_direction_for_a_2d_channel(self) -> None:
        obj = LearnedObject(
            "LM_1",
            "logo",
            "patch_1",
            Points(
                np.zeros((1, 3)),
                {"pose_vectors": self.pose_vectors, "edge_strength": np.array([0.4])},
            ),
        )
        self.assertTrue(obj.is_2d)
        nptest.assert_array_equal(obj.morphology, [[0.0, 1, 0]])


class LearnedObjectColorsTest(unittest.TestCase):
    def test_converts_stored_hsv_to_rgb(self) -> None:
        obj = LearnedObject(
            "LM_0",
            "cube",
            "patch_0",
            Points(np.zeros((1, 3)), {"hsv": np.array([[0.0, 1.0, 1.0]])}),
        )
        nptest.assert_allclose(obj.colors, [[1.0, 0.0, 0.0]])

    def test_is_gray_without_a_stored_color(self) -> None:
        obj = LearnedObject(
            "LM_2", "cube", "learning_module_0", Points(np.zeros((2, 3)), {})
        )
        nptest.assert_array_equal(obj.colors, np.full((2, 3), 0.5))
