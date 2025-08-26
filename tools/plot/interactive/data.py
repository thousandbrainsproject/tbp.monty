# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Literal

import numpy as np
import trimesh
from vedo import Mesh

from tbp.monty.frameworks.utils.logging_utils import deserialize_json_chunks

if TYPE_CHECKING:
    from os import PathLike


class YCBMeshLoader:
    """Load YCB object meshes as `vedo.Mesh` with texture.

    This helper looks for meshes in a YCB-like folder structure where each
    object directory ends with the object name and contains
    `google_16k/textured.glb.orig`.

    Attributes:
        data_path: Root directory that contains YCB object folders.
    """

    def __init__(self, data_path: str):
        """Initialize the loader.

        Args:
            data_path: Path to the root directory holding YCB object folders.
        """
        self.data_path = data_path

    def _find_glb_file(self, obj_name: str) -> str:
        """Search for the .glb.orig file of a given YCB object in a directory.

        Args:
            obj_name: The object name to search for (e.g., "potted_meat_can").

        Returns:
            Full path to the .glb.orig file.

        Raises:
            FileNotFoundError: If the .glb.orig file for the object is not found.

        """
        for path in Path(self.data_path).rglob("*"):
            if path.is_dir() and path.name.endswith(obj_name):
                glb_orig_path = path / "google_16k" / "textured.glb.orig"
                if glb_orig_path.exists():
                    return str(glb_orig_path)

        raise FileNotFoundError(
            f"Could not find .glb.orig file for '{obj_name}' in '{self.data_path}'"
        )

    def create_mesh(self, obj_name: str) -> Mesh:
        """Reads a 3D object file in glb format and returns a Vedo Mesh object.

        This loads the GLB via `trimesh`, constructs a `vedo.Mesh` from its vertices
        and faces, applies the base color texture and UVs, recenters the geometry
        to its bounding box mean, and rotates it so the object uses a conventional
        up/front orientation.

        Args:
            obj_name: Name of the object to load.

        Returns:
            vedo.Mesh object with UV texture and transformed orientation.
        """
        file_path = self._find_glb_file(obj_name)
        with open(file_path, "rb") as f:
            mesh = trimesh.load_mesh(f, file_type="glb")

        # create mesh from vertices and faces
        obj = Mesh([mesh.vertices, mesh.faces])

        # add texture
        obj.texture(
            tname=np.array(mesh.visual.material.baseColorTexture),
            tcoords=mesh.visual.uv,
        )

        # Shift to geometry mean and rotate to the up/front of the glb
        obj.shift(-np.mean(obj.bounds().reshape(3, 2), axis=1))
        obj.rotate_x(-90)

        return obj


class DataParser:
    """Parser that navigates nested JSON-like data using a `DataLocator`.

    Attributes:
        data: Parsed JSON-like content loaded from `path`.
    """

    def __init__(self, path: str | PathLike[str]):
        """Initialize the parser by loading the JSON data.

        Args:
            path: Filesystem path to a JSON or JSON-lines file.

        """
        path = os.path.join(path, "detailed_run_stats.json")
        self.data = deserialize_json_chunks(path)

    def extract(self, locator: DataLocator, **kwargs: Any) -> Any:
        """Extract a value by following a `DataLocator` path.

        For each step in `locator`, this resolves the access value from
        `kwargs[step.name]` if provided, otherwise from `step.value`.
        All steps that do not have a fixed `value` must be supplied in `kwargs`.
        Type consistency is enforced based on `step.type`.

        Args:
            locator: A locator describing the navigation path into `self.data`.
            **kwargs: Values for missing steps, keyed by step name.

        Returns:
            The value found at the end of the path.

        Raises:
            ValueError: If a required step is missing or a provided value type
                does not match the step's defined type.
        """
        # Check if all missing steps are provided in kwargs
        for step in locator.missing_steps():
            if step.name not in kwargs:
                raise ValueError(f"Missing required value for step: {step.name}")

            if (step.type == "index" and not isinstance(kwargs[step.name], int)) or (
                step.type == "key" and not isinstance(kwargs[step.name], str)
            ):
                raise ValueError(
                    f"Provided path step value does not match step type for step: ",
                    step.name,
                )

        curr = self.data
        for step in locator.path:
            access_value = kwargs.get(step.name, step.value)
            curr = curr[access_value]
        return curr

    def query(self, locator: DataLocator, **kwargs: Any) -> list[int] | list[str]:
        """Return available values for the first unresolved step in the path.

        Iterates the locator's path using any fixed `step.value` and any
        overrides provided in `kwargs`. When it encounters the first step
        whose access value is not resolved (None and not provided in kwargs),
        it returns the set of valid choices at that point.

        For steps with `type == "index"`, this returns a list of valid indices.
        For steps with `type == "key"`, this returns a list of valid dictionary keys.

        Args:
            locator: A locator describing the navigation path into `self.data`.
            **kwargs: Values for preceding steps, keyed by step name.

        Returns:
            A list of candidate values for the first unresolved step

        Raises:
            ValueError: If there are no missing values to query.
        """
        curr = self.data
        for step in locator.path:
            access_value = kwargs.get(step.name, step.value)
            if access_value is None:
                if step.type == "index":
                    return list(range(len(curr)))  # For list steps, return indices
                elif step.type == "key":
                    return list(curr.keys())  # For dict steps, return keys

            curr = curr[access_value]

        raise ValueError("No missing values to query")


@dataclass
class DataLocatorStep:
    """One step in a data locator path.

    Attributes:
        name: Descriptive name of the step, used as a key into kwargs.
        type: Access type, either "key" for dict indexing or "index" for list indexing.
        value: Optional fixed value to use for this step. If None, callers
            must provide a value in `kwargs` when navigating.
    """

    name: str
    type: Literal["key", "index"]
    value: str | int = None

    @classmethod
    def key(cls, name: str, value: str | None = None) -> DataLocatorStep:
        return cls(name=name, type="key", value=value)

    @classmethod
    def index(cls, name: str, value: int | None = None) -> DataLocatorStep:
        return cls(name=name, type="index", value=value)


@dataclass
class DataLocator:
    """A sequence of path steps that navigates into a nested JSON structure.

    Attributes:
        path: Ordered list of steps describing how to reach a target value.
    """

    path: List[DataLocatorStep]

    def missing_steps(self) -> list[DataLocatorStep]:
        """Return steps that do not have values.

        Returns:
            A list of steps whose `value` is None.
        """
        return [step for step in self.path if step.value is None]

    def extend(self, steps: list[DataLocatorStep]) -> DataLocator:
        """Clone and append multiple steps.

        Returns:
            Cloned DataLocator with extended path steps

        """
        return DataLocator(path=[*self.path, *[deepcopy(s) for s in steps]])

    def __repr__(self) -> str:
        """Return a human-readable representation of the path."""
        steps = " -> ".join(
            f"[{step.name}]" if step.type == "index" else f".{step.name}"
            for step in self.path
        )
        return f"Path: root{steps}"
