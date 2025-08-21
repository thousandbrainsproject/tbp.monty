# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Literal

from tbp.monty.frameworks.utils.logging_utils import deserialize_json_chunks

if TYPE_CHECKING:
    from os import PathLike


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
        self.data = deserialize_json_chunks(path)

    def extract(self, locator: DataLocator, **kwargs: dict[str, Any]) -> Any:
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

    def query(
        self, locator: DataLocator, **kwargs: dict[str, Any]
    ) -> list[int] | list[str]:
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

    def __repr__(self) -> str:
        """Return a human-readable representation of the path."""
        steps = " -> ".join(
            f"[{step.name}]" if step.type == "index" else f".{step.name}"
            for step in self.path
        )
        return f"Path: root{steps}"
