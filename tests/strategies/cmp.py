# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from tbp.monty.cmp import AttentionRegion, Goal
from tests.strategies.arrays import float_array_n_by_3


@st.composite
def goal(
    draw: st.DrawFn,
    location_strategy: st.SearchStrategy[npt.NDArray[np.floating] | None] | None = None,
    morphological_features_strategy: st.SearchStrategy[dict[str, Any] | None]
    | None = None,
    non_morphological_features_strategy: st.SearchStrategy[dict[str, Any] | None]
    | None = None,
    confidence_strategy: st.SearchStrategy[float] | None = None,
    pass_message_strategy: st.SearchStrategy[bool] | None = None,
    sender_id_strategy: st.SearchStrategy[str] | None = None,
    sender_type_strategy: st.SearchStrategy[str] | None = None,
    process_features_in_lm_strategy: st.SearchStrategy[bool] | None = None,
    goal_tolerances_strategy: st.SearchStrategy[dict[str, Any] | None] | None = None,
    info_strategy: st.SearchStrategy[dict[str, Any] | None] | None = None,
) -> Goal:
    if location_strategy is None:
        location_strategy = st.just(None)

    if morphological_features_strategy is None:
        morphological_features_strategy = st.just(None)

    if non_morphological_features_strategy is None:
        non_morphological_features_strategy = st.just(None)

    if confidence_strategy is None:
        confidence_strategy = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)

    if pass_message_strategy is None:
        pass_message_strategy = st.booleans()

    if sender_id_strategy is None:
        sender_id_strategy = st.just("")

    if sender_type_strategy is None:
        sender_type_strategy = st.one_of(st.just("GSG"), st.just("SM"))

    if process_features_in_lm_strategy is None:
        process_features_in_lm_strategy = st.booleans()

    if goal_tolerances_strategy is None:
        goal_tolerances_strategy = st.just(None)

    if info_strategy is None:
        info_strategy = st.just(None)

    return Goal(
        location=draw(location_strategy),
        morphological_features=draw(morphological_features_strategy),
        non_morphological_features=draw(non_morphological_features_strategy),
        confidence=draw(confidence_strategy),
        pass_message=draw(pass_message_strategy),
        sender_id=draw(sender_id_strategy),
        sender_type=draw(sender_type_strategy),
        process_features_in_lm=draw(process_features_in_lm_strategy),
        goal_tolerances=draw(goal_tolerances_strategy),
        info=draw(info_strategy),
    )


@st.composite
def goals(
    draw: st.DrawFn,
    location_strategy: st.SearchStrategy[npt.NDArray[np.floating] | None] | None = None,
    morphological_features_strategy: st.SearchStrategy[dict[str, Any] | None]
    | None = None,
    non_morphological_features_strategy: st.SearchStrategy[dict[str, Any] | None]
    | None = None,
    confidence_strategy: st.SearchStrategy[float] | None = None,
    pass_message_strategy: st.SearchStrategy[bool] | None = None,
    sender_id_strategy: st.SearchStrategy[str] | None = None,
    sender_type_strategy: st.SearchStrategy[str] | None = None,
    process_features_in_lm_strategy: st.SearchStrategy[bool] | None = None,
    goal_tolerances_strategy: st.SearchStrategy[dict[str, Any] | None] | None = None,
    info_strategy: st.SearchStrategy[dict[str, Any] | None] | None = None,
) -> list[Goal]:
    return draw(
        st.lists(
            goal(
                location_strategy=location_strategy,
                morphological_features_strategy=morphological_features_strategy,
                non_morphological_features_strategy=non_morphological_features_strategy,
                confidence_strategy=confidence_strategy,
                pass_message_strategy=pass_message_strategy,
                sender_id_strategy=sender_id_strategy,
                sender_type_strategy=sender_type_strategy,
                process_features_in_lm_strategy=process_features_in_lm_strategy,
                goal_tolerances_strategy=goal_tolerances_strategy,
                info_strategy=info_strategy,
            )
        )
    )


@st.composite
def goals_at(
    draw: st.DrawFn,
    locations: list[npt.NDArray[np.floating]],
) -> list[Goal]:
    out = []
    for location in locations:
        out.append(
            draw(
                goal(
                    location_strategy=st.just(location),
                )
            )
        )
    return out


@st.composite
def attention_region(draw: st.DrawFn) -> AttentionRegion:
    """Returns an AttentionRegion with valid locations and weights."""
    locations = draw(float_array_n_by_3())
    weights = draw(
        arrays(
            dtype=np.float64,
            shape=st.tuples(st.just(locations.shape[0])),
            elements=st.floats(allow_nan=False, allow_infinity=False),
            fill=st.just(0.0),
        )
    )
    return AttentionRegion(locations=locations, weights=weights)


@st.composite
def attention_regions(draw: st.DrawFn) -> list[AttentionRegion]:
    """Returns a list of AttentionRegions with valid locations and weights."""
    return draw(st.lists(attention_region(), min_size=0, max_size=10))
