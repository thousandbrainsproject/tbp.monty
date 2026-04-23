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
from unittest.mock import Mock, patch

from unittest_parametrize import ParametrizedTestCase, parametrize

from tbp.monty.cmp import Goal
from tbp.monty.frameworks.models.motor_policies import MotorPolicyResult, PolicyStatus
from tbp.monty.frameworks.models.motor_policy_selectors import (
    DistantPolicySelector,
    SinglePolicySelector,
    highest_confidence_goal,
)


class HighestConfidenceGoalTest(unittest.TestCase):
    def test_returns_goal_with_highest_confidence(self):
        best_goal = Mock(confidence=0.9)
        second_best_goal = Mock(confidence=0.8)
        goals = [best_goal, second_best_goal]
        goal = highest_confidence_goal(goals)
        self.assertIs(goal, best_goal)

    def test_returns_first_goal_when_confidence_tied(self):
        first_goal = Mock(confidence=0.9)
        second_goal = Mock(confidence=0.9)
        goals = [first_goal, second_goal]
        goal = highest_confidence_goal(goals)
        self.assertIs(goal, first_goal)


class SinglePolicySelectorTest(unittest.TestCase):
    def setUp(self):
        self.policy = Mock()
        self.selector = SinglePolicySelector(self.policy)
        self.ctx = Mock()
        self.observations = Mock()
        self.state = Mock()
        self.percept = Mock()
        self.expected_result = Mock()
        self.policy.return_value = self.expected_result

    def test_delegates_to_configured_policy(self):
        self.selector(self.ctx, self.observations, self.state, self.percept, [])
        self.policy.assert_called_once_with(
            self.ctx,
            self.observations,
            self.state,
            self.percept,
            None,
        )

    def test_returns_result_from_policy(self):
        result = self.selector(
            self.ctx, self.observations, self.state, self.percept, []
        )
        self.assertIs(result, self.expected_result)

    @patch("tbp.monty.frameworks.models.motor_policy_selectors.highest_confidence_goal")
    def test_calls_policy_with_goal_of_highest_confidence(
        self,
        highest_confidence_goal_mock: Mock,
    ) -> None:
        best_goal = Mock()
        goals = Mock()
        highest_confidence_goal_mock.return_value = best_goal

        self.selector(
            self.ctx,
            self.observations,
            self.state,
            self.percept,
            goals,
        )

        highest_confidence_goal_mock.assert_called_once_with(goals)
        self.policy.assert_called_once_with(
            self.ctx,
            self.observations,
            self.state,
            self.percept,
            best_goal,
        )

    def test_pre_episode_calls_pre_episode_on_policy(self):
        motor_system = Mock()
        self.selector.pre_episode(motor_system)
        self.policy.pre_episode.assert_called_once_with(motor_system)

    def test_state_dict_includes_policy_state_dict(self):
        state_dict = Mock()
        self.policy.state_dict.return_value = state_dict
        self.assertIs(self.selector.state_dict()["policy"], state_dict)


class DistantPolicySelectorTest(ParametrizedTestCase):
    def setUp(self):
        self.jump_to_goal = Mock()
        self.look_at_goal = Mock()
        self.default_policy = Mock()
        self.selector = DistantPolicySelector(
            self.jump_to_goal, self.look_at_goal, self.default_policy
        )
        self.ctx = Mock()
        self.observations = Mock()
        self.state = Mock()
        self.percept = Mock()
        # self.goals = [Mock(confidence=0.9), Mock(confidence=0.8)]

    def test_returns_default_policy_result_when_no_goals_are_present(self):
        default_policy_result = Mock(spec=MotorPolicyResult)
        self.default_policy.return_value = default_policy_result

        result = self.selector(
            self.ctx,
            self.observations,
            self.state,
            self.percept,
            [],
        )

        self.default_policy.assert_called_once_with(
            self.ctx,
            self.observations,
            self.state,
            self.percept,
            None,
        )
        self.assertIs(result, default_policy_result)

    def test_returns_jump_to_goal_result_when_gsg_goal_is_present(self):
        gsg_goal = Mock(sender_type="GSG")
        goals = [
            Mock(sender_type="SM"),
            gsg_goal,
            Mock(sender_type="SM"),
        ]
        jump_to_goal_result = Mock(spec=MotorPolicyResult)
        self.jump_to_goal.return_value = jump_to_goal_result

        result = self.selector(
            self.ctx,
            self.observations,
            self.state,
            self.percept,
            goals,
        )

        self.jump_to_goal.assert_called_once_with(
            self.ctx, self.observations, self.state, self.percept, gsg_goal
        )
        self.assertIs(result, jump_to_goal_result)

    @patch("tbp.monty.frameworks.models.motor_policy_selectors.highest_confidence_goal")
    def test_invokes_jump_to_goal_with_highest_confidence_gsg_goal(
        self, highest_confidence_goal_mock: Mock
    ):
        best_gsg_goal = Mock(sender_type="GSG")
        gsg_goal = Mock(sender_type="GSG")
        goals = [
            Mock(sender_type="SM"),
            gsg_goal,
            best_gsg_goal,
            Mock(sender_type="SM"),
        ]
        highest_confidence_goal_mock.return_value = best_gsg_goal
        jump_to_goal_result = Mock(spec=MotorPolicyResult)
        self.jump_to_goal.return_value = jump_to_goal_result

        result = self.selector(
            self.ctx,
            self.observations,
            self.state,
            self.percept,
            goals,
        )

        highest_confidence_goal_mock.assert_called_once_with(
            Goals([gsg_goal, best_gsg_goal])
        )
        self.jump_to_goal.assert_called_once_with(
            self.ctx, self.observations, self.state, self.percept, best_gsg_goal
        )
        self.assertIs(result, jump_to_goal_result)

    def test_returns_look_at_goal_result_when_only_sm_goals_are_present(self):
        goals = [
            Mock(sender_type="SM", confidence=0.9),
            Mock(sender_type="SM", confidence=0.8),
        ]
        look_at_goal_result = Mock(spec=MotorPolicyResult)
        self.look_at_goal.return_value = look_at_goal_result

        result = self.selector(
            self.ctx,
            self.observations,
            self.state,
            self.percept,
            goals,
        )

        self.look_at_goal.assert_called_once_with(
            self.ctx, self.observations, self.state, self.percept, goals[0]
        )
        self.assertIs(result, look_at_goal_result)

    @patch("tbp.monty.frameworks.models.motor_policy_selectors.highest_confidence_goal")
    def test_invokes_look_at_goal_with_highest_confidence_gsg_goal(
        self, highest_confidence_goal_mock: Mock
    ):
        best_sm_goal = Mock(sender_type="SM")
        sm_goal = Mock(sender_type="SM")
        goals = [
            sm_goal,
            best_sm_goal,
        ]
        highest_confidence_goal_mock.return_value = best_sm_goal
        look_at_goal_result = Mock(spec=MotorPolicyResult)
        self.look_at_goal.return_value = look_at_goal_result

        result = self.selector(
            self.ctx,
            self.observations,
            self.state,
            self.percept,
            goals,
        )

        highest_confidence_goal_mock.assert_called_once_with(
            Goals([sm_goal, best_sm_goal])
        )
        self.look_at_goal.assert_called_once_with(
            self.ctx, self.observations, self.state, self.percept, best_sm_goal
        )
        self.assertIs(result, look_at_goal_result)

    @parametrize(
        ("undo", "new_lm_goal", "new_sm_goal"),
        [
            #   undo=False, goals=[]
            #   jump_to_goal_result: actions=[], status=READY
            #   default_policy_result: actions=[...], status=READY
            #   returned: default_policy_result
            (False, False, False),
            #   undo=False, goals=[sm_goal, sm_goal, ...]
            #   jump_to_goal_result: actions=[], status=READY
            #   look_at_goal_result: actions=[...], status=READY
            #   returned: look_at_goal_result
            (False, False, True),
            #   undo=False, goals=[lm_goal, lm_goal, ...]
            #   jump_to_goal_result: actions=[...], status=IN_PROGRESS
            #   returned: jump_to_goal_result
            (False, True, False),
            #   undo=False, goals=[lm_goal, lm_goal, sm_goal, sm_goal, ...]
            #   jump_to_goal_result: actions=[...], status=IN_PROGRESS
            #   returned: jump_to_goal_result
            (False, True, True),
            #   undo=True, goals=[]
            #   jump_to_goal_result: actions=[...], status=READY
            #   returned: jump_to_goal_result
            (True, False, False),
            #   undo=True, goals=[sm_goal, sm_goal, ...]
            #   jump_to_goal_result: actions=[...], status=READY
            #   returned: jump_to_goal_result
            (True, False, True),
            #   undo=True, goals=[lm_goal, lm_goal, ...]
            #   jump_to_goal_result: actions=[...], status=READY
            #   returned: jump_to_goal_result
            (True, True, False),
            #   undo=True, goals=[lm_goal, lm_goal, sm_goal, sm_goal, ...]
            #   jump_to_goal_result: actions=[...], status=READY
            #   returned: jump_to_goal_result
            (True, True, True),
        ],
    )
    @patch("tbp.monty.frameworks.models.motor_policy_selectors.highest_confidence_goal")
    def test_post_jump_behavior(
        self,
        highest_confidence_goal_mock: Mock,
        undo: bool,
        new_lm_goal: bool,
        new_sm_goal: bool,
    ) -> None:
        """Test post-jump behavior.

        We simulate needing to undo by having the jump-to-goal mock
        return a result with (non-empty) actions and IN_PROGRESS status.
        """
        # Put the selector into a jumping state.
        init_goal = Mock(sender_type="GSG", confidence=1.0)
        init_result_mock = Mock(
            spec=MotorPolicyResult,
            actions=[Mock(), Mock()],
            status=PolicyStatus.IN_PROGRESS,
        )
        self.jump_to_goal.return_value = init_result_mock
        highest_confidence_goal_mock.return_value = init_goal

        init_result = self.selector(
            self.ctx,
            self.observations,
            self.state,
            self.percept,
            [init_goal],
        )

        self.jump_to_goal.assert_called_once_with(
            self.ctx,
            self.observations,
            self.state,
            self.percept,
            init_goal,
        )
        self.assertIs(init_result, init_result_mock)

        # Setup inputs and outputs for the post-jump step.
        lm_goal = Mock(sender_type="GSG", confidence=1.0) if new_lm_goal else None
        if lm_goal is not None:
            highest_confidence_goal_mock.return_value = lm_goal
        goals: list[Goal] = list(
            filter(
                lambda g: g is not None,
                [
                    lm_goal,
                    Mock(sender_type="SM", confidence=1.0) if new_sm_goal else None,
                ],
            )
        )
        default_result = Mock(
            name="default_policy_result", actions=[Mock()], status=PolicyStatus.READY
        )
        look_at_goal_result = Mock(
            name="look_at_goal_result",
            actions=[Mock(), Mock()],
            status=PolicyStatus.READY,
        )
        if undo:
            jump_result = Mock(
                name="jump_to_goal_result",
                actions=[Mock(), Mock()],
                status=PolicyStatus.READY,
            )
            expected_result = jump_result
        elif new_lm_goal:
            jump_result = Mock(
                name="jump_to_goal_result",
                actions=[Mock(), Mock()],
                status=PolicyStatus.IN_PROGRESS,
            )
            expected_result = jump_result
        elif new_sm_goal:
            jump_result = Mock(
                name="jump_to_goal_result", actions=[], status=PolicyStatus.READY
            )
            expected_result = look_at_goal_result
        else:
            jump_result = Mock(
                name="jump_to_goal_result", actions=[], status=PolicyStatus.READY
            )
            expected_result = default_result

        self.jump_to_goal.return_value = jump_result
        self.look_at_goal.return_value = look_at_goal_result
        self.default_policy.return_value = default_result

        result = self.selector(
            self.ctx,
            self.observations,
            self.state,
            self.percept,
            goals,
        )

        self.jump_to_goal.assert_called_with(
            self.ctx,
            self.observations,
            self.state,
            self.percept,
            lm_goal,
        )
        if lm_goal is not None:
            highest_confidence_goal_mock.assert_called_with(Goals([lm_goal]))
        self.assertIs(result, expected_result)


class Goals:  # noqa: PLW1641
    def __init__(self, goals: list[Goal]):
        self.goals = goals

    def __eq__(self, other: object) -> bool:
        if not hasattr(other, "__iter__"):
            return False
        return set(self.goals) == set(other)
