---
title: Writing a Custom Motor Policy
---

A motor policy chooses the actions an agent should take at each motor step. In this tutorial you will write a policy that sweeps a camera left and right, exercise it through Monty's `MotorSystem`, and select it in a Hydra experiment configuration. The first two sections use the Monty Python environment from [Getting Started](../getting-started.md) and work without a simulator or a dataset.

## Implement the policy

The sweep starts at the agent's initial heading. Each action is a relative rotation, measured in degrees. One cycle spans three calls to the policy.

| Call | Action | Angle | Heading relative to the start |
| --- | --- | --- | --- |
| 1 | `TurnLeft` | 5° | 5° left |
| 2 | `TurnRight` | 10° | 5° right |
| 3 | `TurnLeft` | 5° | Initial heading |

Create `sweep_policy.py` in the repository root next to `run.py`.

```python
# sweep_policy.py
from tbp.monty.cmp import Goal, Message
from tbp.monty.context import RuntimeContext
from tbp.monty.experiment.motor_system import ExperimentMotorSystem
from tbp.monty.frameworks.actions.actions import TurnLeft, TurnRight
from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.models.abstract_monty_classes import Observations
from tbp.monty.frameworks.models.motor_policies import MotorPolicy, MotorPolicyResult
from tbp.monty.frameworks.models.motor_system_state import MotorSystemState
from tbp.monty.memento import Memento


class SweepPolicy(MotorPolicy):
    """Rotate a distant agent across a fixed range of headings."""

    def __init__(self, agent_id: AgentID, sweep_degrees: float = 5.0) -> None:
        self.agent_id = agent_id
        self.sweep_degrees = sweep_degrees
        self.reset()

    def fixme_provide_motor_system(self, motor_system: ExperimentMotorSystem) -> None:
        # This policy keeps its state on the policy instance.
        pass

    def reset(self) -> None:
        self._next_turn = 0

    def state_dict(self) -> Memento:
        return {"next_turn": self._next_turn}

    def load_state_dict(self, memento: Memento) -> None:
        self._next_turn = memento["next_turn"]

    def __call__(
        self,
        ctx: RuntimeContext,
        observations: Observations,
        state: MotorSystemState,
        percept: Message,
        goal: Goal | None,
    ) -> MotorPolicyResult:
        action_type, turn_scale = (
            (TurnLeft, 1),
            (TurnRight, 2),
            (TurnLeft, 1),
        )[self._next_turn]
        self._next_turn = (self._next_turn + 1) % 3
        return MotorPolicyResult(
            actions=[action_type(self.agent_id, self.sweep_degrees * turn_scale)]
        )
```

The policy keeps track of the next turn. The motor system resets the policy between episodes, so the first turn is always left from the new starting heading. The policy's snapshot methods let you save and restore the next turn. The agent ID and sweep angle are constructor settings, so restore a snapshot into a policy configured with the same values.

The return value contains an action for the environment to execute. `TurnLeft` and `TurnRight` use positive angle magnitudes, and `agent_id` tells the environment which agent should move. Choose actions the target agent implements. A distant MuJoCo agent supports these yaw actions.

This sweep is open-loop. It bases each turn on its own position in the sequence. Other policies can use the RNG in `ctx`, sensor observations, the agent poses in `state`, the sensor module's `percept`, or a learning module's `goal`. `goal` may be `None`. See [Policy](../../how-monty-works/motor-system/policy.md) for examples that use these inputs.

## Test the action sequence

Create `try_sweep.py` next to `sweep_policy.py`. This script provides empty observations and a placeholder percept because the sweep does not use them. It calls the motor system with a `SinglePolicySelector`, the selector we will also use in the experiment.

```python
# try_sweep.py
import numpy as np

from sweep_policy import SweepPolicy
from tbp.monty.cmp import Message
from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.actions.actions import TurnLeft, TurnRight
from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.models.abstract_monty_classes import Observations
from tbp.monty.frameworks.models.motor_policy_selectors import SinglePolicySelector
from tbp.monty.frameworks.models.motor_system import MotorSystem
from tbp.monty.frameworks.models.motor_system_state import ProprioceptiveState


def main() -> None:
    policy = SweepPolicy(agent_id=AgentID("agent_id_0"), sweep_degrees=5.0)
    motor_system = MotorSystem(SinglePolicySelector(policy))
    ctx = RuntimeContext(rng=np.random.RandomState(42))
    percept = Message(
        location=None,
        morphological_features={},
        non_morphological_features={},
        confidence=0.0,
        pass_message=False,
        sender_id="patch",
        sender_type="SM",
        process_features_in_lm=False,
    )

    def print_next_turn() -> None:
        (action,) = motor_system(ctx, Observations(), ProprioceptiveState(), percept, [])
        assert isinstance(action, (TurnLeft, TurnRight))
        print(f"{type(action).__name__}: {action.rotation_degrees:g} degrees")

    print_next_turn()
    snapshot = policy.state_dict()
    print_next_turn()
    print_next_turn()
    print_next_turn()

    print("After restore:")
    policy.load_state_dict(snapshot)
    print_next_turn()

    print("After reset:")
    motor_system.reset()
    print_next_turn()


if __name__ == "__main__":
    main()
```

Run the script from the repository root.

```sh
uv run python try_sweep.py
```

The fourth call begins the next cycle. Restoring the snapshot taken after the first call resumes with the second action. Resetting starts a fresh cycle.

```text
TurnLeft: 5 degrees
TurnRight: 10 degrees
TurnLeft: 5 degrees
TurnLeft: 5 degrees
After restore:
TurnRight: 10 degrees
After reset:
TurnLeft: 5 degrees
```

The script checks the decisions made by the motor system. In a real experiment the environment receives these actions and moves the agent, and the next call receives observations from its new position.

## Select the policy in an experiment

You can now run the policy in the MuJoCo environment from [Running Your First Experiment](running-your-first-experiment.md). Install the `simulator_mujoco` extra from [Getting Started](../getting-started.md) for this section. The command uses MuJoCo's built-in `cubeSolid`, so it works without downloading the YCB dataset.

The experiment's `SinglePolicySelector` already passes control to one policy. Change the policy target to `sweep_policy.SweepPolicy`, remove the built-in policy's `action_sampler` argument, and add `sweep_degrees`. The existing configuration supplies `agent_id_0`.

```sh
uv run python run.py experiment=tutorial/first_experiment \
  experiment.config.monty_config.motor_system_config.policy_selector.policy._target_=sweep_policy.SweepPolicy \
  '~experiment.config.monty_config.motor_system_config.policy_selector.policy.action_sampler' \
  '+experiment.config.monty_config.motor_system_config.policy_selector.policy.sweep_degrees=5.0' \
  experiment.config.environment.env_init_args.data_path=null \
  'experiment.config.train_env_interface_args.object_names=[cubeSolid]' \
  experiment.config.logging.run_name=first_experiment_sweep
```

Hydra imports the policy from `sweep_policy.py`. The `~` override removes a parameter from the original policy, and the `+` override adds the new sweep angle. The settings for `data_path` and `object_names` select the built-in cube. Look for `sweep_policy.SweepPolicy` in the printed configuration to confirm which policy the experiment is using. The run saves results under `first_experiment_sweep` in your Monty models directory.

The first experiment is a short smoke test. To see the entire action cycle independently of the experiment's termination settings, run `try_sweep.py`. To reuse this policy in a longer experiment, use a distant agent that supports the same turns and make sure its configured ID matches the policy's `agent_id`.
