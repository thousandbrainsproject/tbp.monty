"""The experiment a config names, with LiveView mixed in.

Hydra instantiates whatever `experiment._target_` names. Pointing that at
`MontyExperimentWithLiveView` — which is what `run.sh` used to do for every base
config — runs a *different* experiment whenever the config named a subclass of
`MontyExperiment`: `MontyObjectRecognitionExperiment` and
`MontySupervisedObjectPretrainingExperiment` each carry their own `pre_episode`,
`post_episode` and `run`, and replacing the class drops them silently.

`monty_experiment_with_liveview` is a target Hydra can call instead: it takes the
class the config named, mixes the LiveView behaviour into it, and constructs
that. The experiment stays the experiment; only the dashboard is added.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hydra.utils import get_class

from .monty_experiment_with_liveview import MontyExperimentWithLiveView

if TYPE_CHECKING:
    from tbp.monty.frameworks.experiments.monty_experiment import MontyExperiment


def liveview_experiment_class(base: type) -> type:
    """`base`, with the LiveView behaviour mixed in front of it.

    A class that already has the behaviour is returned as it is: mixing it in
    twice would put the state broadcasts in the chain twice.

    Order matters. `MontyExperimentWithLiveView` comes first so its `run`,
    `pre_episode` and `post_episode` are the ones called, and each of them calls
    `super()`, which is where `base`'s own versions run — the target object told
    to the model, the statistics a pretraining epoch writes, and so on.
    """
    if issubclass(base, MontyExperimentWithLiveView):
        return base

    return type(f"{base.__name__}WithLiveView", (MontyExperimentWithLiveView, base), {})


def monty_experiment_with_liveview(
    base_experiment_class: str, **experiment_config: Any
) -> MontyExperiment:
    """Build the config's own experiment, broadcasting to the dashboard.

    Hydra calls this where it would have called the class in `_target_`, with the
    rest of the `experiment` node as keyword arguments — `base_experiment_class`
    is the class it would have used, which `run.sh` reads out of the composed
    config and passes here.
    """
    experiment_class = liveview_experiment_class(get_class(base_experiment_class))
    return experiment_class(**experiment_config)
