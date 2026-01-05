"""MontyExperiment with LiveView support via ZMQ.

The LiveView server is started separately (e.g., by run.sh) and listens to ZMQ.
This class only sets up the ZMQ broadcaster to send updates.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

try:
    from hydra.core.global_hydra import GlobalHydra
except ImportError:
    GlobalHydra = None  # type: ignore[assignment, unused-ignore, misc]


try:
    import hydra

    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False
    hydra = None  # type: ignore[assignment, unused-ignore]

from tbp.monty.frameworks.experiments.monty_experiment import (
    ExperimentMode,
    MontyExperiment,
)

from .experiment_initializer import ExperimentInitializer
from .metadata_extractor import MetadataExtractor
from .state_update_builder import StateUpdateBuilder

if TYPE_CHECKING:
    from .types import ConfigDict
    from .zmq_broadcaster import ZmqBroadcaster
else:
    from .types import ConfigDict  # noqa: TC001


logger = logging.getLogger(__name__)


class MontyExperimentWithLiveView(MontyExperiment):
    """MontyExperiment extended with ZMQ-based LiveView for real-time monitoring.

    The LiveView server should be started separately (e.g., by run.sh).
    This class only sets up the ZMQ broadcaster to send updates.
    """

    def __init__(
        self,
        config: ConfigDict,
        liveview_port: int | None = None,
        liveview_host: str | None = None,
        enable_liveview: bool | None = None,
        zmq_port: int | None = None,
    ) -> None:
        """Initialize the experiment with LiveView support via ZMQ.

        The LiveView server should be started separately (e.g., by run.sh).
        This class only sets up the ZMQ broadcaster to send updates.

        Args:
            config: Experiment configuration (same as MontyExperiment).
                   May contain liveview_port, liveview_host,
                   enable_liveview, zmq_port.
            liveview_port: Port for the LiveView server
                (for reference only). Defaults to 8000.
            liveview_host: Host for the LiveView server
                (for reference only). Defaults to "127.0.0.1".
            enable_liveview: Whether to enable LiveView broadcasting.
                Defaults to True.
            zmq_port: Port for ZMQ publisher. Defaults to 5555.
        """
        logger.info("MontyExperimentWithLiveView.__init__ called")

        self.broadcaster: ZmqBroadcaster | None = None

        config_values = ExperimentInitializer.extract_config(
            config, liveview_port, liveview_host, enable_liveview, zmq_port
        )
        self.liveview_port = config_values["liveview_port"]
        self.liveview_host = config_values["liveview_host"]
        self.enable_liveview = config_values["enable_liveview"]
        self.zmq_port = config_values["zmq_port"]
        self.zmq_host = config_values["zmq_host"]

        super().__init__(config)

        if self.enable_liveview:
            metadata = self._get_experiment_metadata()
            self.broadcaster = ExperimentInitializer.setup_broadcaster(
                self,
                self.zmq_port,
                self.zmq_host,
                getattr(self, "run_name", "Experiment"),
                metadata,
            )
        else:
            logger.info("LiveView broadcasting disabled")

        if not hasattr(self, "_experiment_start_time"):
            self._experiment_start_time = datetime.now(timezone.utc)
            self._experiment_status = "initializing"

    def _get_experiment_metadata(self) -> dict[str, str]:
        """Get experiment metadata (environment, experiment name, config path).

        Returns:
            Dictionary with metadata fields
        """
        extractor = MetadataExtractor(
            config=getattr(self, "config", None),
            run_name=getattr(self, "run_name", None),
        )
        metadata = extractor.extract()
        return metadata.to_dict()

    def setup_experiment(self, config: ConfigDict) -> None:
        """Set up the experiment and broadcast initial state."""
        super().setup_experiment(config)

        self._experiment_status = "running"

        if not self.broadcaster:
            return

        metadata = self._get_experiment_metadata()
        model_path = getattr(self, "model_path", None)

        setup_message = StateUpdateBuilder.build_setup_message(
            metadata,
            self.run_name,
            self.do_train,
            self.max_train_steps,
            self.n_train_epochs,
            self.do_eval,
            self.max_eval_steps,
            self.n_eval_epochs,
            self.config,
            model_path,
        )

        state_update = StateUpdateBuilder.build_initial_state_update(
            self.run_name,
            metadata,
            self._experiment_start_time,
            self._experiment_status,
            self.max_train_steps,
            self.max_eval_steps,
            self.max_total_steps,
            self.n_train_epochs,
            self.n_eval_epochs,
            self.do_train,
            self.do_eval,
            self.config,
            setup_message,
            model_path,
        )

        self.broadcaster.publish_state(state_update)

        if hasattr(self, "model") and self.model:
            model_info = StateUpdateBuilder.build_model_info_update(
                len(self.model.learning_modules),
                len(self.model.sensor_modules),
            )
            self.broadcaster.publish_state(model_info)

    def _update_state_from_experiment(self) -> None:
        """Update and broadcast experiment state."""
        if not self.broadcaster:
            return

        state = {
            "total_train_steps": self.total_train_steps,
            "train_episodes": self.train_episodes,
            "train_epochs": self.train_epochs,
            "total_eval_steps": self.total_eval_steps,
            "eval_episodes": self.eval_episodes,
            "eval_epochs": self.eval_epochs,
            "last_update": datetime.now(timezone.utc).isoformat(),
        }

        mode, epoch, episode = self.get_epoch_state()
        # The epoch/episode counters represent COMPLETED counts (0-based)
        # For display, we want to show the CURRENT epoch/episode (1-based)
        # So we add 1 to show what's currently running
        state.update(
            {
                "experiment_mode": mode.value,
                "current_epoch": epoch
                + 1,  # Convert from 0-based completed count to 1-based current
                "current_episode": episode
                + 1,  # Convert from 0-based completed count to 1-based current
            }
        )

        self.broadcaster.publish_state(state)

    def pre_step(self, step: int, observation: Any) -> None:
        """Update state before each step."""
        super().pre_step(step, observation)
        if self.broadcaster:
            # Calculate cumulative step count: total from previous episodes +
            # current step in this episode. This makes "Current Step" match
            # the cumulative progress shown in "Total Steps"
            if self.experiment_mode == ExperimentMode.TRAIN:
                cumulative_step = self.total_train_steps + step
            else:
                cumulative_step = self.total_eval_steps + step
            self.broadcaster.publish_state({"current_step": cumulative_step})
        self._update_state_from_experiment()

    def post_step(self, step: int, observation: Any) -> None:
        """Update state after each step."""
        super().post_step(step, observation)
        self._update_state_from_experiment()

    def pre_episode(self) -> None:
        """Update state before each episode.

        Override to pass primary_target to model.pre_episode() which requires it.
        """
        # Call model.pre_episode with primary_target (required by model)
        if hasattr(self, "env_interface") and hasattr(
            self.env_interface, "primary_target"
        ):
            self.model.pre_episode(self.env_interface.primary_target)
        else:
            # Fallback if primary_target not available
            self.model.pre_episode()

        # Call env_interface.pre_episode()
        self.env_interface.pre_episode()

        # Set max_steps based on mode
        self.max_steps = self.max_train_steps
        if self.experiment_mode != ExperimentMode.TRAIN:
            self.max_steps = self.max_eval_steps

        # Call logger handler
        self.logger_handler.pre_episode(self.logger_args)

        # Update LiveView state
        self._update_state_from_experiment()

    def post_episode(self, steps: int) -> None:
        """Update state after each episode."""
        super().post_episode(steps)
        self._update_state_from_experiment()

    def pre_epoch(self) -> None:
        """Update state before each epoch."""
        super().pre_epoch()
        self._update_state_from_experiment()

    def post_epoch(self) -> None:
        """Update state after each epoch."""
        super().post_epoch()
        self._update_state_from_experiment()

    def run(self) -> None:
        """Run the experiment with LiveView broadcasting."""
        try:
            self._experiment_status = "running"
            if self.broadcaster:
                self.broadcaster.publish_state({"status": self._experiment_status})

            super().run()

            self._experiment_status = "completed"
            if self.broadcaster:
                self.broadcaster.publish_state({"status": self._experiment_status})
                # Small delay to ensure final status message is sent
                # before context cleanup
                time.sleep(0.05)
        except Exception as e:
            logger.exception("Experiment failed")
            self._experiment_status = "error"
            if self.broadcaster:
                self.broadcaster.publish_state(
                    {"status": self._experiment_status, "error_message": str(e)}
                )
                # Small delay to ensure error status message is sent
                # before context cleanup
                time.sleep(0.05)
            raise
        finally:
            self._update_state_from_experiment()

    def close(self) -> None:
        """Close the experiment and ZMQ broadcaster."""
        if self.broadcaster:
            self.broadcaster.close()
        super().close()
