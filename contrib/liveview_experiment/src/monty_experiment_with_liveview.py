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

from .metadata_extractor import MetadataExtractor
from .zmq_broadcaster import ZmqBroadcaster

if TYPE_CHECKING:
    from .types import ConfigDict
else:
    from .types import ConfigDict  # noqa: TC001

from .broadcaster_initializer import BroadcasterInitializer
from .experiment_config_extractor import ExperimentConfigExtractor

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

        # Initialize broadcaster to None (will be set up later if enabled)
        self.broadcaster: ZmqBroadcaster | None = None

        # Extract and normalize configuration
        config_values = ExperimentConfigExtractor.extract_liveview_config(
            config, liveview_port, liveview_host, enable_liveview, zmq_port
        )
        self.liveview_port = config_values["liveview_port"]
        self.liveview_host = config_values["liveview_host"]
        self.enable_liveview = config_values["enable_liveview"]
        self.zmq_port = config_values["zmq_port"]
        self.zmq_host = config_values["zmq_host"]

        # Initialize parent class first
        logger.info(
            "MontyExperimentWithLiveView: About to call super().__init__(config)"
        )
        try:
            super().__init__(config)
            logger.info(
                "MontyExperimentWithLiveView: super().__init__(config) completed"
            )
        except Exception:
            logger.exception("MontyExperimentWithLiveView: Error in super().__init__")
            raise

        # Set up ZMQ broadcaster for cross-process communication
        # The LiveView server is started separately and listens to ZMQ
        # Do this after parent init to avoid blocking during instantiation
        if self.enable_liveview:
            self.broadcaster = self._setup_broadcaster()
        else:
            self.broadcaster = None
            logger.info("LiveView broadcasting disabled")

        # Track experiment state for broadcasting
        if not hasattr(self, "_experiment_start_time"):
            self._experiment_start_time = datetime.now(timezone.utc)
            self._experiment_status = "initializing"

    def _setup_broadcaster(self) -> ZmqBroadcaster | None:
        """Set up ZMQ broadcaster and broadcast initial state.

        Returns:
            Broadcaster instance or None if setup failed
        """
        try:
            broadcaster = ZmqBroadcaster(zmq_port=self.zmq_port, zmq_host=self.zmq_host)

            self._experiment_start_time = datetime.now(timezone.utc)
            self._experiment_status = "initializing"
            metadata = self._get_experiment_metadata()

            BroadcasterInitializer.initialize_and_broadcast(
                broadcaster,
                getattr(self, "run_name", "Experiment"),
                metadata,
            )

            return broadcaster
        except (OSError, RuntimeError) as e:
            logger.warning(
                "Failed to initialize ZMQ broadcaster: %s. "
                "Continuing without LiveView updates.",
                e,
            )
            return None

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

        # Broadcast initial state with configuration details
        if self.broadcaster:
            metadata = self._get_experiment_metadata()

            # Build setup message (formatted configuration summary)
            training_info = (
                f"Training: {'Yes' if self.do_train else 'No'} "
                f"(max steps: {self.max_train_steps}, epochs: {self.n_train_epochs})"
            )
            eval_info = (
                f"Evaluation: {'Yes' if self.do_eval else 'No'} "
                f"(max steps: {self.max_eval_steps}, epochs: {self.n_eval_epochs})"
            )
            setup_lines = [
                f"Experiment: {metadata['experiment_name']}",
                f"Environment: {metadata['environment_name']}",
                f"Config: {metadata['config_path']}",
                f"Run: {self.run_name}",
                training_info,
                eval_info,
            ]
            if self.config.get("seed") is not None:
                setup_lines.append(f"Seed: {self.config.get('seed')}")
            if self.config.get("model_name_or_path"):
                setup_lines.append(f"Model: {self.config.get('model_name_or_path')}")
            if hasattr(self, "model_path") and self.model_path:
                setup_lines.append(f"Model Path: {self.model_path}")
            if self.config.get("min_lms_match") is not None:
                setup_lines.append(f"Min LMs Match: {self.config.get('min_lms_match')}")
            if self.config.get("show_sensor_output") is not None:
                setup_lines.append(
                    f"Show Sensor Output: {self.config.get('show_sensor_output')}"
                )

            setup_message = "\n".join(setup_lines)

            state_update = {
                "run_name": self.run_name,
                "experiment_name": metadata["experiment_name"],
                "environment_name": metadata["environment_name"],
                "config_path": metadata["config_path"],
                "experiment_start_time": self._experiment_start_time.isoformat(),
                "status": self._experiment_status,
                "max_train_steps": self.max_train_steps,
                "max_eval_steps": self.max_eval_steps,
                "max_total_steps": self.max_total_steps,
                "n_train_epochs": self.n_train_epochs,
                "n_eval_epochs": self.n_eval_epochs,
                "do_train": self.do_train,
                "do_eval": self.do_eval,
                "show_sensor_output": (
                    self.config.get("show_sensor_output", False)
                    if hasattr(self, "config")
                    else False
                ),
                "seed": self.config.get("seed") if hasattr(self, "config") else None,
                "model_name_or_path": (
                    self.config.get("model_name_or_path")
                    if hasattr(self, "config")
                    else None
                ),
                "min_lms_match": (
                    self.config.get("min_lms_match")
                    if hasattr(self, "config")
                    else None
                ),
                "setup_message": setup_message,  # Cache the formatted setup message
            }

            # Add model path if available
            if hasattr(self, "model_path") and self.model_path:
                state_update["model_path"] = str(self.model_path)

            self.broadcaster.publish_state(state_update)

            # Add model info after model is initialized
            if hasattr(self, "model") and self.model:
                self.broadcaster.publish_state(
                    {
                        "learning_module_count": len(self.model.learning_modules),
                        "sensor_module_count": len(self.model.sensor_modules),
                    }
                )

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
