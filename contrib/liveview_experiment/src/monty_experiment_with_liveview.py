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

from .experiment_config import (
    BroadcasterSetupParams,
    CoreStateFields,
    EvaluationConfig,
    ExperimentLimits,
    LiveViewConfigOverrides,
    SetupMessageParams,
    TrainingConfig,
)
from .experiment_initializer import ExperimentInitializer
from .metadata_extractor import ExperimentMetadata, MetadataExtractor
from .state_update_builder import StateUpdateBuilder

if TYPE_CHECKING:
    from .types import ConfigDict

from .zmq_broadcaster import ZmqBroadcaster  # noqa: TC001

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

        overrides = LiveViewConfigOverrides(
            liveview_port=liveview_port,
            liveview_host=liveview_host,
            enable_liveview=enable_liveview,
            zmq_port=zmq_port,
        )
        self._extract_and_set_config(config, overrides)
        super().__init__(config)

        self._initialize_broadcaster_if_enabled()
        self._initialize_experiment_timing()

    def _extract_and_set_config(
        self,
        config: ConfigDict,
        overrides: LiveViewConfigOverrides,
    ) -> None:
        """Extract and set LiveView configuration values.

        Args:
            config: Experiment configuration
            overrides: LiveView configuration overrides
        """
        config_values = ExperimentInitializer.extract_config(config, overrides)
        self.liveview_port = config_values["liveview_port"]
        self.liveview_host = config_values["liveview_host"]
        self.enable_liveview = config_values["enable_liveview"]
        self.zmq_port = config_values["zmq_port"]
        self.zmq_host = config_values["zmq_host"]

    def _initialize_broadcaster_if_enabled(self) -> None:
        """Initialize ZMQ broadcaster if LiveView is enabled."""
        if self.enable_liveview:
            metadata = self._get_experiment_metadata()
            params = BroadcasterSetupParams(
                zmq_port=self.zmq_port,
                zmq_host=self.zmq_host,
                run_name=getattr(self, "run_name", "Experiment"),
                metadata=metadata.to_dict(),
            )
            self.broadcaster = ExperimentInitializer.setup_broadcaster(self, params)
        else:
            logger.info("LiveView broadcasting disabled")

    def _initialize_experiment_timing(self) -> None:
        """Initialize experiment start time and status if not already set."""
        if not hasattr(self, "_experiment_start_time"):
            self._experiment_start_time = datetime.now(timezone.utc)
            self._experiment_status = "initializing"

    def _get_experiment_metadata(self) -> ExperimentMetadata:
        """Get experiment metadata (environment, experiment name, config path).

        Returns:
            Experiment metadata object
        """
        extractor = MetadataExtractor(
            config=getattr(self, "config", None),
            run_name=getattr(self, "run_name", None),
        )
        return extractor.extract()

    def setup_experiment(self, config: ConfigDict) -> None:
        """Set up the experiment and broadcast initial state."""
        super().setup_experiment(config)

        self._experiment_status = "running"

        if not self.broadcaster:
            return

        self._broadcast_initial_state()
        self._broadcast_model_info_if_available()

    def _broadcast_initial_state(self) -> None:
        """Broadcast initial experiment state."""
        metadata = self._get_experiment_metadata()
        model_path = getattr(self, "model_path", None)

        training = self._create_training_config()
        evaluation = self._create_evaluation_config()
        limits = self._create_experiment_limits()

        metadata_dict = metadata.to_dict()
        setup_params = SetupMessageParams(
            metadata=metadata_dict,
            run_name=self.run_name,
            training=training,
            evaluation=evaluation,
            config=self.config,
            model_path=model_path,
        )
        setup_message = StateUpdateBuilder.build_setup_message_from_params(setup_params)
        core_fields = CoreStateFields(
            run_name=self.run_name,
            metadata=metadata_dict,
            experiment_start_time=self._experiment_start_time,
            status=self._experiment_status,
            limits=limits,
            training=training,
            evaluation=evaluation,
            setup_message=setup_message,
        )

        state_update = StateUpdateBuilder.build_initial_state_update(
            core_fields,
            self.config,
            model_path,
        )

        if self.broadcaster:
            self.broadcaster.publish_state(state_update)

    def _create_training_config(self) -> TrainingConfig:
        """Create training configuration from experiment attributes."""
        return TrainingConfig(
            do_train=self.do_train,
            max_train_steps=self.max_train_steps,
            n_train_epochs=self.n_train_epochs,
        )

    def _create_evaluation_config(self) -> EvaluationConfig:
        """Create evaluation configuration from experiment attributes."""
        return EvaluationConfig(
            do_eval=self.do_eval,
            max_eval_steps=self.max_eval_steps,
            n_eval_epochs=self.n_eval_epochs,
        )

    def _create_experiment_limits(self) -> ExperimentLimits:
        """Create experiment limits from experiment attributes."""
        return ExperimentLimits(
            max_train_steps=self.max_train_steps,
            max_eval_steps=self.max_eval_steps,
            max_total_steps=self.max_total_steps,
            n_train_epochs=self.n_train_epochs,
            n_eval_epochs=self.n_eval_epochs,
        )

    def _broadcast_model_info_if_available(self) -> None:
        """Broadcast model information if model is available."""
        if hasattr(self, "model") and self.model and self.broadcaster:
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
        self._broadcast_evidence_data(step)

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

    def _broadcast_evidence_data(self, step: int) -> None:
        """Broadcast evidence data for live visualization charts.

        Extracts evidence scores from learning modules and publishes them
        via the ZMQ telemetry stream for the LiveView chart.

        Args:
            step: Current step number within the episode.
        """
        if not self.broadcaster or not hasattr(self, "model"):
            return

        # Calculate cumulative step for consistent x-axis
        if self.experiment_mode == ExperimentMode.TRAIN:
            cumulative_step = self.total_train_steps + step
            current_episode = self.train_episodes + 1
        else:
            cumulative_step = self.total_eval_steps + step
            current_episode = self.eval_episodes + 1

        # Get current target object
        target_object = self._get_current_target_object()

        # Extract evidence from each learning module
        for lm_idx, lm in enumerate(self.model.learning_modules):
            evidences = self._extract_lm_evidence(lm)
            if not evidences:
                continue

            # Publish evidence chart data
            self.broadcaster.publish_data(
                "evidence_chart",
                {
                    "step": cumulative_step,
                    "evidences": evidences,
                    "target_object": target_object,
                    "episode": current_episode,
                    "lm_id": lm_idx,
                    "timestamp": time.time(),
                },
            )

    def _get_current_target_object(self) -> str:
        """Get the current target object name.

        Returns:
            Target object name or empty string if not available.
        """
        if hasattr(self, "env_interface") and hasattr(
            self.env_interface, "primary_target"
        ):
            target = self.env_interface.primary_target
            if target:
                return str(target)
        return ""

    def _extract_lm_evidence(self, lm: Any) -> dict[str, float]:
        """Extract evidence scores from a learning module.

        Uses the learning module's get_evidence_for_each_graph() method
        if available, which returns max evidence per known object.

        Args:
            lm: A learning module instance.

        Returns:
            Dictionary mapping object names to their max evidence scores.
        """
        if not hasattr(lm, "get_evidence_for_each_graph"):
            return {}

        # Check if LM has any known objects (during training it may be empty)
        if hasattr(lm, "get_all_known_object_ids"):
            known_ids = lm.get_all_known_object_ids()
            if not known_ids:
                return {}

        try:
            graph_ids, evidences = lm.get_evidence_for_each_graph()
            # Filter out placeholder values or empty results
            if not graph_ids or graph_ids[0] == "patch_off_object":
                return {}
            return dict(zip(graph_ids, [float(e) for e in evidences]))
        except (AttributeError, TypeError, ValueError, KeyError, IndexError):
            return {}

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
