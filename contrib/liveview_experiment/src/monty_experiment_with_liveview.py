"""MontyExperiment with LiveView support via ZMQ.

The LiveView server is started separately (e.g., by run.sh) and listens to ZMQ.
This class only sets up the ZMQ broadcaster to send updates.
"""

from __future__ import annotations

import base64
import io
import logging
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image  # type: ignore[import-untyped]

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

from .command_subscriber import CommandSubscriber
from .zmq_broadcaster import ZmqBroadcaster  # noqa: TC001

logger = logging.getLogger(__name__)


class ExperimentAbortedError(Exception):
    """Exception raised when experiment is aborted via LiveView UI."""


class MontyExperimentWithLiveView(MontyExperiment):
    """MontyExperiment extended with ZMQ-based LiveView for real-time monitoring.

    The LiveView server should be started separately (e.g., by run.sh).
    This class only sets up the ZMQ broadcaster to send updates.
    """

    def __init__(
        self,
        config: ConfigDict,
        *,
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
        self.command_subscriber: CommandSubscriber | None = None
        self._abort_requested = False

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
        """Initialize ZMQ broadcaster and command subscriber if LiveView is enabled."""
        if self.enable_liveview:
            metadata = self._get_experiment_metadata()
            params = BroadcasterSetupParams(
                zmq_port=self.zmq_port,
                zmq_host=self.zmq_host,
                run_name=getattr(self, "run_name", "Experiment"),
                metadata=metadata.to_dict(),
            )
            self.broadcaster = ExperimentInitializer.setup_broadcaster(self, params)

            # Initialize command subscriber (port = zmq_port + 1 by convention)
            self._initialize_command_subscriber()
        else:
            logger.info("LiveView broadcasting disabled")

    def _initialize_command_subscriber(self) -> None:
        """Initialize the ZMQ command subscriber for receiving LiveView commands."""
        command_port = self.zmq_port + 1
        self.command_subscriber = CommandSubscriber(
            host=self.zmq_host, port=command_port
        )

        if self.command_subscriber.initialize():
            # Register abort handler
            self.command_subscriber.register_handler(
                "abort", self._handle_abort_command
            )
            self.command_subscriber.start()
            logger.info("Command subscriber started on port %d", command_port)
        else:
            logger.warning("Failed to initialize command subscriber")
            self.command_subscriber = None

    def _handle_abort_command(self, command: Any) -> None:
        """Handle abort command from LiveView.

        Args:
            command: The received abort command.
        """
        reason = command.payload.get("reason", "Unknown reason")
        logger.warning("Abort command received: %s", reason)
        self._abort_requested = True

    def _initialize_experiment_timing(self) -> None:
        """Initialize experiment start time and status if not already set."""
        if not hasattr(self, "_experiment_start_time"):
            self._experiment_start_time = datetime.now(timezone.utc)
            self._experiment_status = "initializing"
        # Throttle image broadcasts (expensive to encode/transmit)
        self._last_image_broadcast_time = 0.0
        self._sensor_image_throttle_ms = getattr(
            self.config, "sensor_image_throttle_ms", 100
        )

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
        # Check for abort request before processing step
        self._check_abort_request()

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

    def _check_abort_request(self) -> None:
        """Check if abort has been requested and raise exception if so."""
        # Check both local flag and subscriber
        if self._abort_requested:
            raise ExperimentAbortedError("Experiment aborted via LiveView UI")

        if self.command_subscriber and self.command_subscriber.is_abort_requested():
            self._abort_requested = True
            raise ExperimentAbortedError("Experiment aborted via LiveView UI")

    def post_step(self, step: int, observation: Any) -> None:
        """Update state after each step."""
        super().post_step(step, observation)
        self._update_state_from_experiment()
        self._broadcast_evidence_data(step)
        self._broadcast_sensor_images(step, observation)

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

    def _broadcast_sensor_images(self, step: int, observation: Any) -> None:
        """Broadcast sensor images for live visualization.

        Captures camera image and depth image from sensor observations,
        encodes them as base64 PNG, and broadcasts via ZMQ.

        Images are throttled to avoid overwhelming the network/browser.

        Args:
            step: Current step number within the episode.
            observation: The observation from the environment.
        """
        if not self.broadcaster or not hasattr(self, "model"):
            return

        # Throttle image broadcasts
        current_time = time.time() * 1000  # Convert to ms
        if (
            current_time - self._last_image_broadcast_time
            < self._sensor_image_throttle_ms
        ):
            return
        self._last_image_broadcast_time = current_time

        try:
            images = self._extract_sensor_images(observation, step)
            if images:
                self.broadcaster.publish_data("sensor_images", images)
        except (KeyError, AttributeError, TypeError, ValueError) as e:
            # Don't let image extraction failures crash the experiment
            logger.debug("Failed to extract sensor images: %s", e)

    def _extract_sensor_images(
        self, observation: Any, step: int
    ) -> dict[str, Any] | None:
        """Extract sensor images from observation and encode as base64.

        Follows the same pattern as LivePlotter.hardcoded_assumptions().

        Args:
            observation: The observation from the environment.
            step: Current step number within the episode.

        Returns:
            Dictionary with base64-encoded images, or None if extraction fails.
        """
        if not hasattr(self, "model") or not self.model:
            return None

        try:
            agent_id = self._get_agent_id()
            if agent_id is None:
                return None

            agent_observation = observation.get(agent_id, {})

            camera_b64 = self._extract_camera_image_b64(agent_observation)
            depth_b64 = self._extract_depth_image_b64(agent_observation)

            if not (camera_b64 or depth_b64):
                return None

            return self._build_sensor_image_payload(camera_b64, depth_b64, step)
        except (KeyError, AttributeError, TypeError, IndexError) as e:
            logger.debug("Error extracting sensor images: %s", e)
            return None

    def _get_agent_id(self) -> str | None:
        """Safely get the agent id from the model."""
        motor_system = getattr(self.model, "motor_system", None)
        if motor_system is None or not hasattr(motor_system, "_policy"):
            return None
        agent_id = getattr(motor_system._policy, "agent_id", None)
        return str(agent_id) if agent_id is not None else None

    def _extract_camera_image_b64(self, agent_observation: Any) -> str | None:
        """Extract base64-encoded camera image from agent observation."""
        view_finder = agent_observation.get("view_finder")
        if not isinstance(view_finder, dict):
            return None

        rgba = view_finder.get("rgba")
        if rgba is None:
            return None

        return self._numpy_to_base64_png(rgba)

    def _extract_depth_image_b64(self, agent_observation: Any) -> str | None:
        """Extract base64-encoded depth image from agent observation."""
        if not getattr(self.model, "sensor_modules", None):
            return None

        first_module = self.model.sensor_modules[0]
        first_sm_id = getattr(first_module, "sensor_module_id", None)
        if first_sm_id is None:
            return None

        sm_obs = agent_observation.get(first_sm_id)
        if not isinstance(sm_obs, dict):
            return None

        depth = sm_obs.get("depth")
        if depth is None:
            return None

        return self._depth_to_base64_png(depth)

    def _build_sensor_image_payload(
        self, camera_b64: str | None, depth_b64: str | None, step: int
    ) -> dict[str, Any]:
        """Build payload dictionary for sensor images."""
        return {
            "camera_image": camera_b64,
            "depth_image": depth_b64,
            "step": step,
            "timestamp": time.time(),
        }

    def _numpy_to_base64_png(self, img_array: np.ndarray) -> str | None:
        """Convert numpy RGBA array to base64-encoded PNG.

        Args:
            img_array: RGBA image as numpy array (H, W, 4) or RGB (H, W, 3).

        Returns:
            Base64-encoded PNG string, or None on failure.
        """
        try:
            normalized = self._normalize_image_array(img_array)
            img = self._create_pil_image_from_array(normalized)
            return self._encode_image_to_base64(img)
        except (ValueError, TypeError, OSError) as e:
            logger.debug("Failed to encode image: %s", e)
            return None

    def _normalize_image_array(self, img_array: np.ndarray) -> np.ndarray:
        """Normalize an image array to uint8 RGB(A) format."""
        if img_array.dtype != np.uint8:
            max_value = float(img_array.max())
            if max_value <= 1.0:
                img_array = (img_array * 255).astype(np.uint8)
            else:
                img_array = img_array.astype(np.uint8)
        return img_array

    def _create_pil_image_from_array(self, img_array: np.ndarray) -> Image.Image:
        """Create a PIL Image from a numpy array."""
        if img_array.shape[-1] == 4:
            return Image.fromarray(img_array, mode="RGBA")
        return Image.fromarray(img_array, mode="RGB")

    def _encode_image_to_base64(self, img: Image.Image) -> str:
        """Encode a PIL Image as a base64 PNG string."""
        buffer = io.BytesIO()
        img.save(buffer, format="PNG", optimize=True)
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("utf-8")

    def _depth_to_base64_png(self, depth_array: np.ndarray) -> str | None:
        """Convert depth array to base64-encoded PNG with colormap.

        Args:
            depth_array: Depth values as numpy array (H, W).

        Returns:
            Base64-encoded PNG string, or None on failure.
        """
        try:
            # Normalize depth to 0-255
            depth = np.array(depth_array)
            if depth.size == 0:
                return None

            # Handle NaN/Inf values
            depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

            # Normalize to 0-1, then to 0-255
            d_min, d_max = depth.min(), depth.max()
            if d_max > d_min:
                depth_norm = (depth - d_min) / (d_max - d_min)
            else:
                depth_norm = np.zeros_like(depth)

            # Apply viridis-like colormap (simplified)
            # For proper viridis, would use matplotlib, but keeping it simple
            depth_uint8 = (depth_norm * 255).astype(np.uint8)

            # Create grayscale image (inverted so closer = brighter)
            img = Image.fromarray(255 - depth_uint8, mode="L")

            # Encode to PNG
            buffer = io.BytesIO()
            img.save(buffer, format="PNG", optimize=True)
            buffer.seek(0)

            return base64.b64encode(buffer.read()).decode("utf-8")
        except (ValueError, TypeError, OSError) as e:
            logger.debug("Failed to encode depth image: %s", e)
            return None

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
        except ExperimentAbortedError as e:
            logger.warning("Experiment aborted: %s", e)
            self._experiment_status = "aborted"
            if self.broadcaster:
                self.broadcaster.publish_state(
                    {"status": self._experiment_status, "error_message": str(e)}
                )
                time.sleep(0.05)
            # Don't re-raise abort - it's a controlled stop
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
        """Close the experiment, ZMQ broadcaster, and command subscriber."""
        if self.command_subscriber:
            self.command_subscriber.close()
        if self.broadcaster:
            self.broadcaster.close()
        super().close()
