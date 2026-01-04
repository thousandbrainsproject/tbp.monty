"""MontyExperiment with LiveView support via ZMQ.

The LiveView server is started separately (e.g., by run.sh) and listens to ZMQ.
This class only sets up the ZMQ broadcaster to send updates.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

try:
    import hydra
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False
    hydra = None  # type: ignore

from tbp.monty.frameworks.experiments.monty_experiment import (
    ExperimentMode,
    MontyExperiment,
)

from .zmq_broadcaster import ZmqBroadcaster

logger = logging.getLogger(__name__)


class MontyExperimentWithLiveView(MontyExperiment):
    """MontyExperiment extended with ZMQ-based LiveView for real-time monitoring.
    
    The LiveView server should be started separately (e.g., by run.sh).
    This class only sets up the ZMQ broadcaster to send updates.
    """

    def __init__(
        self, 
        config: Any, 
        liveview_port: Optional[int] = None, 
        liveview_host: Optional[str] = None,
        enable_liveview: Optional[bool] = None,
        zmq_port: Optional[int] = None
    ) -> None:
        """Initialize the experiment with LiveView support via ZMQ.
        
        The LiveView server should be started separately (e.g., by run.sh).
        This class only sets up the ZMQ broadcaster to send updates.

        Args:
            config: Experiment configuration (same as MontyExperiment).
                   May contain liveview_port, liveview_host, enable_liveview, zmq_port.
            liveview_port: Port for the LiveView server (for reference only). Defaults to 8000.
            liveview_host: Host for the LiveView server (for reference only). Defaults to "127.0.0.1".
            enable_liveview: Whether to enable LiveView broadcasting. Defaults to True.
            zmq_port: Port for ZMQ publisher. Defaults to 5555.
        """
        logger.info("MontyExperimentWithLiveView.__init__ called")
        
        # Extract settings from config if not provided as parameters
        if hasattr(config, 'get'):
            self.liveview_port = liveview_port if liveview_port is not None else config.get('liveview_port', 8000)
            self.liveview_host = liveview_host if liveview_host is not None else config.get('liveview_host', '127.0.0.1')
            self.enable_liveview = enable_liveview if enable_liveview is not None else config.get('enable_liveview', True)
            self.zmq_port = zmq_port if zmq_port is not None else config.get('zmq_port', 5555)
            # zmq_host defaults to liveview_host if not specified, but normalize to 127.0.0.1
            zmq_host_from_config = config.get('zmq_host', None)
            if zmq_host_from_config:
                self.zmq_host = zmq_host_from_config
            else:
                self.zmq_host = self.liveview_host
            # Normalize localhost to 127.0.0.1 for ZMQ
            if not self.zmq_host or self.zmq_host == 'localhost':
                self.zmq_host = '127.0.0.1'
        else:
            self.liveview_port = liveview_port if liveview_port is not None else 8000
            self.liveview_host = liveview_host if liveview_host is not None else '127.0.0.1'
            self.enable_liveview = enable_liveview if enable_liveview is not None else True
            self.zmq_port = zmq_port if zmq_port is not None else 5555
            self.zmq_host = '127.0.0.1'  # Default to localhost for ZMQ
        
        # Initialize parent class first
        logger.info("MontyExperimentWithLiveView: About to call super().__init__(config)")
        try:
            super().__init__(config)
            logger.info("MontyExperimentWithLiveView: super().__init__(config) completed")
        except Exception as e:
            logger.error(f"MontyExperimentWithLiveView: Error in super().__init__: {e}", exc_info=True)
            raise
        
        # Set up ZMQ broadcaster for cross-process communication
        # The LiveView server is started separately and listens to ZMQ
        # Do this after parent init to avoid blocking during instantiation
        if self.enable_liveview:
            try:
                self.broadcaster = ZmqBroadcaster(
                    zmq_port=self.zmq_port,
                    zmq_host=self.zmq_host
                )
                self.broadcaster.connect()
                logger.info(f"ZMQ broadcaster initialized on tcp://{self.zmq_host}:{self.zmq_port}")
                
                # Broadcast initial state immediately so LiveView isn't empty
                self._experiment_start_time = datetime.now(timezone.utc)
                self._experiment_status = "initializing"
                metadata = self._get_experiment_metadata()
                self.broadcaster.publish_state({
                    "run_name": getattr(self, 'run_name', 'Experiment'),
                    "experiment_name": metadata["experiment_name"],
                    "environment_name": metadata["environment_name"],
                    "config_path": metadata["config_path"],
                    "experiment_start_time": self._experiment_start_time.isoformat(),
                    "status": self._experiment_status,
                    "experiment_mode": "train",  # Default mode
                })
                logger.info("Initial state broadcasted to LiveView")
            except Exception as e:
                logger.warning(f"Failed to initialize ZMQ broadcaster: {e}. Continuing without LiveView updates.")
                self.broadcaster = None
        else:
            self.broadcaster = None
            logger.info("LiveView broadcasting disabled")
        
        # Track experiment state for broadcasting (if broadcaster wasn't set up)
        if not hasattr(self, '_experiment_start_time'):
            self._experiment_start_time = datetime.now(timezone.utc)
            self._experiment_status = "initializing"

    def _get_experiment_metadata(self) -> Dict[str, str]:
        """Get experiment metadata (environment, experiment name, config path)."""
        metadata = {
            "environment_name": os.environ.get("CONDA_DEFAULT_ENV", ""),
            "experiment_name": "",
            "config_path": "",
        }
        
        # Try to get experiment name from Hydra or config
        if HYDRA_AVAILABLE:
            try:
                from hydra.core.global_hydra import GlobalHydra
                hydra_instance = GlobalHydra.instance()
                if hydra_instance is not None:
                    cfg = hydra_instance.hydra
                    if cfg is not None:
                        # Try to get experiment name from Hydra's config
                        if hasattr(cfg, 'runtime') and hasattr(cfg.runtime, 'choices'):
                            # Hydra stores experiment choices in runtime.choices
                            if 'experiment' in cfg.runtime.choices:
                                metadata["experiment_name"] = cfg.runtime.choices['experiment']
            except Exception as e:
                logger.debug(f"Could not get experiment name from Hydra: {e}")
        
        # Fallback: try to get from config or run_name
        if not metadata["experiment_name"]:
            if hasattr(self, 'config') and hasattr(self.config, 'get'):
                # Check if experiment name is in the config
                exp_config = self.config.get("experiment", {})
                if isinstance(exp_config, dict) and "_name_" in exp_config:
                    metadata["experiment_name"] = exp_config["_name_"]
            if not metadata["experiment_name"]:
                # Use run_name as fallback
                metadata["experiment_name"] = getattr(self, 'run_name', '') or ""
        
        # Try to get config path from Hydra
        if HYDRA_AVAILABLE:
            try:
                from hydra.core.global_hydra import GlobalHydra
                hydra_instance = GlobalHydra.instance()
                if hydra_instance is not None:
                    cfg = hydra_instance.hydra
                    if cfg is not None:
                        # Get the config path from Hydra's runtime config_sources
                        if hasattr(cfg, 'runtime') and hasattr(cfg.runtime, 'config_sources'):
                            # Look for experiment config file in config_sources
                            for source in cfg.runtime.config_sources:
                                if hasattr(source, 'path'):
                                    path_str = str(source.path)
                                    # Look for experiment config files
                                    if 'experiment' in path_str and (path_str.endswith('.yaml') or path_str.endswith('.yml')):
                                        metadata["config_path"] = path_str
                                        break
                            # If not found, get the first config source that looks like a config file
                            if not metadata["config_path"]:
                                for source in cfg.runtime.config_sources:
                                    path_str = str(getattr(source, 'path', ''))
                                    if path_str and (path_str.endswith('.yaml') or path_str.endswith('.yml')):
                                        metadata["config_path"] = path_str
                                        break
            except Exception as e:
                logger.debug(f"Could not get config path from Hydra: {e}")
        
        return metadata

    def setup_experiment(self, config: Dict[str, Any]) -> None:
        """Set up the experiment and broadcast initial state."""
        super().setup_experiment(config)
        
        self._experiment_status = "running"
        
        # Broadcast initial state with configuration details
        if self.broadcaster:
            metadata = self._get_experiment_metadata()
            
            # Build setup message (formatted configuration summary)
            setup_lines = [
                f"Experiment: {metadata['experiment_name']}",
                f"Environment: {metadata['environment_name']}",
                f"Config: {metadata['config_path']}",
                f"Run: {self.run_name}",
                f"Training: {'Yes' if self.do_train else 'No'} (max steps: {self.max_train_steps}, epochs: {self.n_train_epochs})",
                f"Evaluation: {'Yes' if self.do_eval else 'No'} (max steps: {self.max_eval_steps}, epochs: {self.n_eval_epochs})",
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
                setup_lines.append(f"Show Sensor Output: {self.config.get('show_sensor_output')}")
            
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
                "show_sensor_output": self.config.get("show_sensor_output", False) if hasattr(self, "config") else False,
                "seed": self.config.get("seed") if hasattr(self, "config") else None,
                "model_name_or_path": self.config.get("model_name_or_path") if hasattr(self, "config") else None,
                "min_lms_match": self.config.get("min_lms_match") if hasattr(self, "config") else None,
                "setup_message": setup_message,  # Cache the formatted setup message
            }
            
            # Add model path if available
            if hasattr(self, "model_path") and self.model_path:
                state_update["model_path"] = str(self.model_path)
            
            self.broadcaster.publish_state(state_update)
            
            # Add model info after model is initialized
            if hasattr(self, "model") and self.model:
                self.broadcaster.publish_state({
                    "learning_module_count": len(self.model.learning_modules),
                    "sensor_module_count": len(self.model.sensor_modules),
                })

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
        state.update({
            "experiment_mode": mode.value,
            "current_epoch": epoch + 1,  # Convert from 0-based completed count to 1-based current
            "current_episode": episode + 1,  # Convert from 0-based completed count to 1-based current
        })
        
        self.broadcaster.publish_state(state)

    def pre_step(self, step: int, observation: Any) -> None:
        """Update state before each step."""
        super().pre_step(step, observation)
        if self.broadcaster:
            # Calculate cumulative step count: total from previous episodes + current step in this episode
            # This makes "Current Step" match the cumulative progress shown in "Total Steps"
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
        if hasattr(self, 'env_interface') and hasattr(self.env_interface, 'primary_target'):
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
                # Small delay to ensure final status message is sent before context cleanup
                time.sleep(0.05)
        except Exception as e:
            logger.error(f"Experiment failed: {e}", exc_info=True)
            self._experiment_status = "error"
            if self.broadcaster:
                self.broadcaster.publish_state({
                    "status": self._experiment_status,
                    "error_message": str(e)
                })
                # Small delay to ensure error status message is sent before context cleanup
                time.sleep(0.05)
            raise
        finally:
            self._update_state_from_experiment()

    def close(self) -> None:
        """Close the experiment and ZMQ broadcaster."""
        if self.broadcaster:
            self.broadcaster.close()
        super().close()
