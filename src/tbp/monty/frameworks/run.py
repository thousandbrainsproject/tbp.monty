# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from tbp.monty.hydra import register_resolvers

logger = logging.getLogger(__name__)


def print_config(config: DictConfig) -> None:
    """Print config with nice formatting."""
    print("\n\n")
    print("Printing config below")
    print("-" * 100)
    print(OmegaConf.to_yaml(config))
    print("-" * 100)


# Global reference to experiment for signal handler
_experiment_instance = None


def _signal_handler(signum, frame):
    """Handle Ctrl-C gracefully."""
    print("\nRUN.PY: Received interrupt signal (Ctrl-C), shutting down...", file=sys.stderr)
    sys.stderr.flush()
    if _experiment_instance is not None:
        try:
            # Try to clean up the experiment if it exists
            if hasattr(_experiment_instance, '__exit__'):
                _experiment_instance.__exit__(None, None, None)
        except Exception:
            pass
    # Force exit - Hydra might be catching signals, so we need to be forceful
    os._exit(1)


@hydra.main(config_path="../../../conf", config_name="experiment", version_base=None)
def main(cfg: DictConfig):
    global _experiment_instance
    
    # Set up signal handler for graceful shutdown
    # Note: Hydra may interfere with signal handling, so we set it up here
    # and also try to handle it in the signal handler itself
    original_sigint = signal.signal(signal.SIGINT, _signal_handler)
    original_sigterm = signal.signal(signal.SIGTERM, _signal_handler)
    
    if cfg.quiet_habitat_logs:
        os.environ["MAGNUM_LOG"] = "quiet"
        os.environ["HABITAT_SIM_LOG"] = "quiet"
    
    # Try to force headless mode for habitat-sim to avoid OpenGL blocking
    # This may help if habitat-sim is hanging during initialization
    if "HABITAT_SIM_HEADLESS" not in os.environ:
        os.environ["HABITAT_SIM_HEADLESS"] = "1"
    
    # Disable wandb connection attempts to avoid blocking
    # Set wandb to offline mode so it doesn't try to connect during config resolution
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_DISABLED"] = "true"

    print_config(cfg)
    register_resolvers()

    output_dir = (
        Path(cfg.experiment.config.logging.output_dir)
        / cfg.experiment.config.logging.run_name
    )
    cfg.experiment.config.logging.output_dir = str(output_dir)

    output_dir.mkdir(exist_ok=True, parents=True)
    logger.info("Instantiating experiment...")
    print("RUN.PY: About to instantiate experiment...", file=__import__('sys').stderr)
    __import__('sys').stderr.flush()
    print(f"RUN.PY: Target class: {cfg.experiment.get('_target_', 'NOT SET')}", file=__import__('sys').stderr)
    __import__('sys').stderr.flush()
    try:
        print("RUN.PY: Calling hydra.utils.instantiate(cfg.experiment)...", file=__import__('sys').stderr)
        print("RUN.PY: This may take 30-60 seconds (habitat-sim initialization)...", file=__import__('sys').stderr)
        __import__('sys').stderr.flush()
        
        # Start a progress indicator in a separate thread
        progress_stop = threading.Event()
        progress_counter = [0]  # Use list for mutable counter
        
        def progress_indicator():
            while not progress_stop.is_set():
                time.sleep(2)
                if not progress_stop.is_set():
                    progress_counter[0] += 1
                    elapsed = progress_counter[0] * 2
                    print(f"RUN.PY: Still initializing... ({elapsed}s elapsed)", file=__import__('sys').stderr)
                    __import__('sys').stderr.flush()
        
        progress_thread = threading.Thread(target=progress_indicator, daemon=True)
        progress_thread.start()
        
        try:
            experiment = hydra.utils.instantiate(cfg.experiment)
        finally:
            progress_stop.set()
            elapsed = progress_counter[0] * 2
            print(f"RUN.PY: Initialization complete! (took {elapsed}s)", file=__import__('sys').stderr)
            __import__('sys').stderr.flush()
        
        _experiment_instance = experiment  # Store for signal handler
        logger.info(f"Experiment instantiated: {type(experiment)}")
        print(f"RUN.PY: Experiment instantiated successfully: {type(experiment)}", file=__import__('sys').stderr)
        __import__('sys').stderr.flush()
    except Exception as e:
        logger.error(f"Failed to instantiate experiment: {e}", exc_info=True)
        print(f"RUN.PY: ERROR during instantiation: {e}", file=__import__('sys').stderr)
        import traceback
        traceback.print_exc(file=__import__('sys').stderr)
        __import__('sys').stderr.flush()
        raise
    start_time = time.time()
    logger.info("Entering experiment context manager...")
    print("RUN.PY: Entering experiment context manager...", file=__import__('sys').stderr)
    __import__('sys').stderr.flush()
    with experiment:
        logger.info("Calling experiment.run()...")
        print("RUN.PY: Calling experiment.run()...", file=__import__('sys').stderr)
        __import__('sys').stderr.flush()
        experiment.run()

    logger.info(f"Done running {experiment} in {time.time() - start_time} seconds")
