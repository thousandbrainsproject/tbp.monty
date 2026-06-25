---
title: Code Base Structure
---

# Code Base Structure

The basic repository structure looks as follows:

```
.
|-- benchmarks/                 # CSVs with latest benchmark results
|-- docs/                       # Source files for documentation
|-- rfcs/                       # Merged RFCs
|-- src/tbp/monty/
|   |-- conf/                   # Monty configurations
|   |   |-- constants/          # Constants used in configurations
|   |   |-- env_interface/      # Configurations for environment interfaces
|   |   |-- environment/        # Configurations for environments
|   |   |-- experiment/         # Configurations for experiments
|   |   |-- logging/            # Configurations for logging
|   |   `-- monty/              # Configurations for Monty and its internals
|   |-- experiment/             # Intended future home of all experiment code
|   |-- frameworks/
|   |   |-- actions
|   |   |-- environment_utils
|   |   |-- environments        # Environments Monty can learn in
|   |   |-- experiments         # Current location of experiment classes
|   |   |-- loggers
|   |   |-- models              # LMs, SMs, motor system, & CMP
|   |   `-- utils
|   |-- simulators/
|   |   |-- habitat/            # Habitat simulator code
|   |   |   |-- actions.py
|   |   |   |-- actuator.py
|   |   |   |-- agents.py
|   |   |   |-- environment.py
|   |   |   |-- sensors.py
|   |   |   `-- simulator.py
|   |   `-- mujoco/             # MuJoCo simulator code
|   |-- cmp.py                  # Cortical Message Protocol code
|   |-- context.py              # Runtime context code
|   |-- hydra.py                # Hydra configuration code
|   |-- math.py                 # Math code
|   `-- path.py                 # Filesystem path code
|-- tests/
|-- tools/
`-- README.md
```

This is a slightly handpicked selection of folders and subfolders which tries to highlight to most important folders to get started.
The frameworks, simulators, and tests folders contain many files that are not listed here. The main code used for modeling can be found in `src/tbp/monty/frameworks/models/`.
