---
title: Code Base Structure
---

# Code Base Structure

The basic repository structure looks as follows:

```
.
|-- benchmarks/                 # CSVs with latest benchmark results
|-- conf/                       # Monty configurations
|   |-- benchmarks/             # Some shared benchmark constants
|   |-- experiment/             # Configurations used for experiment
|   `-- test/                   # Configurations used in tests
|-- docs/                       # Source files for documentation
|-- rfcs/                       # Merged RFCs
|-- src/tbp/monty/
|   |-- frameworks/
|   |   |-- actions
|   |   |-- config_utils
|   |   |-- environment_utils
|   |   |-- environments        # Environments Monty can learn in
|   |   |-- experiments         # Experiment classes
|   |   |-- loggers
|   |   |-- models              # LMs, SMs, motor system, & CMP
|   |   `-- utils
|   `-- simulators/
|       `-- habitat/
|           |-- actions.py
|           |-- actuator.py
|           |-- agents.py
|           |-- environment.py
|           |-- sensors.py
|           `-- simulator.py
|-- tests/
|-- tools/
`-- README.md
```

This is a slightly handpicked selection of folders and subfolders which tries to highlight to most important folders to get started.
The frameworks, simulators, and tests folders contain many files that are not listed here. The main code used for modeling can be found in `src/tbp/monty/frameworks/models/`. 
