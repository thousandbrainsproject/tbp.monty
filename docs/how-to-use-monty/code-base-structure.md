---
title: Code Base Structure
---

# Code Base Structure

The basic repository structure looks as follows:

```
.
|-- docs/                       # .md files for documentation
|-- rfcs/                       # Merged RFCs
|-- benchmarks/                 # experiments testing Monty
|   |-- configs/
|   |-- run_parallel.py
|   `-- run.py
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
|           |-- actions
|           |-- agents
|           |-- sensors
|           `-- simulator
|-- tests/                      # Unit tests
|-- tools/
`-- README.md
```

This is a slightly handpicked selection of folders and subfolders which tries to highlight to most important folders to get started.
The frameworks, simulators, and tests folders contain many files that are not listed here. The main code used for modeling can be found in `src/tbp/monty/frameworks/models/`. 