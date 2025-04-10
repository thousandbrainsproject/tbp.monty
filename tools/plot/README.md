# Plot a figure

A tool that plots a figure from experiment log output.

## Setup

Ensure you setup Monty. See [Getting Started - 2. Set up Your Environment](https://thousandbrainsproject.readme.io/docs/getting-started#2-set-up-your-environment).

No additional dependencies are required.

## Usage

```
$ python -m tools.plot.cli -h

usage: cli.py [-h]

usage: cli.py [-h] [--debug] {objects_evidence_over_time,pose_error_over_time} ...

Plot a figure

positional arguments:
  {objects_evidence_over_time,pose_error_over_time}
    objects_evidence_over_time
                        Plot evidence scores for each object over time.
    pose_error_over_time
                        Plot MLH pose error and theoretical limits over time.

optional arguments:
  -h, --help            show this help message and exit
  --debug               Enable debug logging
```

## Tests

```
pytest
```
