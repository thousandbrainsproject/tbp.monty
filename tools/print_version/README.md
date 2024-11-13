# Print Version

A tool that prints the current version of the project

## Setup

First, ensure you setup Monty. See [Getting Started - 2. Set up Your Environment](https://thousandbrainsproject.readme.io/docs/getting-started#2-set-up-your-environment).

Next, from the root Monty directory, install this tool's dependencies:

```
pip install -e '.[dev,print_version_tool]'
```

## Usage

```
> python -m tools.print_version.cli -h

usage: cli.py [-h] [{full,major,minor,patch}]

Version parser

positional arguments:
  {full,major,minor,patch}
                        Which part of the version to return

options:
  -h, --help            show this help message and exit
```

## Tests

```
pytest --cov=.
```
