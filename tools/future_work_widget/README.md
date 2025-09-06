# Future Work Widget

A tool that

1. processes an index.json that is produced by the github_readme_sync tool and makes it consumable by the widget
2. uploads that data file and the widget HTML to an S3 bucket.

## Setup

First, ensure you setup Monty. See [Getting Started - 2. Set up Your Environment](https://thousandbrainsproject.readme.io/docs/getting-started#2-set-up-your-environment).

Next, from the root Monty directory, install this tool's dependencies:

```
pip install -e '.[dev,github_readme_sync_tool,future_work_widget]'
```

## Usage

### Setup environment variables
In your shell:


```
> python -m tools.future_work_widget.cli /tmp/index.json


```

## Tests

```
pytest --cov=.
```
