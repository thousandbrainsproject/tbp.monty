# Future Work Widget

A tool that

1. processes an index.json that is produced by the github_readme_sync tool and converts it into a format consumable by the widget

## Setup

First, ensure you setup Monty. See [Getting Started - 2. Set up Your Environment](https://thousandbrainsproject.readme.io/docs/getting-started#2-set-up-your-environment).

Next, from the root Monty directory, install this tool's dependencies:

```
pip install -e '.[dev,github_readme_sync_tool,future_work_widget_tool]'
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
