# Future Work Widget

A tool that

1. processes an index.json that is produced by the github_readme_sync tool and converts it into a format consumable by the widget

## Setup

First, ensure you setup Monty. See [Getting Started - 2. Set up Your Environment](https://thousandbrainsproject.readme.io/docs/getting-started#2-set-up-your-environment).

Next, from the root Monty directory, install this tool's dependencies:

```
pip install -e '.[future_work_widget_tool]'
```

## Usage

```
python -m tools.future_work_widget.cli build /tmp/index.json tools/future_work_widget/app --help
usage: cli.py build [-h] [--docs-snippets-dir DOCS_SNIPPETS_DIR] index_file output_dir

positional arguments:
  index_file            The JSON file to validate and transform
  output_dir            The output directory to create and save data.json

optional arguments:
  -h, --help            show this help message and exit
  --docs-snippets-dir DOCS_SNIPPETS_DIR
                        Optional path to docs/snippets directory for validation files
```


## Tests

```
pytest -n 0 tools/future_work_widget
```

## Running Locally

To try the tool out simply run the following command from the tbp.monty directory:

```
source tools/future_work_widget/run-local.sh
```