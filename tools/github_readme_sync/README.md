# github-readme-sync

A tool that

1. exports docs from readme.com to the file system
2. uploads docs from the file system to readme.com under a specified version.

Notes:
- This tool only supports guides in readme.com, it does not support API reference docs or other types of docs.
- It does not support moving categories as the readme API does not support moving categories.  You can move categories manually in the readme UI.

## Requirements
You need to have a readme account.
You need to have a readme API key.

## Setup

First, ensure you setup Monty. See [Getting Started - 2. Set up Your Environment](https://thousandbrainsproject.readme.io/docs/getting-started#2-set-up-your-environment).

Next, from the root Monty directory, install this tool's dependencies:

```
pip install -e '.[dev,github_readme_sync_tool]'
```

## Usage

### Setup environment variables
In your shell:

```
export README_API_KEY=<readme_api_key>
export IMAGE_PATH=thousandbrainsproject/tbp.monty/refs/heads/main/docs/figures
```

```
> python -m tools.github_readme_sync.cli -h

usage: cli.py [-h] {export,check,upload,check-external} ...

CLI tool to manage exporting, checking, and uploading docs.

positional arguments:
  {export,check,upload,check-external}
    export              Export the readme docs and create a hierarchy.md file
    check               Check the hierarchy.md file and ensure all docs exist
    upload              Upload the docs in the folder to ReadMe under the specified version
    check-external      Check external links in all markdown files from the specified directory

optional arguments:
  -h, --help            show this help message and exit
```

## Tests

```
pytest --cov=.
```


## Troubleshooting

You can set the `LOG_LEVEL` environment variable to `DEBUG` to get more detailed logs.

```
LOG_LEVEL=DEBUG python -m tools.github_readme_sync.cli check docs
```