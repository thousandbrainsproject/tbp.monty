# Monty documentation

This directory contains the sources and instructions to build **tbp.monty** API documentation.

## Requirements

### API Docs

The API documentation is collected from the source code **docstring** and built
using [sphinx](https://www.sphinx-doc.org/en/master/) tools.

From the `tbp.monty` directory, use the following command to install the required packages:

    pip install -e '.[generate_api_docs_tool]'

Once the dependencies are installed, change directory to `tools/generate_api_docs` and use the following command to
create the API documentation source files:

    make apidoc

And this command to generate `HTML` output:

    make html

The resulting HTML documentation can be found in the `build/html` directory.

Use the following command to get a full list of all options availabe:

    make help

