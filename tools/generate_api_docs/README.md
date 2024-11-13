# Monty documentation

This directory contains the sources and instructions to build **tbp.monty** 
API documentation.

## Requirements

### API Docs

The API documentation is collected from the source code **docstring** and built
using [sphinx](https://www.sphinx-doc.org/en/master/) tools. Use the following command to
install the required packages:

    pip install -e "../..[docs]"

Once the dependencies are installed you could use the following command to 
update the API documentation:

    make apidoc

And this command to generate `HTML` output:

    make html

The resulting HTML documentation can be found in the `build/html` directory.

Use the following commande to get a full list of all options availabe:

    make help

