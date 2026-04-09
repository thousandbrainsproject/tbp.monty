# `uv` Prototype Environment

While the repository contains a `uv.lock` file, this is currently experimental and not supported.
In the future this will change, but for now, avoid trying to use `uv` with this project.

## Setup Notes

Some notes on how to set up this environment.

This process is more complicated than it ideally would be at the moment because of some
misconfigurations in the `torch-scatter` and `torch-sparse` libraries and how they 
need `torch` to build.

We have to install `setuptools==81.0.0`, because later versions remove `pkg_resources`
and `torch-scatter` and `torch-sparse`, at least the versions we're on, still use it.

```sh
# The --seed is needed so we can build the torch packages
uv venv -p 3.9.22 --seed
uv pip install setuptools==81.0.0
uv pip install torch==1.13.1 # Use the version from pyproject.toml
uv sync --extra dev --extra simulator_mujoco
```
