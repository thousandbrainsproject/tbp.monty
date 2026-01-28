# `uv` Prototype Environment

While the repository contains a `uv.lock` file, this is currently experimental and not supported.
In the future this will change, but for now, avoid trying to use `uv` with this project.

## Setup Notes

Some notes on how to set up this environment.

```sh
# The --seed is needed so we can build the torch packages
uv venv -p 3.9.22 --seed
uv pip install torch==1.13.1 # Use the version from pyproject.toml
uv sync --extra dev --extra simulator_mujoco
```
