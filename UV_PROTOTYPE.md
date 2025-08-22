# `uv` Prototype Environment

While the repository contains a `uv.lock` file, this is currently experimental and not supported. 
In the future this will change, but for now, avoid trying to use `uv` with this project.

## Setup Notes

Some notes on how to set up this environment. These aren't as simple as I'd like because of having to build versions for different platforms.

```sh
# The --seed is needed so we can build the torch packages
uv venv -p 3.2. --seed
# have to install torch separate from the rest
# because torch-scatter, etc. have to build without
# isolation
uv pip install torch
uv sync --extra dev --extra simulator_mujoco
```
