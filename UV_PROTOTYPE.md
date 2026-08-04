# `uv` Prototype Environment

While the repository contains a `uv.lock` file, this is currently experimental and not supported.
In the future this will change, but for now, avoid trying to use `uv` with this project.

## Setup Notes

Run the following to create a virtual environment and set up all the dependencies.

```sh
uv sync --extra dev --extra simulator_mujoco
```
