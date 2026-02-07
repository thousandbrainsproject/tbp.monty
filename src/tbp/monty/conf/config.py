# Copyright 2025-2026 Thousand Brains Project
# Copyright 2021-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

__all__ = ["Config"]


class Config:
    def __init__(self, config: DictConfig) -> None:
        self.config = config

    def __repr__(self) -> str:
        return self.config.__repr__()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.config, name)

    def to_yaml(self) -> str:
        return OmegaConf.to_yaml(self.config)

    @classmethod
    def get(cls, args: dict) -> Config:
        hydra_args = [f"{k}={v}" for k, v in args.items() if k in {"experiment"}]
        with hydra.initialize(config_path=".", version_base=None):
            return cls(hydra.compose(config_name="experiment", overrides=hydra_args))
