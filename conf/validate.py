# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.


import hydra


@hydra.main(config_path=".", config_name="config", version_base=None)
def validate(cfg):
    print(cfg)

if __name__ == "__main__":
    validate()
