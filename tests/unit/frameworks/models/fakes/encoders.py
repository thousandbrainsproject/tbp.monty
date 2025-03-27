# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.


import json
from typing import Any

from tests.unit.frameworks.models.fakes.encoder_classes import FakeClass


class FakeJSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, FakeClass):
            return obj.data
        return super().default(obj)


def fake_encoder(obj: Any) -> Any:
    return obj.data


def fake_dict_encoder(obj: Any) -> Any:
    return dict(data=obj.data)


def fake_list_encoder(obj: Any) -> Any:
    return [obj.data]


def fake_mixin_encoder(obj: Any) -> Any:
    return dict(mixin=obj.data)
