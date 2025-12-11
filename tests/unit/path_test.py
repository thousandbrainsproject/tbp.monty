# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.


from pathlib import Path

import pytest

from tbp.monty.path import monty_data_path

MOCK_MONTY_DATA = "/monty/data"


@pytest.mark.parametrize(
    ("data_path", "subpath", "expected_result"),
    [
        (None, "default/subpath", Path(MOCK_MONTY_DATA) / "default" / "subpath"),
        (None, "", Path(MOCK_MONTY_DATA)),
        ("/custom/data/path", "default/subpath", Path("/custom/data/path")),
        (Path("/custom/already/path"), "default/subpath", Path("/custom/already/path")),
        (None, Path("sub") / "aspath", Path(MOCK_MONTY_DATA) / "sub" / "aspath"),
        # An empty custom path is treated as the current directory
        ("", "default/subpath", Path()),
    ],
)
def test_monty_data_path(data_path, subpath, expected_result):
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("MONTY_DATA", MOCK_MONTY_DATA)

        result = monty_data_path(data_path, subpath)
        assert result == expected_result
