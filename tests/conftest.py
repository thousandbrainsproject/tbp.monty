# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
import pytest


@pytest.hookimpl(trylast=True)
def pytest_collection_finish(session):
    """Keep per-test coverage contexts on xdist workers, not the controller."""
    if getattr(session.config, "workerinput", None) is not None:
        return
    if getattr(session.config.option, "numprocesses", None) in (None, 0):
        return
    session.config.pluginmanager.unregister(name="_cov_contexts")
