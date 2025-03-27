# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.


class FakeClass:
    def __init__(self, data=0):
        self.data = data


class FakeMixin:
    pass


class FakeSubclass1(FakeClass):
    pass


class FakeSubclass2(FakeSubclass1):
    pass


class FakeSubclass3(FakeSubclass2):
    pass


class FakeSubclass4(FakeMixin, FakeSubclass3):
    pass
