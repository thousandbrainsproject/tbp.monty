# Copyright 2025 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
import dataclasses
import unittest
from typing import Any, NamedTuple, Union

from tbp.monty.frameworks.config_utils.config_args import Dataclass
from tbp.monty.frameworks.utils import dataclass_utils


@dataclasses.dataclass
class SimpleDataclass:
    field1: str
    field2: int


@dataclasses.dataclass
class DataclassWithDict:
    field1: str
    field2: int
    field3: dict = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class NestedDataclass:
    field1: str
    field2: SimpleDataclass


@dataclasses.dataclass
class NestedDataclassWithDict:
    field1: str
    field2: SimpleDataclass
    field3: dict = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class DeepNestedDataclass:
    field1: str
    field2: SimpleDataclass
    field3: NestedDataclass


@dataclasses.dataclass
class FakeDataclass:
    name: Any = None
    value: Union[list, tuple, dict, Dataclass] = dataclasses.field(default_factory=list)


class FakeNamedTuple(NamedTuple):
    name: Any
    value: Union[list, tuple, dict, Dataclass]


def sample_function(field1: str, field2: int):
    pass


def sample_function_with_default(field1: str, field2: int, field3: float = 0.1):
    pass


class SampleClass:
    def __init__(self, field1: str, field2: int):
        pass


class SampleClass2:
    """Has partially overlapping args with SampleClass."""

    def __init__(self, field2: int, field3: str):
        pass


class DataclassSerializationTest(unittest.TestCase):
    def test_simple_dataclass(self):
        obj = SimpleDataclass(field1="1", field2=1)

        datadict = dataclass_utils.as_dataclass_dict(obj)
        self.assertDictEqual(
            datadict,
            {
                "field1": "1",
                "field2": 1,
                "__dataclass_type__": f"{__name__}.SimpleDataclass",
            },
        )
        actual = dataclass_utils.from_dataclass_dict(datadict)
        self.assertEqual(obj, actual)

    def test_dataclass_with_dict(self):
        obj = DataclassWithDict(field1="1", field2=1, field3={"a": 1})
        datadict = dataclass_utils.as_dataclass_dict(obj)
        self.assertDictEqual(
            datadict,
            {
                "field1": "1",
                "field2": 1,
                "field3": {"a": 1},
                "__dataclass_type__": f"{__name__}.DataclassWithDict",
            },
        )
        actual = dataclass_utils.from_dataclass_dict(datadict)
        self.assertEqual(obj, actual)

    def test_nested_dataclass(self):
        obj = NestedDataclass(field1="1", field2=SimpleDataclass(field1="2", field2=2))
        datadict = dataclass_utils.as_dataclass_dict(obj)
        self.assertDictEqual(
            datadict,
            {
                "__dataclass_type__": f"{__name__}.NestedDataclass",
                "field1": "1",
                "field2": {
                    "__dataclass_type__": f"{__name__}.SimpleDataclass",
                    "field1": "2",
                    "field2": 2,
                },
            },
        )
        actual = dataclass_utils.from_dataclass_dict(datadict)
        self.assertEqual(obj, actual)

    def test_nested_dataclass_with_dict(self):
        obj = NestedDataclassWithDict(
            field1="1", field2=SimpleDataclass(field1="2", field2=2), field3={"a": 1}
        )
        datadict = dataclass_utils.as_dataclass_dict(obj)
        self.assertDictEqual(
            datadict,
            {
                "__dataclass_type__": f"{__name__}.NestedDataclassWithDict",
                "field1": "1",
                "field2": {
                    "__dataclass_type__": f"{__name__}.SimpleDataclass",
                    "field1": "2",
                    "field2": 2,
                },
                "field3": {"a": 1},
            },
        )
        actual = dataclass_utils.from_dataclass_dict(datadict)
        self.assertEqual(obj, actual)

    def test_deep_nested_dataclass(self):
        obj = DeepNestedDataclass(
            field1="1",
            field2=SimpleDataclass(field1="2", field2=2),
            field3=NestedDataclassWithDict(
                field1="3",
                field2=SimpleDataclass(field1="3", field2=3),
                field3={"c": 3},
            ),
        )
        datadict = dataclass_utils.as_dataclass_dict(obj)
        self.assertDictEqual(
            datadict,
            {
                "__dataclass_type__": f"{__name__}.DeepNestedDataclass",
                "field1": "1",
                "field2": {
                    "__dataclass_type__": f"{__name__}.SimpleDataclass",
                    "field1": "2",
                    "field2": 2,
                },
                "field3": {
                    "__dataclass_type__": f"{__name__}.NestedDataclassWithDict",
                    "field1": "3",
                    "field2": {
                        "__dataclass_type__": f"{__name__}.SimpleDataclass",
                        "field1": "3",
                        "field2": 3,
                    },
                    "field3": {"c": 3},
                },
            },
        )
        actual = dataclass_utils.from_dataclass_dict(datadict)
        self.assertEqual(obj, actual)


class CreateDataclassArgsTest(unittest.TestCase):
    def test_simple_functions(self):
        # Test simple function
        dc = dataclass_utils.create_dataclass_args("test1", sample_function)
        # Expected dataclass with 2 fields
        expected = {
            "field1": (str, dataclasses.MISSING),
            "field2": (int, dataclasses.MISSING),
        }
        actual = {f.name: (f.type, f.default) for f in dataclasses.fields(dc)}
        self.assertTrue(dataclasses.is_dataclass(dc))
        self.assertEqual(dc.__name__, "test1")
        self.assertDictEqual(actual, expected)

    def test_simple_functions_with_default(self):
        # Test simple function with default values
        dc = dataclass_utils.create_dataclass_args(
            "test2", sample_function_with_default
        )
        # Expected dataclass with 3 fields and default value on the third
        expected = {
            "field1": (str, dataclasses.MISSING),
            "field2": (int, dataclasses.MISSING),
            "field3": (float, 0.1),
        }
        actual = {f.name: (f.type, f.default) for f in dataclasses.fields(dc)}
        self.assertTrue(dataclasses.is_dataclass(dc))
        self.assertEqual(dc.__name__, "test2")
        self.assertDictEqual(actual, expected)

    def test_class_method(self):
        # Test class method
        dc = dataclass_utils.create_dataclass_args("test3", SampleClass.__init__)
        # Expected dataclass with 3 fields and default value on the third
        expected = {
            "field1": (str, dataclasses.MISSING),
            "field2": (int, dataclasses.MISSING),
        }
        actual = {f.name: (f.type, f.default) for f in dataclasses.fields(dc)}
        self.assertTrue(dataclasses.is_dataclass(dc))
        self.assertEqual(dc.__name__, "test3")
        self.assertDictEqual(actual, expected)


class ConfigToDictTest(unittest.TestCase):
    def test_simple_dataclass(self):
        config = SimpleDataclass(field1="1", field2=1)

        datadict = dataclass_utils.config_to_dict(config)
        self.assertDictEqual(
            datadict,
            {
                "field1": "1",
                "field2": 1,
            },
        )

    def test_dataclass_with_dict(self):
        config = DataclassWithDict(field1="1", field2=1, field3={"a": 1})
        datadict = dataclass_utils.config_to_dict(config)
        self.assertDictEqual(
            datadict,
            {
                "field1": "1",
                "field2": 1,
                "field3": {"a": 1},
            },
        )

    def test_nested_dataclass(self):
        config = NestedDataclass(
            field1="1", field2=SimpleDataclass(field1="2", field2=2)
        )
        datadict = dataclass_utils.config_to_dict(config)
        self.assertDictEqual(
            datadict,
            {
                "field1": "1",
                "field2": {
                    "field1": "2",
                    "field2": 2,
                },
            },
        )

    def test_nested_dataclass_with_dict(self):
        config = NestedDataclassWithDict(
            field1="1", field2=SimpleDataclass(field1="2", field2=2), field3={"a": 1}
        )
        datadict = dataclass_utils.config_to_dict(config)
        self.assertDictEqual(
            datadict,
            {
                "field1": "1",
                "field2": {
                    "field1": "2",
                    "field2": 2,
                },
                "field3": {"a": 1},
            },
        )

    def test_deep_nested_dataclass(self):
        config = DeepNestedDataclass(
            field1="1",
            field2=SimpleDataclass(field1="2", field2=2),
            field3=NestedDataclassWithDict(
                field1="3",
                field2=SimpleDataclass(field1="3", field2=3),
                field3={"c": 3},
            ),
        )
        datadict = dataclass_utils.config_to_dict(config)
        self.assertDictEqual(
            datadict,
            {
                "field1": "1",
                "field2": {
                    "field1": "2",
                    "field2": 2,
                },
                "field3": {
                    "field1": "3",
                    "field2": {
                        "field1": "3",
                        "field2": 3,
                    },
                    "field3": {"c": 3},
                },
            },
        )

    def test_dict_with_dataclass(self):
        config = dict(
            args=DeepNestedDataclass(
                field1="1",
                field2=SimpleDataclass(field1="2", field2=2),
                field3=NestedDataclassWithDict(
                    field1="3",
                    field2=SimpleDataclass(field1="3", field2=3),
                    field3={"c": 3},
                ),
            )
        )
        datadict = dataclass_utils.config_to_dict(config)
        self.assertDictEqual(
            datadict,
            {
                "args": {
                    "field1": "1",
                    "field2": {
                        "field1": "2",
                        "field2": 2,
                    },
                    "field3": {
                        "field1": "3",
                        "field2": {
                            "field1": "3",
                            "field2": 3,
                        },
                        "field3": {"c": 3},
                    },
                }
            },
        )


class ConfigToDictBasicTest(unittest.TestCase):
    """Tests config_to_dict non-nested data."""

    def test_with_basic_attr(self):
        dict_config = {"name": 0, "value": "a basic attribute"}
        dataclass_config = FakeDataclass(**dict_config)
        expected = {
            "name": 0,
            "value": "a basic attribute",
        }
        dict_result = dataclass_utils.config_to_dict(dict_config)
        dataclass_result = dataclass_utils.config_to_dict(dataclass_config)
        self.assertDictEqual(dict_result, expected)
        self.assertDictEqual(dataclass_result, expected)

    def test_with_basic_dict(self):
        dict_config = {"name": 0, "value": {"a": 0, "b": 1}}
        dataclass_config = FakeDataclass(**dict_config)
        expected = {
            "name": 0,
            "value": {"a": 0, "b": 1},
        }
        dict_result = dataclass_utils.config_to_dict(dict_config)
        dataclass_result = dataclass_utils.config_to_dict(dataclass_config)
        self.assertDictEqual(dict_result, expected)
        self.assertDictEqual(dataclass_result, expected)

    def test_with_basic_list(self):
        dict_config = {"name": 0, "value": [0, 1]}
        dataclass_config = FakeDataclass(**dict_config)
        expected = {
            "name": 0,
            "value": [0, 1],
        }
        dict_result = dataclass_utils.config_to_dict(dict_config)
        dataclass_result = dataclass_utils.config_to_dict(dataclass_config)
        self.assertDictEqual(dict_result, expected)
        self.assertDictEqual(dataclass_result, expected)

    def test_with_basic_tuple(self):
        dict_config = {"name": 0, "value": (0, 1)}
        dataclass_config = FakeDataclass(**dict_config)
        expected = {
            "name": 0,
            "value": (0, 1),
        }
        dict_result = dataclass_utils.config_to_dict(dict_config)
        dataclass_result = dataclass_utils.config_to_dict(dataclass_config)
        self.assertDictEqual(dict_result, expected)
        self.assertDictEqual(dataclass_result, expected)

    def test_with_basic_named_tuple(self):
        dict_config = {"name": 0, "value": FakeNamedTuple(name=1, value=[])}
        dataclass_config = FakeDataclass(**dict_config)
        expected = {
            "name": 0,
            "value": FakeNamedTuple(name=1, value=[]),
        }
        dict_result = dataclass_utils.config_to_dict(dict_config)
        dataclass_result = dataclass_utils.config_to_dict(dataclass_config)
        self.assertDictEqual(dict_result, expected)
        self.assertIsInstance(dict_result["value"], FakeNamedTuple)
        self.assertDictEqual(dataclass_result, expected)
        self.assertIsInstance(dataclass_result["value"], FakeNamedTuple)


class ConfigToDictNestedTest(unittest.TestCase):
    """Tests config_to_dict nested data."""

    def test_with_nested_attr(self):
        obj_0 = FakeDataclass(name=0)
        dict_config = {"name": 1, "value": obj_0}
        dataclass_config = FakeDataclass(**dict_config)
        expected = {
            "name": 1,
            "value": {
                "name": 0,
                "value": [],
            },
        }
        dict_result = dataclass_utils.config_to_dict(dict_config)
        dataclass_result = dataclass_utils.config_to_dict(dataclass_config)
        self.assertDictEqual(dict_result, expected)
        self.assertDictEqual(dataclass_result, expected)

    def test_with_nested_dict(self):
        obj_0, obj_1 = FakeDataclass(name=0), FakeDataclass(name=1)
        dict_config = {"name": 2, "value": {"a": obj_0, "b": obj_1}}
        dataclass_config = FakeDataclass(**dict_config)
        expected = {
            "name": 2,
            "value": {
                "a": {
                    "name": 0,
                    "value": [],
                },
                "b": {
                    "name": 1,
                    "value": [],
                },
            },
        }
        dict_result = dataclass_utils.config_to_dict(dict_config)
        dataclass_result = dataclass_utils.config_to_dict(dataclass_config)
        self.assertDictEqual(dict_result, expected)
        self.assertDictEqual(dataclass_result, expected)

    def test_with_nested_list(self):
        obj_0, obj_1 = FakeDataclass(name=0), FakeDataclass(name=1)
        dict_config = {"name": 2, "value": [obj_0, obj_1]}
        dataclass_config = FakeDataclass(**dict_config)
        expected = {
            "name": 2,
            "value": [
                {
                    "name": 0,
                    "value": [],
                },
                {
                    "name": 1,
                    "value": [],
                },
            ],
        }
        dict_result = dataclass_utils.config_to_dict(dict_config)
        dataclass_result = dataclass_utils.config_to_dict(dataclass_config)
        self.assertDictEqual(dict_result, expected)
        self.assertDictEqual(dataclass_result, expected)

    def test_with_nested_tuple(self):
        obj_0, obj_1 = FakeDataclass(name=0), FakeDataclass(name=1)
        dict_config = {"name": 2, "value": (obj_0, obj_1)}
        dataclass_config = FakeDataclass(**dict_config)
        expected = {
            "name": 2,
            "value": (
                {
                    "name": 0,
                    "value": [],
                },
                {
                    "name": 1,
                    "value": [],
                },
            ),
        }
        dict_result = dataclass_utils.config_to_dict(dict_config)
        dataclass_result = dataclass_utils.config_to_dict(dataclass_config)
        self.assertDictEqual(dict_result, expected)
        self.assertDictEqual(dataclass_result, expected)

    def test_with_nested_named_tuple(self):
        obj_0, obj_1 = FakeDataclass(name=0), FakeDataclass(name=1)
        dict_config = {"name": 2, "value": FakeNamedTuple(name=0, value=[obj_0, obj_1])}
        dataclass_config = FakeDataclass(**dict_config)
        expected = {
            "name": 2,
            "value": FakeNamedTuple(
                name=0,
                value=[
                    {
                        "name": 0,
                        "value": [],
                    },
                    {
                        "name": 1,
                        "value": [],
                    },
                ],
            ),
        }
        dict_result = dataclass_utils.config_to_dict(dict_config)
        dataclass_result = dataclass_utils.config_to_dict(dataclass_config)
        self.assertDictEqual(dict_result, expected)
        self.assertIsInstance(dict_result["value"], FakeNamedTuple)
        self.assertDictEqual(dataclass_result, expected)
        self.assertIsInstance(dataclass_result["value"], FakeNamedTuple)

    def test_deeply_nested(self):
        # Lowest-level objects
        obj_0 = FakeDataclass(name=0, value={"a": 0, "b": 1})
        obj_0_dict = {"name": 0, "value": {"a": 0, "b": 1}}

        obj_1 = FakeDataclass(name=1, value=[obj_0])
        obj_1_dict = {"name": 1, "value": [obj_0_dict]}

        obj_2 = FakeDataclass(name=2, value=(obj_1,))
        obj_2_dict = {"name": 2, "value": (obj_1_dict,)}

        obj_3 = FakeDataclass(name=3, value=FakeNamedTuple(name=None, value=[obj_2]))
        obj_3_dict = {"name": 3, "value": FakeNamedTuple(name=None, value=[obj_2_dict])}

        dict_config = {"name": 4, "value": obj_3}
        dataclass_config = FakeDataclass(**dict_config)
        expected = {"name": 4, "value": obj_3_dict}

        dict_result = dataclass_utils.config_to_dict(dict_config)
        dataclass_result = dataclass_utils.config_to_dict(dataclass_config)
        self.assertDictEqual(dict_result, expected)
        self.assertDictEqual(dataclass_result, expected)


class IsConfigLikeTest(unittest.TestCase):
    def test_dataclass_instance(self):
        self.assertTrue(dataclass_utils.is_config_like(FakeDataclass()))

    def test_dict(self):
        self.assertTrue(dataclass_utils.is_config_like({0: "a", 1: "b"}))

    def test_dataclass_type(self):
        self.assertFalse(dataclass_utils.is_config_like(FakeDataclass))

    def test_list(self):
        self.assertFalse(dataclass_utils.is_config_like([0, 1]))

    def test_tuple(self):
        self.assertFalse(dataclass_utils.is_config_like((0, 1)))

    def test_named_tuple(self):
        self.assertFalse(
            dataclass_utils.is_config_like(FakeNamedTuple(name=0, value=[]))
        )


class IsDataclassInstanceTest(unittest.TestCase):
    def test_dataclass_instance(self):
        self.assertTrue(dataclass_utils.is_dataclass_instance(FakeDataclass()))

    def test_dataclass_type(self):
        self.assertFalse(dataclass_utils.is_dataclass_instance(FakeDataclass))

    def test_dict(self):
        self.assertFalse(dataclass_utils.is_dataclass_instance({0: "a", 1: "b"}))

    def test_list(self):
        self.assertFalse(dataclass_utils.is_dataclass_instance([0, 1]))

    def test_tuple(self):
        self.assertFalse(dataclass_utils.is_dataclass_instance((0, 1)))

    def test_named_tuple(self):
        self.assertFalse(
            dataclass_utils.is_dataclass_instance(FakeNamedTuple(name=0, value=[]))
        )


class GetSubsetArgsTest(unittest.TestCase):
    def test_get_subset_args(self):
        pooled_dict = dict(field1="a", field2=3, field3="b")
        subset_args = dataclass_utils.get_subset_of_args(
            pooled_dict, SampleClass2.__init__
        )

        expected_subset_args = dict(field2=3, field3="b")
        self.assertEqual(subset_args, expected_subset_args)


if __name__ == "__main__":
    unittest.main()
