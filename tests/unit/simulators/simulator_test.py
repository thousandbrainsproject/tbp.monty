import pytest
import unittest

from tbp.monty.simulators.simulator import Simulator

class FakeSimulator(Simulator):
    @property
    def num_objects(self) -> int:
        return 42

class SimulatorTest(unittest.TestCase):
    def test_fails_when_called_as_function(self):
        fake_sim = FakeSimulator()
        with pytest.raises(TypeError):
            fake_sim.num_objects()

    def test_num_objects_property(self):
        fake_sim = FakeSimulator()
        self.assertEqual(42, fake_sim.num_objects)
