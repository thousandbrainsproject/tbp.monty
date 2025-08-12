import pytest
import unittest

from tbp.monty.simulators.simulator import Simulator

class FakeSimulator(Simulator):
    @property
    def num_objects(self) -> int:
        return 42

    @property
    def action_space(self):
        return {}

    @property
    def observations(self):
        return []

class SimulatorTest(unittest.TestCase):
    def test_fails_when_properties_called_as_functions(self):
        fake_sim = FakeSimulator()
        with pytest.raises(TypeError):
            fake_sim.num_objects()

        with pytest.raises(TypeError):
            fake_sim.action_space()

        with pytest.raises(TypeError):
            fake_sim.observations()

    def test_properties(self):
        fake_sim = FakeSimulator()
        self.assertEqual(42, fake_sim.num_objects)
        self.assertEqual({}, fake_sim.action_space)
        self.assertEqual([], fake_sim.observations)
