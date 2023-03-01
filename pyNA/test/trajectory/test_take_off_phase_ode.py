import unittest
import numpy as np
import openmdao.api as om
import pdb
from pyNA.src.trajectory_model.take_off_phase_ode import TakeOffPhaseODE
from pyNA.pyna import pyna
from openmdao.utils.assert_utils import assert_check_partials


class TestTakeOffPhaseODE(unittest.TestCase):

	def test_partials(self):

		py = pyna(trajectory_mode='model',
				  case_name = 'stca',
				  fan_inlet_source = True,
				  fan_discharge_source = True,
				  core_source = True,
				  jet_mixing_source = True,
				  jet_shock_source = True,
				  airframe_source = True)
		py.aircraft.get_aerodynamics_deck(settings=py.settings)		
		py.aircraft.engine.get_performance_deck(settings=py.settings)
		
		# Inputs
		nn = 10
		objective = 'time'

		# Create problem
		phase = 'groundroll'
		prob = om.Problem()
		prob.model.add_subsystem('t', TakeOffPhaseODE(num_nodes=nn, phase=phase, settings=py.settings, aircraft=py.aircraft, objective=objective))

		prob.setup(force_alloc_complex=True)

		x = np.linspace(1, 100, nn)
		z = np.linspace(1, 100, nn)
		v = np.linspace(1, 100, nn)
		alpha = np.linspace(1, 10, nn)
		gamma = np.linspace(1, 15, nn)
		tau = np.linspace(0.8, 1., nn)
		theta_flaps = np.linspace(1, 10., nn)
		theta_slats = np.linspace(1, 10., nn)

		prob.set_val('t.x', x)
		prob.set_val('t.z', z)
		prob.set_val('t.v', v)
		prob.set_val('t.alpha', alpha)
		prob.set_val('t.gamma', gamma)
		prob.set_val('t.propulsion.tau', tau)
		prob.set_val('t.theta_flaps', theta_flaps)
		prob.set_val('t.theta_slats', theta_slats)

		prob.run_model()

		# Check partials
		data = prob.check_partials(method='cs', compact_print=True)
		assert_check_partials(data, atol=1e-6, rtol=1e-6)

if __name__ == '__main__':
	unittest.main()





