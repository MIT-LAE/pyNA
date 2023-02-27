import unittest
import pdb
import openmdao.api as om
import numpy as np
from pyNA.pyna import pyna
from pyNA.src.trajectory_model.aerodynamics import Aerodynamics
from openmdao.utils.assert_utils import assert_check_partials


class TestAerodynamics(unittest.TestCase):

	def test_evaluate(self):
		
		settings = dict()
		settings['case_name'] = 'stca'
		settings['ac_name'] = 'stca'
		settings['engine_deck_file_name'] = 'engine_deck_stca.csv'

		py = pyna(settings, trajectory_mode='model')
		py.aircraft.get_aerodynamics_deck(settings=py.settings)

		nn = 1
		
		prob = om.Problem()
		prob.model.add_subsystem("a", Aerodynamics(vec_size=nn, extrapolate=True, method='cubic', aircraft=py.aircraft))
		prob.setup(force_alloc_complex=True)

		prob.set_val('a.alpha', np.linspace(1, 10, nn))
		prob.set_val('a.theta_flaps', np.ones(nn))
		prob.set_val('a.theta_slats', np.ones(nn))
		prob.run_model()

		self.assertAlmostEqual(prob.get_val('a.c_l')[0], -0.05445165484715201, 4)
		self.assertAlmostEqual(prob.get_val('a.c_d')[0], 0.010356828289943939, 4)
		self.assertAlmostEqual(prob.get_val('a.c_l_max')[0], 1.0688052273911788, 4)

	def test_partials(self):

		settings = dict()
		settings['case_name'] = 'stca'
		settings['ac_name'] = 'stca'
		py = pyna(settings, trajectory_mode='model')
		py.aircraft.get_aerodynamics_deck(settings=py.settings)
		
		nn = 10
		
		prob = om.Problem()
		prob.model.add_subsystem("a", Aerodynamics(vec_size=nn, extrapolate=True, method='cubic', aircraft=py.aircraft))
		prob.setup(force_alloc_complex=True)

		prob.set_val('a.alpha', np.linspace(1, 10, nn))
		prob.set_val('a.theta_flaps', np.ones(nn))
		prob.set_val('a.theta_slats', np.ones(nn))
		prob.run_model()

		data = prob.check_partials(compact_print=True, method='cs')
		assert_check_partials(data, atol=1e-6, rtol=1e-6)

if __name__ == '__main__':
	unittest.main()
