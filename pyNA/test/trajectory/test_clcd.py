import unittest
import pdb
import pandas as pd
import openmdao.api as om
import numpy as np
import os
os.environ["pyna_language"] = 'python'
os.chdir('../../')
from pyNA.pyna import pyna
from pyNA.src.settings import Settings
from pyNA.src.aircraft import Aircraft
from pyNA.src.trajectory_src.clcd import CLCD


class TestFlightDynamics(unittest.TestCase):

	def test_evaluate(self):

		# Load settings and aircraft
		settings = pyna.load_settings(case_name="stca")
		settings.engine_file_name = 'engine_deck_stca.csv'
		settings.pyNA_directory = '.'

		py = pyna(settings)
		py.ac.load_aerodynamics(settings=settings)
		py.engine.load_deck(settings=py.settings)

		# Inputs
		nn = 1

		# Create problem
		prob = om.Problem()
		prob.model.add_subsystem("a", CLCD(vec_size=nn, extrapolate=True, method='cubic', ac=py.ac))
		prob.setup(force_alloc_complex=True)

		prob.set_val('a.alpha', np.ones(nn))
		prob.set_val('a.theta_flaps', np.ones(nn))
		prob.set_val('a.theta_slats', np.ones(nn))

		# Run problem
		prob.run_model()

		self.assertAlmostEqual(prob.get_val('a.c_l')[0], -0.05445165)
		self.assertAlmostEqual(prob.get_val('a.c_l_max')[0], 1.06880523)
		self.assertAlmostEqual(prob.get_val('a.c_d')[0], 0.01035683)


	def test_partials(self):
		# Load settings and aircraft
		settings = pyna.load_settings(case_name="stca")
		settings.engine_file_name = 'engine_deck_stca.csv'
		settings.pyNA_directory = '.'

		py = pyna(settings)
		py.ac.load_aerodynamics(settings=settings)
		py.engine.load_deck(settings=py.settings)

		# Inputs
		nn = 20

		# Create problem
		prob = om.Problem()
		prob.model.add_subsystem("a", CLCD(vec_size=nn, extrapolate=True, method='cubic', ac=py.ac))
		prob.setup(force_alloc_complex=True)

		prob.set_val('a.alpha', np.ones(nn))

		# Check partials 
		prob.check_partials(compact_print=True, method='cs')


if __name__ == '__main__':
	unittest.main()
