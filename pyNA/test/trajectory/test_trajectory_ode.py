import unittest
import numpy as np
import openmdao.api as om
import pandas as pd
import pdb
import sys
sys.path.append("../..")
import os
os.environ["pyna_language"] = 'python'
from pyNA.src.trajectory_src.trajectory_ode import TrajectoryODE
from pyNA.pyna import pyna


class TestPropulsion(unittest.TestCase):

	def test_partials(self):

		pyna_settings = pyna.load_settings(case_name="stca")
		pyna_settings.pyNA_directory = 'pyNA'
		pyna_settings.engine_file_name = 'engine_deck_stca.csv'

		py = pyna(pyna_settings)
		py.ac.load_aerodynamics(settings=py.settings)
		py.engine.load_deck(settings=py.settings)

		# Inputs
		nn = 10
		mode = 'noise'

		# Create problem
		phase = 'liftoff'
		prob = om.Problem()
		prob.model.add_subsystem('t', TrajectoryODE(num_nodes=nn, phase=phase, ac=py.ac, engine=py.engine, settings=py.settings, objective="noise"))
		prob.setup(force_alloc_complex=True)

		x = np.linspace(1, 100, nn)
		z = np.linspace(1, 100, nn)
		v = np.linspace(1, 100, nn)
		alpha = np.linspace(1, 10, nn)
		gamma = np.linspace(1, 15, nn)
		TS = np.linspace(0.8, 1., nn)
		theta_flaps = np.linspace(1, 10., nn)
		theta_slats = np.linspace(1, 10., nn)

		prob.set_val('t.x', x)
		prob.set_val('t.z', z)
		prob.set_val('t.v', v)
		prob.set_val('t.alpha', alpha)
		prob.set_val('t.gamma', gamma)
		prob.set_val('t.propulsion.TS', TS)
		prob.set_val('t.theta_flaps', theta_flaps)
		prob.set_val('t.theta_slats', theta_slats)

		prob.run_model()

		# Check partials
		prob.check_partials(method='cs', compact_print=True)


if __name__ == '__main__':
	unittest.main()





