import unittest
import pandas as pd
import openmdao.api as om
import numpy as np
import os
os.environ["pyna_language"] = 'python'
os.chdir('../../')
from pyNA.pyna import pyna
from pyNA.src.trajectory_src.atmosphere import Atmosphere


class TestAtmosphere(unittest.TestCase):

	def test_evaluate(self):

		# Load settings and aircraft
		settings = pyna.load_settings(case_name="nasa_stca_standard")

		# Inputs
		nn = 2
		z = np.linspace(0,11000, nn)

		# Create problem
		prob = om.Problem()
		comp = Atmosphere(num_nodes=nn, settings=settings)
		prob.model.add_subsystem("a", comp)
		prob.setup(force_alloc_complex=True)

		prob.set_val('a.z', z)

		# Run problem
		prob.run_model()

		self.assertAlmostEqual(prob.get_val('a.p_0')[0], 101325.)
		self.assertAlmostEqual(prob.get_val('a.rho_0')[0], 1.18385805)
		self.assertAlmostEqual(prob.get_val('a.drho_0_dz')[0], -1.14552297e-04)
		self.assertAlmostEqual(prob.get_val('a.T_0')[0], 298.1669)
		self.assertAlmostEqual(prob.get_val('a.c_0')[0], 346.15651388)
		self.assertAlmostEqual(prob.get_val('a.mu_0')[0], 1.83733457e-05)
		self.assertAlmostEqual(prob.get_val('a.I_0')[0], 409.79813483)
		self.assertAlmostEqual(prob.get_val('a.rh')[0],  70.)

	def test_partials(self):

		# Load settings and aircraft
		settings = pyna.load_settings(case_name="nasa_stca_standard")

		# Inputs
		nn = 20
		z = np.linspace(1,2000, nn)

		# Create problem
		prob = om.Problem()
		comp = Atmosphere(num_nodes=nn, settings=settings)
		prob.model.add_subsystem("a", comp)
		prob.setup(force_alloc_complex=True)

		prob.set_val('a.z', z)

		# Check partials 
		prob.check_partials(compact_print=True, method='cs')

if __name__ == '__main__':
	unittest.main()

