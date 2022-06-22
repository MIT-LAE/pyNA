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
from pyNA.src.trajectory_src.aerodynamics import Aerodynamics


class TestAerodynamics(unittest.TestCase):

	def test_evaluate(self):

		# Load settings and aircraft
		settings = pyna.load_settings(case_name="nasa_stca_standard")
		settings.pyNA_directory = '.'
		ac = Aircraft(name=settings.ac_name, version=settings.ac_version, settings=settings)

		# Inputs
		nn = 20
		c_l = 0.5*np.ones(nn)
		c_d = 0.1*np.ones(nn)
		rho_0 = 1.*np.ones(nn)
		v = 100.*np.ones(nn)
		c_0 = 300.*np.ones(nn)

		# Create problem
		prob = om.Problem()
		comp = Aerodynamics(num_nodes=nn, ac=ac, phase='vnrs')
		prob.model.add_subsystem("a", comp)
		prob.setup(force_alloc_complex=True)

		prob.set_val('a.c_l', c_l)
		prob.set_val('a.c_d', c_d)
		prob.set_val('a.rho_0', rho_0)
		prob.set_val('a.v', v)
		prob.set_val('a.c_0', c_0)

		# Run problem
		prob.run_model()

		self.assertEqual(prob.get_val('a.q').tolist(), (0.5*rho_0*v**2).tolist())
		self.assertEqual(prob.get_val('a.L').tolist(), (c_l*0.5*rho_0*v**2*ac.af_S_w).tolist())
		self.assertEqual(prob.get_val('a.D').tolist(), (c_d*0.5*rho_0*v**2*ac.af_S_w).tolist())
		self.assertEqual(prob.get_val('a.M_0').tolist(), (1/3*np.ones(nn)).tolist())

	def test_partials(self):
		
		# Load settings and aircraft
		settings = pyna.load_settings(case_name="nasa_stca_standard")
		settings.pyNA_directory = '.'
		ac = Aircraft(name=settings.ac_name, version=settings.ac_version, settings=settings)

		# Inputs
		nn = 2
		rho_0 = 1.225*np.ones(nn)
		v = 100.*np.ones(nn)
		c_0 = 330.*np.ones(nn)

		# Create problem
		prob = om.Problem()
		comp = Aerodynamics(num_nodes=nn, ac=ac, phase='vnrs')
		prob.model.add_subsystem("a", comp)
		prob.setup(force_alloc_complex=True)

		prob.set_val('a.rho_0', rho_0)
		prob.set_val('a.v', v)
		prob.set_val('a.c_0', c_0)

		# Check partials 
		prob.check_partials(compact_print=True, method='cs')


if __name__ == '__main__':
	unittest.main()
