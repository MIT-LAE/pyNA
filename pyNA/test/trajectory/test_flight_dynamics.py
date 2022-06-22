import unittest
import pandas as pd
import openmdao.api as om
import numpy as np
import os
os.environ["pyna_language"] = 'python'
os.chdir('../../')
from pyNA.pyna import pyna
from pyNA.src.settings import Settings
from pyNA.src.aircraft import Aircraft
from pyNA.src.trajectory_src.flight_dynamics import FlightDynamics


class TestFlightDynamics(unittest.TestCase):

	def test_evaluate(self):

		# Load settings and aircraft
		settings = pyna.load_settings(case_name="nasa_stca_standard")
		settings.pyNA_directory = '.'
		ac = Aircraft(name=settings.ac_name, version=settings.ac_version, settings=settings)

		# Inputs
		nn = 1
		phase_name = 'flapsdown'

		x = np.linspace(1,100,nn)
		z = np.linspace(1,100,nn)
		v = np.linspace(1,100,nn)
		F_n = 210000*np.ones(nn,)
		alpha = np.linspace(1,10,nn)
		L = 54000*9.81*np.ones(nn,)
		D = 3000*np.ones(nn,)
		gamma = np.linspace(1,20,nn)
		rho_0 = 1.25*np.ones(nn,)
		drho_0_dz = -0.05*np.ones(nn,)

		# Create problem
		prob = om.Problem()
		comp = FlightDynamics(num_nodes=nn, settings=settings, phase=phase_name, ac=ac, objective='noise')
		prob.model.add_subsystem("f", comp)
		prob.setup(force_alloc_complex=True)
		    
		prob.set_val('f.x', x)
		prob.set_val('f.z', z)
		prob.set_val('f.v', v)
		prob.set_val('f.F_n', F_n)
		prob.set_val('f.alpha', alpha)
		prob.set_val('f.L', L)
		prob.set_val('f.D', D)
		prob.set_val('f.gamma', gamma)
		prob.set_val('f.rho_0', rho_0)
		prob.set_val('f.drho_0_dz', drho_0_dz)

		# Run problem
		prob.run_model()

		self.assertAlmostEqual(prob.model.get_val('f.x_dot')[0], 0.9998477)
		self.assertAlmostEqual(prob.model.get_val('f.z_dot')[0], 0.01745241)
		self.assertAlmostEqual(prob.model.get_val('f.alpha_dot')[0], 1.)
		self.assertAlmostEqual(prob.model.get_val('f.gamma_dot')[0], 1.)
		self.assertAlmostEqual(prob.model.get_val('f.v_dot')[0], 1.)
		self.assertAlmostEqual(prob.model.get_val('f.eas_dot')[0], 1.00979995)
		self.assertAlmostEqual(prob.model.get_val('f.y')[0], 0.)
		self.assertAlmostEqual(prob.model.get_val('f.eas')[0], 1.01015254)
		self.assertAlmostEqual(prob.model.get_val('f.n')[0], 1.02511097)
		self.assertAlmostEqual(prob.model.get_val('f.I_landing_gear')[0], 1.)

	def test_partials(self):

		# Load settings and aircraft
		settings = pyna.load_settings(case_name="nasa_stca_standard")
		settings.pyNA_directory = '.'
		ac = Aircraft(name=settings.ac_name, version=settings.ac_version, settings=settings)

		# Inputs
		nn = 20
		phase_name = 'vnrs'

		x = np.linspace(1,100,nn)
		z = np.linspace(1,100,nn)
		v = np.linspace(1,100,nn)
		F_n = 210000*np.ones(nn,)
		alpha = np.linspace(1,10,nn)
		L = 54000*9.81*np.ones(nn,)
		D = 3000*np.ones(nn,)
		gamma = np.linspace(1,20,nn)
		rho_0 = 1.25*np.ones(nn,)
		drho_0_dz = -0.05*np.ones(nn,)

		# Create problem
		prob = om.Problem()
		comp = FlightDynamics(num_nodes=nn, settings=settings, phase=phase_name, ac=ac, objective='noise')
		prob.model.add_subsystem("f", comp)
		prob.setup(force_alloc_complex=True)
		    
		prob.set_val('f.x', x)
		prob.set_val('f.z', z)
		prob.set_val('f.v', v)
		prob.set_val('f.F_n', F_n)
		prob.set_val('f.alpha', alpha)
		prob.set_val('f.L', L)
		prob.set_val('f.D', D)
		prob.set_val('f.gamma', gamma)
		prob.set_val('f.rho_0', rho_0)
		prob.set_val('f.drho_0_dz', drho_0_dz)

		# Check partials 
		prob.check_partials(compact_print=True, method='cs')


if __name__ == '__main__':
	unittest.main()
