import unittest
import pandas as pd
import openmdao.api as om
import numpy as np
import os
os.environ["pyna_language"] = 'python'
from pyNA.pyna import pyna
from pyNA.src.airframe import Airframe
from pyNA.src.trajectory_src.flight_dynamics import FlightDynamics


class TestFlightDynamics(unittest.TestCase):

	def test_partials(self):

		py = pyna()
		py.ac_name = 'stca'
		py.case_name = 'nasa_stca_standard'
		py.atmosphere_type = 'sealevel'
		py.initialize()

		# Inputs
		nn = 20
		phase_name = 'cutback'

		x = np.linspace(1,100,nn)
		z = np.linspace(1,100,nn)
		v = np.linspace(1,100,nn)
		F_n = 210000*np.ones(nn,)
		alpha = np.linspace(1,10,nn)
		L = 54000*9.81*np.ones(nn,)
		D = 3000*np.ones(nn,)
		gamma = np.linspace(1,20,nn)

		# Create problem
		prob = om.Problem()
		comp = FlightDynamics(num_nodes=nn, phase=phase_name, airframe=py.airframe, sealevel_atmosphere=py.sealevel_atmosphere)
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

		# Run problem
		prob.run_model()

		# Check partials 
		prob.check_partials(compact_print=True, method='cs')


if __name__ == '__main__':
	unittest.main()
