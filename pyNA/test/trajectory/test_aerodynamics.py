import unittest
import pdb
import pandas as pd
import openmdao.api as om
import numpy as np
import os
os.environ["pyna_language"] = 'python'
from pyNA.pyna import pyna
from pyNA.src.trajectory_src.aerodynamics import Aerodynamics


class TestAerodynamics(unittest.TestCase):

	def test_partials(self):
		
		py = pyna()
		py.ac_name = 'stca'
		py.case_name = 'nasa_stca_standard'
		py.atmosphere_type = 'sealevel'
		py.initialize()

		# Inputs
		nn = 1
		rho_0 = 1.225*np.ones(nn)
		v = 100.*np.ones(nn)
		c_0 = 330.*np.ones(nn)

		# Create problem
		prob = om.Problem()
		comp = Aerodynamics(num_nodes=nn, airframe=py.airframe, phase='vnrs', atmosphere_type=py.atmosphere_type, sealevel_atmosphere=py.sealevel_atmosphere)
		prob.model.add_subsystem("a", comp)
		prob.setup(force_alloc_complex=True)

		prob.set_val('a.v', v)
		if py.atmosphere_type == 'stratified':
			prob.set_val('a.rho_0', rho_0)
			prob.set_val('a.c_0', c_0)

		# Run model
		prob.run_model()

		# Check partials 
		prob.check_partials(compact_print=True, method='cs')


if __name__ == '__main__':
	unittest.main()
