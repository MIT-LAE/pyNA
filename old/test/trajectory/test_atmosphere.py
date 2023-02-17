import unittest
import pandas as pd
import openmdao.api as om
import numpy as np
import os
os.environ["pyna_language"] = 'python'
from pyNA.pyna import pyna
from pyNA.src.trajectory_src.atmosphere import Atmosphere


class TestAtmosphere(unittest.TestCase):

	def test_partials_stratified(self):

		py = pyna()
		py.ac_name = 'stca'
		py.case_name = 'stca'
		py.atmosphere_type = 'stratified'
		py.initialize()

		# Inputs
		nn = 20
		z = np.linspace(1,2000, nn)

		# Create problem
		prob = om.Problem()
		comp = Atmosphere(num_nodes=nn, sealevel_atmosphere=py.sealevel_atmosphere, atmosphere_dT=10.0169)

		prob.model.add_subsystem("a", comp)
		prob.setup(force_alloc_complex=True)

		prob.set_val('a.z', z)

		# Check partials 
		prob.check_partials(compact_print=True, method='cs')

if __name__ == '__main__':
	unittest.main()

