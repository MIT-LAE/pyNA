import unittest
import pdb
import pandas as pd
import openmdao.api as om
import numpy as np
import os
os.environ["pyna_language"] = 'python'
from pyNA.pyna import pyna
from pyNA.src.airframe import Airframe
from pyNA.src.trajectory_src.clcd import CLCD


class TestCLCD(unittest.TestCase):

	def test_partials(self):
		
		py = pyna()
		py.ac_name = 'stca'
		py.case_name = 'stca'
		py.atmosphere_type = 'stratified'
		py.initialize()
		py.airframe.get_aerodynamics_deck()
		py.engine.get_performance_deck_variables(fan_inlet_source=False, fan_discharge_source=False, core_source=False, jet_mixing_source=False, jet_shock_source=False)
		py.engine.get_performance_deck()

		# Inputs
		nn = 20

		# Create problem
		prob = om.Problem()
		prob.model.add_subsystem("a", CLCD(vec_size=nn, extrapolate=True, method='cubic', airframe=py.airframe))
		prob.setup(force_alloc_complex=True)

		prob.set_val('a.alpha', np.ones(nn))

		# Check partials 
		prob.check_partials(compact_print=True, method='cs')


if __name__ == '__main__':
	unittest.main()
