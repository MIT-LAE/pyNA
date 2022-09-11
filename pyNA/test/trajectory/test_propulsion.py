import unittest
import numpy as np
import openmdao.api as om
import pandas as pd
import pdb
import sys
import os
os.environ["pyna_language"] = 'python'
from pyNA.src.trajectory_src.propulsion import Propulsion
from pyNA.pyna import pyna


class TestPropulsion(unittest.TestCase):

	def test_partials(self):

		py = pyna()
		py.ac_name = 'stca'
		py.case_name = 'stca'
		py.atmosphere_type = 'stratified'
		py.initialize()
		py.engine.get_performance_deck_variables(fan_inlet_source=False, fan_discharge_source=False, core_source=False, jet_mixing_source=False, jet_shock_source=False)
		py.engine.get_performance_deck()

		nn = 10
		M_0 = np.linspace(0, 0.4, nn)
		z = np.linspace(0, 3000, nn)
		TS = np.linspace(0.7, 1, nn)

		# Create problem
		prob = om.Problem()
		comp = Propulsion(vec_size=nn, engine=py.engine)
		prob.model.add_subsystem('p', comp)
		prob.setup(force_alloc_complex=True)

		prob.set_val('p.M_0', M_0)
		prob.set_val('p.z', z)
		prob.set_val('p.TS', TS)

		# Check partials
		prob.check_partials(compact_print=True, method='fd')


if __name__ == '__main__':
	unittest.main()



