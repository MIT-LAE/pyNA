import unittest
import numpy as np
import openmdao.api as om
import pandas as pd
import pdb
import sys
sys.path.append("../..")
import os
os.environ["pyna_language"] = 'python'
from pyNA.src.trajectory_src.propulsion import Propulsion
from pyNA.pyna import pyna


class TestPropulsion(unittest.TestCase):

	def test_evaluate(self):
		
		pyna_settings = pyna.load_settings(case_name="stca")
		pyna_settings.engine_file_name = 'engine_deck_stca.csv'
		pyna_settings.pyNA_directory = 'pyNA'

		py = pyna(pyna_settings)
		py.engine.load_deck(settings=py.settings)

		nn = 1
		M_0 = np.linspace(0, 0.4, nn)
		z = np.linspace(0, 3000, nn)
		TS = np.linspace(0.7, 1, nn)

		# Create problem
		prob = om.Problem()
		comp = Propulsion(vec_size=nn, settings=py.settings, engine=py.engine)
		prob.model.add_subsystem('p', comp)
		prob.setup(force_alloc_complex=True)

		prob.set_val('p.M_0', M_0)
		prob.set_val('p.z', z)
		prob.set_val('p.TS', TS)
		
		# Run model
		prob.run_model()
		self.assertAlmostEqual(prob.get_val('p.W_f')[0], 0.7333)
		self.assertAlmostEqual(prob.get_val('p.F_n')[0], 58678.1722)
		self.assertAlmostEqual(prob.get_val('p.Tti_c')[0], 660.6229599999997)
		self.assertAlmostEqual(prob.get_val('p.Pti_c')[0], 1495530.4160699998)


	def test_partials(self):

		pyna_settings = pyna.load_settings(case_name="stca")
		pyna_settings.engine_file_name = 'engine_deck_stca.csv'
		pyna_settings.pyNA_directory = 'pyNA'

		py = pyna(pyna_settings)
		py.engine.load_deck(settings=py.settings)

		nn = 10
		M_0 = np.linspace(0, 0.4, nn)
		z = np.linspace(0, 3000, nn)
		TS = np.linspace(0.7, 1, nn)

		# Create problem
		prob = om.Problem()
		comp = Propulsion(vec_size=nn, settings=py.settings, engine=py.engine)
		prob.model.add_subsystem('p', comp)
		prob.setup(force_alloc_complex=True)

		prob.set_val('p.M_0', M_0)
		prob.set_val('p.z', z)
		prob.set_val('p.TS', TS)

		# Check partials
		prob.check_partials(compact_print=True, method='fd')


if __name__ == '__main__':
	unittest.main()



