import unittest
import numpy as np
import openmdao.api as om
import pdb
from pyNA.src.trajectory_model.mux import Mux
from openmdao.utils.assert_utils import assert_check_partials


class TestMux(unittest.TestCase):

	def test_evaluate(self):
		time_0 = np.array([ 0, 1, 2, 3, 4, 5, ])
		time_1 = np.array([ 0, 1, 2, 3, 4, 5, ]) + 5.
		time_2 = np.array([ 0, 1, 2, 3, 4, 5, ]) + 10.

		# Create problem
		prob = om.Problem()
		mux_t = prob.model.add_subsystem(name='m', subsys=Mux(input_size_array=[6, 6, 6], output_size=16))
		mux_t.add_var('t', units='s')
		prob.setup(force_alloc_complex=True)
		prob.set_val('m.t_0', time_0)
		prob.set_val('m.t_1', time_1)
		prob.set_val('m.t_2', time_2)
		prob.run_model()

		self.assertTrue((prob.get_val('m.t') == np.linspace(0, 15, 16)).all()) 
		
	def test_partials(self):
		time_0 = np.array([ 0, 1, 2, 3, 4, 5, ])
		time_1 = np.array([ 0, 1, 2, 3, 4, 5, ]) + 5.
		time_2 = np.array([ 0, 1, 2, 3, 4, 5, ]) + 10.

		prob = om.Problem()
		mux_t = prob.model.add_subsystem(name='m', subsys=Mux(input_size_array=[6, 6, 6], output_size=16))
		mux_t.add_var('t', units='s')
		prob.setup(force_alloc_complex=True)
		prob.set_val('m.t_0', time_0)
		prob.set_val('m.t_1', time_1)
		prob.set_val('m.t_2', time_2)

		data = prob.check_partials(method='cs', compact_print=True)
		assert_check_partials(data, atol=1e-6, rtol=1e-6)

if __name__ == '__main__':
	unittest.main()


