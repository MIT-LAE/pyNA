import unittest
import numpy as np
import openmdao.api as om
import pandas as pd
import pdb
import os
os.environ['pyna_language'] = 'python'
import sys
from pyNA.src.trajectory_src.mux import Mux


class TestMux(unittest.TestCase):

	def test_partials(self):
		time_0 = np.array([ 0,1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
		time_1 = np.array([ 0,1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + 10.
		time_2 = np.array([ 0,1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + 20.

		# Create problem
		prob = om.Problem()
		model = om.Group()
		mux_t = prob.model.add_subsystem(name='m', subsys=Mux(input_size_array=[11, 11, 11], output_size=31))

		mux_t.add_var('t', units='s')

		prob.setup(force_alloc_complex=True)

		prob.set_val('m.t_0', time_0)
		prob.set_val('m.t_1', time_1)
		prob.set_val('m.t_2', time_2)

		# Check partials
		prob.check_partials(method='cs', compact_print=True)


if __name__ == '__main__':
	unittest.main()


