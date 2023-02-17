import unittest
import numpy as np
import openmdao.api as om
import pandas as pd
import pdb
import sys
import os
os.environ["pyna_language"] = 'python'
from pyNA.src.trajectory_src.take_off_phase_ode import TakeOffPhaseODE
from pyNA.pyna import pyna


class TestTakeOffPhaseODE(unittest.TestCase):

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
		nn = 10
		mode = 'noise'

		# Create problem
		phase = 'liftoff'
		prob = om.Problem()
		comp = TakeOffPhaseODE(num_nodes=nn, phase=phase, airframe=py.airframe, engine=py.engine, sealevel_atmosphere=py.sealevel_atmosphere, atmosphere_dT=py.atmosphere_dT, atmosphere_type=py.atmosphere_type)
		prob.model.add_subsystem('t', comp)

		prob.setup(force_alloc_complex=True)

		x = np.linspace(1, 100, nn)
		z = np.linspace(1, 100, nn)
		v = np.linspace(1, 100, nn)
		alpha = np.linspace(1, 10, nn)
		gamma = np.linspace(1, 15, nn)
		TS = np.linspace(0.8, 1., nn)
		theta_flaps = np.linspace(1, 10., nn)
		theta_slats = np.linspace(1, 10., nn)

		prob.set_val('t.x', x)
		prob.set_val('t.z', z)
		prob.set_val('t.v', v)
		prob.set_val('t.alpha', alpha)
		prob.set_val('t.gamma', gamma)
		prob.set_val('t.propulsion.TS', TS)
		prob.set_val('t.theta_flaps', theta_flaps)
		prob.set_val('t.theta_slats', theta_slats)

		prob.run_model()

		# Check partials
		prob.check_partials(method='cs', compact_print=True)


if __name__ == '__main__':
	unittest.main()





