import unittest
import openmdao.api as om
import numpy as np
from pyNA.pyna import pyna
from pyNA.src.trajectory_model.sst.flight_dynamics import FlightDynamics
from openmdao.utils.assert_utils import assert_check_partials


class TestFlightDynamics(unittest.TestCase):

	def test_evaluate(self):

		py = pyna(trajectory_mode='model', 
				  case_name = 'stca',
				  engine_name='engine_derivative')
		nn = 1
		phase_name = 'cutback'
		objective = 'time'

		x = np.linspace(1,100,nn)
		z = np.linspace(1,100,nn)
		v = np.linspace(1,100,nn)
		alpha = np.linspace(1,10,nn)
		gamma = np.linspace(1,20,nn)
		F_n = 210000*np.ones(nn,)
		c_0 = 300*np.ones(nn,)
		rho_0 = 1.2*np.ones(nn,)
		c_l = 0.5*np.ones(nn,)
		c_d = 0.02*np.ones(nn,)
		c_l_max = np.ones(nn,)

		# Create problem
		prob = om.Problem()
		comp = FlightDynamics(num_nodes=nn, phase=phase_name, settings=py.settings, aircraft=py.aircraft, objective=objective)
		prob.model.add_subsystem("f", comp)
		prob.setup(force_alloc_complex=True)
		    
		prob.set_val('f.x', x)
		prob.set_val('f.z', z)
		prob.set_val('f.v', v)
		prob.set_val('f.alpha', alpha)
		prob.set_val('f.gamma', gamma)
		prob.set_val('f.F_n', F_n)
		prob.set_val('f.c_0', c_0)
		prob.set_val('f.rho_0', rho_0)
		prob.set_val('f.c_l', c_l)
		prob.set_val('f.c_d', c_d)
		prob.set_val('f.c_l_max', c_l_max)
		
		prob.run_model()
	
		self.assertAlmostEqual(prob.get_val('f.x_dot')[0], 0.999848, 3)
		self.assertAlmostEqual(prob.get_val('f.z_dot')[0], 0.0174524, 3)
		self.assertAlmostEqual(prob.get_val('f.v_dot')[0], 11.2757, 3)
		self.assertAlmostEqual(prob.get_val('f.alpha_dot')[0], 0., 3)
		self.assertAlmostEqual(prob.get_val('f.gamma_dot')[0], -537.698, 2)
		self.assertAlmostEqual(prob.get_val('f.M_0')[0], 0.00333333, 3)
		self.assertAlmostEqual(prob.get_val('f.n')[0], 0.0428915, 3)

	def test_partials(self):

		py = pyna(trajectory_mode='model', 
				  case_name = 'stca',
				  engine_name='engine_derivative')
		nn = 20
		objective = 'time'
		
		for phase_name in ['groundroll', 'rotation', 'liftoff', 'vnrs','cutback']:
	
			for constant_LD in [None, 7.]:
			
				x = np.linspace(1,100,nn)
				z = np.linspace(1,100,nn)
				v = np.linspace(1,100,nn)
				alpha = np.linspace(1,10,nn)
				gamma = np.linspace(1,20,nn)
				F_n = 210000*np.ones(nn,)
				c_0 = 300*np.ones(nn,)
				rho_0 = 1.2*np.ones(nn,)
				c_l = 0.5*np.ones(nn,)
				c_d = 0.02*np.ones(nn,)
				c_l_max = np.ones(nn,)

				# Create problem
				prob = om.Problem()
				comp = FlightDynamics(num_nodes=nn, phase=phase_name, settings=py.settings, aircraft=py.aircraft, objective=objective, constant_LD=constant_LD)
				prob.model.add_subsystem("f", comp)
				prob.setup(force_alloc_complex=True)
					
				prob.set_val('f.x', x)
				prob.set_val('f.z', z)
				prob.set_val('f.v', v)
				prob.set_val('f.alpha', alpha)
				prob.set_val('f.gamma', gamma)
				prob.set_val('f.F_n', F_n)
				prob.set_val('f.c_0', c_0)
				prob.set_val('f.rho_0', rho_0)
				prob.set_val('f.c_l', c_l)
				prob.set_val('f.c_d', c_d)
				prob.set_val('f.c_l_max', c_l_max)
				
				prob.run_model()

				# Check partials 
				data = prob.check_partials(compact_print=True, method='cs')
				assert_check_partials(data, atol=1e-6, rtol=1e-6)


if __name__ == '__main__':
	unittest.main()
