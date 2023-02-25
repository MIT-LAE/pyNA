import unittest
import openmdao.api as om
import numpy as np
from pyNA.src.trajectory_model.atmosphere import Atmosphere
from openmdao.utils.assert_utils import assert_check_partials


class TestAtmosphere(unittest.TestCase):

	def test_evaluate_stratified(self):

		nn = 1
		z = np.array([1000.])

		prob = om.Problem()
		prob.model.add_subsystem("a", Atmosphere(num_nodes=nn, atmosphere_dT=10.0169, mode='stratified'))
		prob.setup(force_alloc_complex=True)
		prob.set_val('a.z', z)
		prob.run_model()

		self.assertAlmostEqual(prob.get_val('a.I_0')[0], 367.516, 2)
		self.assertAlmostEqual(prob.get_val('a.T_0')[0], 291.667, 2)
		self.assertAlmostEqual(prob.get_val('a.c_0')[0], 342.363, 2)
		self.assertAlmostEqual(prob.get_val('a.drho_0_dz')[0], -0.000106287, 2)
		self.assertAlmostEqual(prob.get_val('a.mu_0')[0], 0.0000180632, 2)
		self.assertAlmostEqual(prob.get_val('a.p_0')[0], 89874.46, 2)
		self.assertAlmostEqual(prob.get_val('a.rh')[0], 57.5328, 2)
		self.assertAlmostEqual(prob.get_val('a.rho_0')[0], 1.07347, 2)

	def test_evaluate_sealevel(self):

		nn = 1
		z = np.array([1000.])

		prob = om.Problem()
		prob.model.add_subsystem("a", Atmosphere(num_nodes=nn, atmosphere_dT=10.0169, mode='sealevel'))
		prob.setup(force_alloc_complex=True)
		prob.set_val('a.z', z)
		prob.run_model()

		print(prob.get_val('a.I_0'))
		print(prob.get_val('a.T_0'))
		print(prob.get_val('a.c_0'))
		print(prob.get_val('a.drho_0_dz'))
		print(prob.get_val('a.mu_0'))
		print(prob.get_val('a.p_0'))
		print(prob.get_val('a.rh'))
		print(prob.get_val('a.rho_0'))

		self.assertAlmostEqual(prob.get_val('a.I_0')[0], 409.798, 2)
		self.assertAlmostEqual(prob.get_val('a.T_0')[0], 298.167, 2)
		self.assertAlmostEqual(prob.get_val('a.c_0')[0], 346.157, 2)
		self.assertAlmostEqual(prob.get_val('a.drho_0_dz')[0], 0., 2)
		self.assertAlmostEqual(prob.get_val('a.mu_0')[0], 0.0000183733, 2)
		self.assertAlmostEqual(prob.get_val('a.p_0')[0], 101325., 2)
		self.assertAlmostEqual(prob.get_val('a.rh')[0], 70., 2)
		self.assertAlmostEqual(prob.get_val('a.rho_0')[0], 1.18386, 2)

	def test_partials_stratified(self):

		nn = 20
		z = np.linspace(1,2000, nn)

		prob = om.Problem()
		prob.model.add_subsystem("a", Atmosphere(num_nodes=nn, atmosphere_dT=10.0169, mode='stratified'))
		prob.setup(force_alloc_complex=True)
		prob.set_val('a.z', z)

		data = prob.check_partials(compact_print=True, method='cs')		
		assert_check_partials(data, atol=1e-6, rtol=1e-6)

	def test_partials_sealevel(self):

		nn = 20
		z = np.linspace(1,2000, nn)

		prob = om.Problem()
		prob.model.add_subsystem("a", Atmosphere(num_nodes=nn, atmosphere_dT=10.0169, mode='sealevel'))
		prob.setup(force_alloc_complex=True)
		prob.set_val('a.z', z)

		data = prob.check_partials(compact_print=True, method='cs')		
		assert_check_partials(data, atol=1e-6, rtol=1e-6)

if __name__ == '__main__':
	unittest.main()

