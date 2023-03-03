import unittest
import jax
import jax.numpy as jnp
from pyNA.src.noise_model.python.level.compute_spl import compute_spl
import pdb


class TestGeometry(unittest.TestCase):

	def test_evaluate(self):
		
		nn = 11
		msap_prop = jnp.logspace(-5, 1, nn)
		c_0 = 300.
		rho_0 = 1.225

		spl = compute_spl(msap_prop, rho_0, c_0)		
		self.assertTrue((spl == jnp.array([ 50.84757, 56.84757, 62.847565, 68.847565, 74.847565, 80.847565, 86.847565, 92.847565, 98.847565, 104.847565, 110.847565])).any())

	def test_clip(self):

		msap_prop = jnp.array([1e-15, ])
		c_0 = 300.
		rho_0 = 1.225

		spl = compute_spl(msap_prop, rho_0, c_0)		
		self.assertTrue((spl == jnp.array([0., ])).any())

	def test_jax(self):
		
		nn = 3
		msap_prop = jnp.logspace(-5, 1, nn)
		c_0 = 300.
		rho_0 = 1.225

		# Get jacobian function
		f_prime_0 = jax.jacrev(compute_spl, argnums=0)
		J = f_prime_0(msap_prop, rho_0, c_0)
		self.assertTrue((J == jnp.diag(10./msap_prop/jnp.log(10))).any())

		f_prime_1 = jax.jacrev(compute_spl, argnums=1)
		G = f_prime_1(msap_prop, rho_0, c_0)
		self.assertTrue(sum(abs(G - 20./rho_0/jnp.log(10)*jnp.ones(msap_prop.shape))) < 1e-6)

		f_prime_2 = jax.jacrev(compute_spl, argnums=2)
		H = f_prime_2(msap_prop, rho_0, c_0)
		self.assertTrue(sum(abs(H - 40./c_0/jnp.log(10)*jnp.ones(msap_prop.shape))) < 1e-6)

if __name__ == '__main__':
	unittest.main()
