import unittest
import jax
import jax.numpy as jnp
from pyNA.src.noise_model.python.propagation.calculate_direct_propagation import calculate_direct_propagation
import pdb


class TestDirectPropagation(unittest.TestCase):

	def test_evaluate(self):
		
		settings = dict()
		settings['r_0'] = 0.3048

		msap_source = jnp.ones(24,)
		r = 3000.
		I_0 = 390.
		msap_prop = compute_direct_propagation(msap_source, r, I_0, settings)
		self.assertTrue((msap_prop == 1.084504e-08).all())


	def test_jax(self):
		
		settings = dict()
		settings['r_0'] = 0.3048

		msap_source = jnp.ones(24,)
		r = 3000.
		I_0 = 390.

		# Get jacobian function
		f_prime_0 = jax.jacrev(compute_direct_propagation, argnums=0)
		F = f_prime_0(msap_source, r, I_0, settings)
		self.assertTrue(jnp.count_nonzero(F - jnp.diag(jnp.diagonal(F))) == 0.)
		self.assertTrue((jnp.diagonal(F)==1.084504e-8).all())

		# Get jacobian function
		f_prime_1 = jax.jacrev(compute_direct_propagation, argnums=1)
		G = f_prime_1(msap_source, r, I_0, settings)
		self.assertTrue((G==-7.2300278e-12).all())

		# Get jacobian function
		f_prime_2 = jax.jacrev(compute_direct_propagation, argnums=2)
		H = f_prime_2(msap_source, r, I_0, settings)
		self.assertTrue((H==-2.7807794e-11).all())


if __name__ == '__main__':
	unittest.main()
