import unittest
import jax
import jax.numpy as jnp
from pyNA.src.noise_model.python.calculate_geometry import calculate_geometry
import pdb


class TestComputeSPL(unittest.TestCase):

	def test_evaluate(self):
		
		x = 1000.
		y = 0.
		z = 100.
		alpha = 10.
		gamma = 15.
		t_s = 100.
		c_0 = 340.
		T_0 = 300.
		x_mic = jnp.array([3500., 450., 1.2])

		ans = calculate_geometry(x, y, z, alpha, gamma, t_s, c_0, T_0, x_mic)

		self.assertAlmostEqual(ans[0], 2542.2566, 2) 			# r
		self.assertAlmostEqual(ans[1], 29.054958785790166, 2) 	# theta
		self.assertAlmostEqual(ans[2], -21.375457790601132, 2) 	# phi
		self.assertAlmostEqual(ans[3], 2.3174735804318063, 2)	# beta
		self.assertAlmostEqual(ans[4], 107.33259848419225, 2)	# t_o
		self.assertAlmostEqual(ans[5], 346.706095700933, 2)		# c_bar


	def test_jax(self):
		
		x = 1000.
		y = 0.
		z = 100.
		alpha = 10.
		gamma = 15.
		t_s = 100.
		c_0 = 340.
		T_0 = 300.
		x_mic = jnp.array([3500., 450., 1.2])
	
		# Get jacobian function	
		f_prime_0 = jax.jacrev(calculate_geometry, argnums=0)
		F = f_prime_0(x, y, z, alpha, gamma, t_s, c_0, T_0, x_mic)
		self.assertAlmostEqual(F[0], -0.98337835, 3)
		self.assertAlmostEqual(F[1], 0.00216642, 3)
		self.assertAlmostEqual(F[2], -0.00714827, 3)
		self.assertAlmostEqual(F[3], 0.00089692, 3)
		self.assertAlmostEqual(F[4], -0.00283635, 3)
		self.assertAlmostEqual(F[5], 0., 3)

		f_prime_1 = jax.jacrev(calculate_geometry, argnums=1)
		G = f_prime_1(x, y, z, alpha, gamma, t_s, c_0, T_0, x_mic)
		self.assertAlmostEqual(G[0], -0.1770081, 3)
		self.assertAlmostEqual(G[1], -0.00718063, 3)
		self.assertAlmostEqual(G[2], 0.04321454, 3)
		self.assertAlmostEqual(G[3], 0.00016145, 3)
		self.assertAlmostEqual(G[4], -0.00051054, 3)
		self.assertAlmostEqual(G[5], 0., 3)

		f_prime_2 = jax.jacrev(calculate_geometry, argnums=2)
		H = f_prime_2(x, y, z, alpha, gamma, t_s, c_0, T_0, x_mic)
		self.assertAlmostEqual(H[0], 0.04043652, 3)
		self.assertAlmostEqual(H[1], 0.02125272, 3)
		self.assertAlmostEqual(H[2], 0.01532951, 3)
		self.assertAlmostEqual(H[3], 0.02251894, 3)
		self.assertAlmostEqual(H[4], 8.049769e-05, 3)
		self.assertAlmostEqual(H[5], 0.00170846, 3)

		f_prime_3 = jax.jacrev(calculate_geometry, argnums=3)
		I = f_prime_3(x, y, z, alpha, gamma, t_s, c_0, T_0, x_mic)
		self.assertAlmostEqual(I[0], 0., 3)
		self.assertAlmostEqual(I[1], 0.9312118, 3)
		self.assertAlmostEqual(I[2], 0.6560506, 3)
		self.assertAlmostEqual(I[3], 0., 3)
		self.assertAlmostEqual(I[4], 0., 3)
		self.assertAlmostEqual(I[5], 0., 3)

		f_prime_4 = jax.jacrev(calculate_geometry, argnums=4)
		J = f_prime_4(x, y, z, alpha, gamma, t_s, c_0, T_0, x_mic)
		self.assertAlmostEqual(J[0], 0., 3)
		self.assertAlmostEqual(J[1], 0.9312118, 3)
		self.assertAlmostEqual(J[2], 0.6560506, 3)
		self.assertAlmostEqual(J[3], 0., 3)
		self.assertAlmostEqual(J[4], 0., 3)
		self.assertAlmostEqual(J[5], 0., 3)

		f_prime_5 = jax.jacrev(calculate_geometry, argnums=5)
		K = f_prime_5(x, y, z, alpha, gamma, t_s, c_0, T_0, x_mic)
		self.assertAlmostEqual(K[0], 0., 3)
		self.assertAlmostEqual(K[1], 0., 3)
		self.assertAlmostEqual(K[2], 0., 3)
		self.assertAlmostEqual(K[3], 0., 3)
		self.assertAlmostEqual(K[4], 1., 3)
		self.assertAlmostEqual(K[5], 0., 3)

		f_prime_6 = jax.jacrev(calculate_geometry, argnums=6)
		L = f_prime_6(x, y, z, alpha, gamma, t_s, c_0, T_0, x_mic)
		self.assertAlmostEqual(L[0], 0., 3)
		self.assertAlmostEqual(L[1], 0., 3)
		self.assertAlmostEqual(L[2], 0., 3)
		self.assertAlmostEqual(L[3], 0., 3)
		self.assertAlmostEqual(L[4], -0.00192267, 3)
		self.assertAlmostEqual(L[5], 0.09090909, 3)

		f_prime_7 = jax.jacrev(calculate_geometry, argnums=7)
		M = f_prime_7(x, y, z, alpha, gamma, t_s, c_0, T_0, x_mic)
		self.assertAlmostEqual(M[0], 0., 3)
		self.assertAlmostEqual(M[1], 0., 3)
		self.assertAlmostEqual(M[2], 0., 3)
		self.assertAlmostEqual(M[3], 0., 3)
		self.assertAlmostEqual(M[4], -0.01111944, 3)
		self.assertAlmostEqual(M[5], 0.52575886, 3)

if __name__ == '__main__':
	unittest.main()
