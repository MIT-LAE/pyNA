from jax import random
import jax.numpy as jnp
from scipy.interpolate import RectBivariateSpline

from pyNA.src.noise_model.python.utils.jax_interp2d import interp2d


def f(x, y):
    return jnp.cos(x) * jnp.sin(x * y)

xp = jnp.linspace(-3, 3, 150)
yp = jnp.linspace(-3, 3, 200)
Xp, Yp = jnp.meshgrid(xp, yp, indexing="ij")
zp = f(Xp, Yp)

key_x, key_y = random.split(random.PRNGKey(890))
x = random.uniform(key_x, shape=(10000,)) * (xp.max() - xp.min()) + xp.min()
y = random.uniform(key_y, shape=(10000,)) * (yp.max() - yp.min()) + yp.min()

# Interpolate!
z_interp = interp2d(x, y, xp, yp, zp)
z_scipy_interp = RectBivariateSpline(xp, yp, zp, kx=1, ky=1)(x, y, grid=False)

assert jnp.allclose(z_interp, z_scipy_interp), "Interpolation does not match scipy"