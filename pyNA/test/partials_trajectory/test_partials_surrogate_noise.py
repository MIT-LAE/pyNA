import openmdao.api as om
import numpy as np
import os
os.environ["pyna_language"] = 'python'
os.chdir('../../')
from pyNA.src.trajectory_src.surrogate_noise import SurrogateNoise

# Inputs
nn = 20
x_observer = np.array([5., 0., 1.])

x = np.linspace(0, 10, nn)
y = np.zeros(nn)
z = np.linspace(0,10, nn)
t_s = np.linspace(0, 100, nn)

# Create problem
prob = om.Problem()
comp = SurrogateNoise(num_nodes=nn, x_observer=x_observer)
prob.model.add_subsystem("m", comp)
prob.setup(force_alloc_complex=True)

prob.set_val('m.x', x)
prob.set_val('m.y', y)
prob.set_val('m.z', z)
prob.set_val('m.t_s', t_s)

# Run problem
prob.run_model()

print(prob.get_val('m.r_obs_int'))

# Check partials 
prob.check_partials(compact_print=True)

