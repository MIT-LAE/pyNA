import numpy as np
import openmdao.api as om
import pandas as pd
import pdb
import os
os.environ['pyna_language'] = 'python'
import sys
sys.path.append("../..")
from pyNA.src.trajectory_src.mux import Mux

time_0 = np.array([ 0,1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12,13,14,15])
time_1 = np.array([ 0,1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12,13,14,15]) + 15.
time_2 = np.array([ 0,1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12,13,14,15]) + 30.

# Create problem
prob = om.Problem()
model = om.Group()
mux_t = prob.model.add_subsystem(name='m', subsys=Mux(size_inputs=[16, 16, 16], size_output=46))

mux_t.add_var('t', units='s')

prob.setup(force_alloc_complex=True)

prob.set_val('m.t_0', time_0)
prob.set_val('m.t_1', time_1)
prob.set_val('m.t_2', time_2)

# Run model
prob.run_model()

print(prob.get_val('m.t'))

# Check partials
prob.check_partials(method='cs', compact_print=True)


