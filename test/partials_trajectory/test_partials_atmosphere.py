import pandas as pd
import openmdao.api as om
import numpy as np
import os
os.environ["pyna_language"] = 'python'
os.chdir('../../')
from pyNA.pyna import pyna
from pyNA.src.trajectory_src.atmosphere import Atmosphere


# Load settings and aircraft
settings = pyna.load_settings(case_name="NASA STCA Standard")

# Inputs
nn = 20
z = np.linspace(1,2000, nn)

# Create problem
prob = om.Problem()
comp = Atmosphere(num_nodes=nn, settings=settings)
prob.model.add_subsystem("a", comp)
prob.setup(force_alloc_complex=True)

prob.set_val('a.z', z)

# Run problem
prob.run_model()

# Check partials 
prob.check_partials(compact_print=True, method='cs')

