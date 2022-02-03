import pdb
import pandas as pd
import openmdao.api as om
import numpy as np
import os
os.environ["pyna_language"] = 'python'
os.chdir('../../')
from pyNA.pyna import pyna
from pyNA.src.settings import Settings
from pyNA.src.aircraft import Aircraft
from pyNA.src.trajectory_src.aerodynamics import Aerodynamics


# Load settings and aircraft
settings = pyna.load_settings(case_name="NASA STCA Standard")
settings.pyNA_directory = 'pyNA'
ac = Aircraft(name=settings.ac_name, settings=settings)

# Inputs
nn = 20
rho_0 = 1.225*np.ones(nn)
v = 100.*np.ones(nn)
c_0 = 330.*np.ones(nn)

# Create problem
prob = om.Problem()
comp = Aerodynamics(num_nodes=nn, ac=ac)
prob.model.add_subsystem("a", comp)
prob.setup(force_alloc_complex=True)

prob.set_val('a.rho_0', rho_0)
prob.set_val('a.v', v)
prob.set_val('a.c_0', c_0)

# Run problem
prob.run_model()

# Check partials 
prob.check_partials(compact_print=True, method='cs')