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
from pyNA.src.trajectory_src.clcd import CLCD


# Load settings and aircraft
settings = pyna.load_settings(case_name="NASA STCA Standard")
settings.pyNA_directory = '/Users/laurensvoet/Documents/Research/pyNA/pyNA'

py = pyna(pyna_settings)
py.ac.load_aerodynamics(settings=settings)
py.engine.load_deck(settings=py.settings)

# Inputs
nn = 20

# Create problem
prob = om.Problem()
comp = CLCD(vec_size=nn, extrapolate=True, method='cubic', ac=py.ac)
prob.model.add_subsystem("a", comp)
prob.setup(force_alloc_complex=True)

prob.set_val('a.alpha', np.ones(nn))

# Run problem
prob.run_model()

# Check partials 
prob.check_partials(compact_print=True, method='cs')

