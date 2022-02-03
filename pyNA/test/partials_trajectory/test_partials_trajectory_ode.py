import numpy as np
import openmdao.api as om
import pandas as pd
import pdb
import sys
sys.path.append("../..")
import os
os.environ["pyna_language"] = 'python'
from pyNA.src.trajectory_src.trajectory_ode import TrajectoryODE
from pyNA.pyna import pyna

# Is not used for NASA STCA Standard!
pyna_settings = pyna.load_settings(case_name="stca")
pyna_settings.pyNA_directory = 'pyNA'
pyna_settings.engine_file_name = 'engine_deck_stca.csv'

py = pyna(pyna_settings)
py.ac.load_aerodynamics(settings=py.settings)
py.engine.load_deck(settings=py.settings)

# Inputs
nn = 10
mode = 'noise'

# Create problem
prob = om.Problem()
comp = TrajectoryODE(num_nodes=nn, phase='climb', ac=py.ac, engine=py.engine, settings=py.settings, mode=mode)
prob.model.add_subsystem('t', comp)
prob.setup(force_alloc_complex=True)

# Run model
prob.run_model()

# Check partials
prob.check_partials(method='cs', compact_print=True)





