# Imports 
import os
import sys
sys.path.append("..")
os.environ["pyna_language"] = 'python'
import pdb
import numpy as np
import openmdao.api as om
import pandas as pd
import matplotlib.pyplot as plt
from pyNA.pyna import pyna

# Load verification trajectory
nasa_std = pd.read_csv('../../cases/nasa_stca_standard/trajectory/Trajectory_to.csv')

# Run pyna
pyna_settings = pyna.load_settings(case_name='stca')
pyna_settings.engine_file_name = 'engine_deck_stca.csv'
pyna_settings.TS_to = 0.88
pyna_settings.TS_vnrs = 0.88
pyna_settings.TS_cutback = 0.61
pyna_settings.z_cutback = nasa_std['Z [m]'][np.where(nasa_std['TS [-]'] < 0.65)[0][0]]

pyna_settings.max_iter = 50

py = pyna(settings=pyna_settings)
py.ac.z_max = nasa_std['Z [m]'].values[-1]
py.ac.v_max = nasa_std['V [m/s]'].values[-1]
py.ac.k_rot = 1.27

py.compute_trajectory_noise()

# Plot trajectory
py.plot_trajectory(py.problem, nasa_std)

# Plot noise time series
py.plot_noise_time_series(metric='pnlt')