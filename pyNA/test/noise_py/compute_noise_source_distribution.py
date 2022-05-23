import os
import pdb
import numpy as np
import openmdao.api as om
import pandas as pd
import matplotlib.pyplot as plt

os.environ["pyna_language"] = 'python'
from pyNA.pyna import pyna

# Load default pyna settings
pyna_settings = pyna.load_settings(case_name = 'nasa_stca_standard')
pyna_settings.save_results = True

# Run pyna
py = pyna(settings=pyna_settings)
py.settings.validation=False

# Plot results
py.plot_noise_source_distribution(time_step=152, metric='spl', components=['core', 'jet_mixing', 'airframe', 'fan_inlet', 'fan_discharge'])