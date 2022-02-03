import pandas as pd
import openmdao.api as om
import numpy as np
import os
os.environ["pyna_language"] = 'python'
os.chdir('../../')
from pyNA.pyna import pyna
from pyNA.src.settings import Settings
from pyNA.src.aircraft import Aircraft
from pyNA.src.trajectory_src.flight_dynamics import FlightDynamics


# Load settings and aircraft
settings = pyna.load_settings(case_name="nasa_stca_standard")
settings.pyNA_directory = '.'
ac = Aircraft(name=settings.ac_name, settings=settings)

# Inputs
nn = 20
phase_name = 'climb'

x = np.linspace(1,100,nn)
z = np.linspace(1,100,nn)
v = np.linspace(1,100,nn)
F_n = 210000*np.ones(nn,)
alpha = np.linspace(1,10,nn)
L = 54000*9.81*np.ones(nn,)
D = 3000*np.ones(nn,)
gamma = np.linspace(1,20,nn)
rho_0 = 1.25*np.ones(nn,)
drho_0_dz = -0.05*np.ones(nn,)

# Create problem
prob = om.Problem()
comp = FlightDynamics(num_nodes=nn, phase=phase_name, ac=ac)
prob.model.add_subsystem("f", comp)
prob.setup(force_alloc_complex=True)
    
prob.set_val('f.x', x)
prob.set_val('f.z', z)
prob.set_val('f.v', v)
prob.set_val('f.F_n', F_n)
prob.set_val('f.alpha', alpha)
prob.set_val('f.L', L)
prob.set_val('f.D', D)
prob.set_val('f.gamma', gamma)
prob.set_val('f.rho_0', rho_0)
prob.set_val('f.drho_0_dz', drho_0_dz)

# Run problem
prob.run_model()

# Check partials 
prob.check_partials(compact_print=True, method='cs')

