import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import openmdao.api as om
import datetime as dt
import os
import pdb
import pyNA

from pyNA.src.take_off_model import TakeOffModel
from pyNA.src.time_history import TimeHistory


class Trajectory:
    
    def __init__(self, mode:str) -> None:
        
        """
        
        :param mode: use 'data' to input the time history of the trajectory variables and engine operational variables using .csv files; use 'compute' to model the trajectory and engine using Dymos
        :type mode: str

        """

        self.mode = mode
        if self.mode == 'model':
            self.path = TakeOffModel()

        elif self.mode == 'data':
            # Load data .csv files
            self.data = pd.DataFrame()

            # Create openmdao problem
            self.path = TimeHistory()

        self.vars = list()
        self.var_units = dict()

    def load_data(self, settings) -> None:
        
        """
        Load engine and trajectory time history from .csv file.

        :param settings:
        :type settings:

        :return: None
        """

        # Load raw inputs from .csv file
        # Source: validation noise assessment data set of NASA STCA (Berton et al., 2019)
        try:
            self.data = pd.read_csv(pyNA.__path__.__dict__["_path"][0] + '/cases/' + settings['case_name'] + '/trajectory/' + settings['output_directory_name'] + '/' + settings['time_history_file_name'])
        except:
            raise ValueError(settings['time_history_file_name'] + ' file not found at ' + pyNA.__path__.__dict__["_path"][0] + '/cases/' + settings['case_name'] + '/trajectory/' + settings['output_directory_name'] + '/')
        
        # Compute number of time steps in the time history
        self.n_t = np.size(self.data['t_s [s]'])

        return None

    def solve(self, settings, aircraft=None, trajectory_mode='cutback', objective='time',
                    tau={'groundroll':1., 'rotation':1., 'liftoff':1., 'vnrs':1., 'cutback':0.6},
                    theta_flaps={'groundroll':10., 'rotation':10., 'liftoff':10., 'vnrs':10., 'cutback':10.},
                    theta_slats={'groundroll':-6., 'rotation':-6., 'liftoff':-6., 'vnrs':-6., 'cutback':-6.},
                    k_rot=1.2) -> None:
        
        """
        """
        
        if self.mode == 'data':

            Trajectory.load_data(self, settings=settings)

            self.path.create(settings=settings, num_nodes=self.n_t)
            self.path.setup(force_alloc_complex=True)
            self.path.set_initial_conditions(settings, data=self.data)
            self.path.run_model()

        elif self.mode == 'model':

            if not aircraft:
                raise ValueError("'aircraft' not defined")

            self.path.set_controls(tau=tau, theta_flaps=theta_flaps, theta_slats=theta_slats, k_rot=k_rot)
            self.path.create(settings, aircraft, trajectory_mode, objective)
            self.path.set_objective(objective)
            self.path.set_driver_settings(settings, objective)
            self.path.setup(force_alloc_complex=True)
            self.path.set_initial_conditions(settings=settings, aircraft=aircraft, trajectory_mode=trajectory_mode)
            self.path.solve(settings, run_driver=True)

            # Check convergence
            converged = Trajectory.check_convergence(self, settings=settings, filename='IPOPT_trajectory_convergence.out')
            print('Converged:', converged)
  
        return None

    def check_convergence(self, settings, filename: str) -> bool:
        """
        Checks convergence of case using optimizer output file.

        :param settings: pyna settings
        :type settings: Settings
        :param filename: file name of IPOPT output
        :type filename: str

        :return: converged
        :rtype: bool
        """

        # Save convergence info for trajectory
        # Read IPOPT file
        file_ipopt = open(pyNA.__path__.__dict__["_path"][0] + '/cases/' + settings['case_name'] + '/output/' + settings['output_directory_name'] + '/' + filename, 'r')
        ipopt = file_ipopt.readlines()
        file_ipopt.close()

        # Check if convergence summary excel file exists
        cnvg_file_name = pyNA.__path__.__dict__["_path"][0] + '/cases/' + settings['case_name'] + '/output/' + settings['output_directory_name'] + '/' + 'Convergence.csv'
        if not os.path.isfile(cnvg_file_name):
            file_cvg = open(cnvg_file_name, 'w')
            file_cvg.writelines("Trajectory name , Execution date/time,  Converged")
        else:
            file_cvg = open(cnvg_file_name, 'a')

        # Write convergence output to file
        # file = open(cnvg_file_name, 'a')
        if ipopt[-1] in {'EXIT: Optimal Solution Found.\n', 'EXIT: Solved To Acceptable Level.\n'}:
            file_cvg.writelines("\n" + settings['output_file_name'] + ", " + str(dt.datetime.now()) + ", Converged")
            converged = True
        else:
            file_cvg.writelines("\n" + settings['output_file_name'] + ", " + str(dt.datetime.now()) + ", Not converged")
            converged = False
        file_cvg.close()

        return converged
