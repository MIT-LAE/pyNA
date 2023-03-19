import matplotlib.pyplot as plt
import dymos as dm
import pandas as pd
import numpy as np
import openmdao.api as om
import datetime as dt
import os
import pdb
import pyNA
from pyNA.src.aircraft import Aircraft


class Trajectory:
    
    """
    
    Attributes
    ----------
    vars : list
        _
    var_units : dict
        _
        
    """

    def __init__(self) -> None:

        self.vars = list()
        self.var_units = dict()
        self.path = dm.Trajectory()

    def connect(self, problem: om.Problem, settings, aircraft: Aircraft, path=dm.Trajectory()) -> None:
        
        """
        
        Parameters
        ----------
        trajectory_mode : str
            'model' or 'time_history'
        
        path : dm.Trajectory()
            Trajectory model

        """

        self.path = path
        self.path.connect_to_model(problem=problem, settings=settings, aircraft=aircraft)

        return
    
    def set_initial_conditions(self, problem: om.Problem, settings: dict, aircraft:Aircraft, path_init=None):
        """
        
        Parameters
        ----------

        """

        self.path.set_initial_conditions(problem=problem, settings=settings, aircraft=aircraft, path_init=path_init)

        return 

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


