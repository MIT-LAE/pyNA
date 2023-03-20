import openmdao.api as om
import dymos as dm
import os
import datetime as dt
from pyNA.src.aircraft import Aircraft
from pyNA.src.time_history import TimeHistory
from pyNA.src.trajectory import Trajectory
import pyNA


class Problem(om.Problem):
    
    def set_driver_settings(self, settings:dict, objective:str) -> None:
        
        """

        Parameters
        ----------
        settings : dict
            pyna settings
        objective : str
            _
        
        """

        # Set solver settings for the problem
        self.driver = om.pyOptSparseDriver(optimizer='IPOPT')
        self.driver.opt_settings['print_level'] = 5
        self.driver.opt_settings['nlp_scaling_method'] = 'gradient-based'

        self.driver.declare_coloring(tol=1e-12)
        self.model.linear_solver = om.LinearRunOnce()
        self.driver.opt_settings['output_file'] = pyNA.__path__.__dict__["_path"][0] + '/cases/' + settings['case_name'] + '/output/' + settings['output_directory_name'] + '/IPOPT_trajectory_convergence.out'

        if objective == 'noise':
            self.driver.opt_settings['tol'] = 1e-3
            self.driver.opt_settings['acceptable_tol'] = 1e-1
        else:
            self.driver.opt_settings['tol'] = settings['tolerance']
            self.driver.opt_settings['acceptable_tol'] = 1e-2

        self.driver.opt_settings['max_iter'] = settings['max_iter']
        self.driver.opt_settings['mu_strategy'] = 'adaptive'
        self.driver.opt_settings['bound_mult_init_method'] = 'mu-based'
        self.driver.opt_settings['mu_init'] = 0.01
        self.driver.opt_settings['constr_viol_tol'] = 1e-3
        self.driver.opt_settings['compl_inf_tol'] = 1e-3
        self.driver.opt_settings['acceptable_iter'] = 0
        self.driver.opt_settings['acceptable_constr_viol_tol'] = 1e-1
        self.driver.opt_settings['acceptable_compl_inf_tol'] = 1e-1
        self.driver.opt_settings['acceptable_obj_change_tol'] = 1e-1

        return None

    def set_objective(self, objective:str) -> None:
    
        """

        Parameters
        ----------
        objective : str
            _

        """
        
        if objective == 'distance':
            self.model.add_objective('x', index=-1, ref=1000.)

        elif objective == 'time':
            self.model.add_objective('t_s', index=-1, ref=1000.)

        else:
            raise ValueError('Invalid optimization objective specified.')

        return None
    
    @staticmethod
    def check_convergence(settings: dict, filename: str) -> bool:
        """
        Checks convergence of case using optimizer output file.

        settings : dict
            pyna settings
        filename: str
            file name of IPOPT output

        Output
        ------
        converged : bool
            _

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

        # Print output
        print('Converged:', converged)

        return


    def plot_ipopt_convergence_data():
        pass

