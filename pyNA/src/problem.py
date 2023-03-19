import openmdao.api as om
import dymos as dm
from pyNA.src.aircraft import Aircraft
from pyNA.src.time_history import TimeHistory
from pyNA.src.trajectory import Trajectory
import pyNA


class Problem(om.Problem):

    def solve_time_history(self, trajectory: TimeHistory, settings: dict):

        """
        
        Parameters 
        ----------
        trajectory : 
            _
        settings : 
            _

        """

        # OpenMDAO problem setup
        self.setup(force_alloc_complex=True)
        
        # Set trajectory initial conditions
        trajectory.set_initial_conditions(problem=self, settings=settings)

        # OpenMDAO run model
        self.run_model()

        pass

    def solve_model(self, trajectory: Trajectory, settings: dict, aircraft:Aircraft, objective='time', path_init=None) -> None:
        
        """
        """
                        
        Problem.set_objective(self, objective=objective)
        Problem.set_driver_settings(self, settings=settings, objective=objective)
        Problem.setup(self, force_alloc_complex=True)
        trajectory.set_initial_conditions(problem=self, settings=settings, aircraft=aircraft, path_init=path_init)

        # Attach a recorder to the problem to save model data
        if settings['save_results']:
            self.add_recorder(om.SqliteRecorder(pyNA.__path__.__dict__["_path"][0] + '/cases/' + settings['case_name'] + '/output/' + settings['output_directory_name'] + '/' + settings['output_file_name']))

        # Run problem
        dm.run_problem(self, run_driver=True)

        # Save the results
        if settings['save_results']:
            self.record(case_name=settings['case_name'])

        # Check convergence
        converged = Trajectory.check_convergence(self, settings=settings, filename='IPOPT_trajectory_convergence.out')
        print('Converged:', converged)
        
        return None
    
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