import matplotlib.pyplot as plt
import pandas as pd
import os
from pyNA.src.trajectory_model.trajectory_model import TrajectoryModel
from pyNA.src.utils.compute_frequency_bands import compute_frequency_bands
from pyNA.src.utils.compute_frequency_subbands import compute_frequency_subbands


class Trajectory:
    
    def __init__(self, settings, airframe, engine, noise_data) -> None:
        
        self.airframe = airframe
        self.engine = engine
        self.noise_data = noise_data

        self.f = compute_frequency_bands(n_frequency_bands=settings['n_frequency_bands'])
        self.f_sb = compute_frequency_subbands(f=self.f, n_frequency_subbands=settings['n_frequency_subbands'])

    def initialize(self, settings):

        self.engine.get_source_noise_variables(settings=settings)
        self.engine.get_deck(settings=settings)

        # self.airframe.load_aerodynamics_deck()
        
        # self.path = TrajectoryModel()

    def compute_path(self, settings, objective):
        
        self.path.create_trajectory(airframe=self.airframe, engine=self.engine, sealevel_atmosphere=self.sealevel_atmosphere, k_rot=self.k_rot, v_max=self.v_max, TS_to=self.TS_to, TS_vnrs=self.TS_vnrs, TS_cb=self.TS_cb, TS_min=self.TS_cb, theta_flaps=self.theta_flaps, theta_flaps_cb=self.theta_flaps_cb, theta_slats=self.theta_slats, atmosphere_type=self.atmosphere_type, atmosphere_dT=self.atmosphere_dT, pkrot=self.pkrot, ptcb=self.ptcb, phld=self.phld, objective=objective, trajectory_mode=trajectory_mode)
        self.path.set_objective(objective='t_end')
        self.path.set_ipopt_settings(objective=objective, tolerance=self.tolerance, max_iter=self.max_iter)
        self.path.setup(force_alloc_complex=True)
        self.path.set_phases_initial_conditions(airframe=self.airframe, z_cb=self.z_cb, v_max=self.v_max, initialization_trajectory=None, trajectory_mode=trajectory_mode)
        self.path.solve(run_driver=True, save_results=self.save_results)

        Trajectory.check_convergence(filename='IPOPT_trajectory_convergence.out')
        
    def compute_noise():
        pass

    def optimize_path():
        pass

    def save_timeseries():
        pass

    def check_convergence(self, filename: str) -> bool:
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
        file_ipopt = open(self.pyna_directory + '/cases/' + self.case_name + '/output/' + self.output_directory_name + '/' + filename, 'r')
        ipopt = file_ipopt.readlines()
        file_ipopt.close()

        # Check if convergence summary excel file exists
        cnvg_file_name = self.pyna_directory + '/cases/' + self.case_name + '/output/' + self.output_directory_name + '/' + 'Convergence.csv'
        if not os.path.isfile(cnvg_file_name):
            file_cvg = open(cnvg_file_name, 'w')
            file_cvg.writelines("Trajectory name , Execution date/time,  Converged")
        else:
            file_cvg = open(cnvg_file_name, 'a')

        # Write convergence output to file
        # file = open(cnvg_file_name, 'a')
        if ipopt[-1] in {'EXIT: Optimal Solution Found.\n', 'EXIT: Solved To Acceptable Level.\n'}:
            file_cvg.writelines("\n" + self.output_file_name + ", " + str(dt.datetime.now()) + ", Converged")
            converged = True
        else:
            file_cvg.writelines("\n" + self.output_file_name + ", " + str(dt.datetime.now()) + ", Not converged")
            converged = False
        file_cvg.close()

        return converged


    def plot_ipopt_convergence_data():
        pass



    def plot(self, *problem_verify):

        # Check if problem_verify is empty
        if problem_verify:
            verification = True
            problem_verify = problem_verify[0]
        else:
            verification = False
        fig, ax = plt.subplots(2,3, figsize=(20, 8), dpi=100)
        plt.style.use(self.pyna_directory + '/utils/' + 'plot.mplstyle')

        ax[0,0].plot(self.get_val('trajectory.x'), self.get_val('trajectory.z'), '-', label='Take-off trajectory module', color='k')
        if verification:
            ax[0,0].plot(problem_verify['X [m]'], problem_verify['Z [m]'], '--', label='NASA STCA (Berton et al.)', color='tab:orange')
        ax[0,0].set_xlabel('X [m]')
        ax[0,0].set_ylabel('Z [m]')
        ax[0,0].legend(loc='lower left', bbox_to_anchor=(0.0, 1.01), ncol=1, borderaxespad=0, frameon=False)
        ax[0,0].spines['top'].set_visible(False)
        ax[0,0].spines['right'].set_visible(False)

        ax[0,1].plot(self.get_val('trajectory.t_s'), self.get_val('trajectory.v'), '-', color='k')
        if verification:
            ax[0,1].plot(problem_verify['t_source [s]'], problem_verify['V [m/s]'], '--', label='NASA STCA (Berton et al.)', color='tab:orange')
        ax[0,1].set_xlabel('t [s]')
        ax[0,1].set_ylabel(r'$v$ [m/s]')
        ax[0,1].spines['top'].set_visible(False)
        ax[0,1].spines['right'].set_visible(False)

        ax[0,2].plot(self.get_val('trajectory.t_s'), self.get_val('trajectory.gamma'), '-', color='k')
        if verification:
            ax[0,2].plot(problem_verify['t_source [s]'], problem_verify['gamma [deg]'], '--', color='tab:orange')
        ax[0,2].set_xlabel('t [s]')
        ax[0,2].set_ylabel(r'$\gamma$ [deg]')
        ax[0,2].spines['top'].set_visible(False)
        ax[0,2].spines['right'].set_visible(False)

        ax[1,0].plot(self.get_val('trajectory.t_s'), 1 / 1000. * self.get_val('engine.F_n'), '-', color='k')
        if verification:
            ax[1,0].plot(problem_verify['t_source [s]'], 1 / 1000. * problem_verify['F_n [N]'], '--', color='tab:orange')
        ax[1,0].set_xlabel('t [s]')
        ax[1,0].set_ylabel(r'$F_n$ [kN]')
        ax[1,0].spines['top'].set_visible(False)
        ax[1,0].spines['right'].set_visible(False)

        ax[1,1].plot(self.get_val('trajectory.t_s'), self.get_val('trajectory.TS'), '-', color='k')
        if verification:
            ax[1,1].plot(problem_verify['t_source [s]'], problem_verify['TS [-]'], '--', color='tab:orange')
        ax[1,1].set_xlabel('t [s]')
        ax[1,1].set_ylabel(r'$TS$ [-]')
        ax[1,1].spines['top'].set_visible(False)
        ax[1,1].spines['right'].set_visible(False)

        ax[1,2].plot(self.get_val('trajectory.t_s'), self.get_val('trajectory.alpha'), '-', color='k')
        if verification:
            ax[1,2].plot(problem_verify['t_source [s]'], problem_verify['alpha [deg]'], '--', color='tab:orange')
        ax[1,2].set_xlabel('t [s]')
        ax[1,2].set_ylabel(r'$\alpha$ [deg]')
        ax[1,2].spines['top'].set_visible(False)
        ax[1,2].spines['right'].set_visible(False)

        plt.subplots_adjust(hspace=0.37, wspace=0.27)
        plt.show()

        return None


