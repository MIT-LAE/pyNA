
from pyNA.src.trajectory import Trajectory
from pyNA.src.aircraft import Aircraft
import matplotlib.pyplot as plt
import pdb


class pyna:

    def __init__(self, settings, trajectory_mode='data') -> None:
        
        """
        
        :param trajectory_mode: data or model
        :type trajectory_mode: str
        """

        self.settings = settings.copy()

        self.aircraft = Aircraft(settings=settings)
        if trajectory_mode == 'model':
            self.aircraft.get_aerodynamics_deck(settings=settings)
            self.aircraft.engine.get_performance_deck(settings=settings)

        self.trajectory = Trajectory(mode=trajectory_mode)


    def plot_ipopt_convergence_data():
        pass

    def plot_trajectory(self, paths_compare=[], labels_compare=[]):

        """
        
        :param paths_compare:
        :type paths_compare: tuple of om.Problem()
        """

        fig, ax = plt.subplots(2,3, figsize=(20, 8), dpi=100)
        plt.style.use('plot.mplstyle')

        ax[0,0].plot(self.trajectory.path.get_val('trajectory.x'), self.trajectory.path.get_val('trajectory.z'), '-', label='Take-off trajectory module', color='k')
        for i,path in enumerate(paths_compare):
            ax[0,0].plot(path.get_val('trajectory.x'), path.get_val('trajectory.z'), '-', label=labels_compare[i])
        ax[0,0].set_xlabel('X [m]')
        ax[0,0].set_ylabel('Z [m]')
        ax[0,0].legend(loc='lower left', bbox_to_anchor=(0.0, 1.01), ncol=1, borderaxespad=0, frameon=False)
        ax[0,0].spines['top'].set_visible(False)
        ax[0,0].spines['right'].set_visible(False)

        ax[0,1].plot(self.trajectory.path.get_val('trajectory.t_s'), self.trajectory.path.get_val('trajectory.v'), '-', color='k')
        for path in paths_compare:
            ax[0,1].plot(path.get_val('trajectory.t_s'), path.get_val('trajectory.v'), '-')
        ax[0,1].set_xlabel('t [s]')
        ax[0,1].set_ylabel(r'$v$ [m/s]')
        ax[0,1].spines['top'].set_visible(False)
        ax[0,1].spines['right'].set_visible(False)

        ax[0,2].plot(self.trajectory.path.get_val('trajectory.t_s'), self.trajectory.path.get_val('trajectory.gamma'), '-', color='k')
        for path in paths_compare:
            ax[0,2].plot(path.get_val('trajectory.t_s'), path.get_val('trajectory.gamma'), '-')
        ax[0,2].set_xlabel('t [s]')
        ax[0,2].set_ylabel(r'$\gamma$ [deg]')
        ax[0,2].spines['top'].set_visible(False)
        ax[0,2].spines['right'].set_visible(False)

        ax[1,0].plot(self.trajectory.path.get_val('trajectory.t_s'), 1 / 1000. * self.trajectory.path.get_val('trajectory.F_n'), '-', color='k')
        for path in paths_compare:
            ax[1,0].plot(path.get_val('trajectory.t_s'), 1 / 1000. * path.get_val('trajectory.F_n'), '-')
        ax[1,0].set_xlabel('t [s]')
        ax[1,0].set_ylabel(r'$F_n$ [kN]')
        ax[1,0].spines['top'].set_visible(False)
        ax[1,0].spines['right'].set_visible(False)

        ax[1,1].plot(self.trajectory.path.get_val('trajectory.t_s'), self.trajectory.path.get_val('trajectory.tau'), '-', color='k')
        for path in paths_compare:
            ax[1,1].plot(path.get_val('trajectory.t_s'), path.get_val('trajectory.tau'), '-')
        ax[1,1].set_xlabel('t [s]')
        ax[1,1].set_ylabel(r'$\tau$ [-]')
        ax[1,1].spines['top'].set_visible(False)
        ax[1,1].spines['right'].set_visible(False)

        ax[1,2].plot(self.trajectory.path.get_val('trajectory.t_s'), self.trajectory.path.get_val('trajectory.alpha'), '-', color='k')
        for path in paths_compare:
            ax[1,2].plot(path.get_val('trajectory.t_s'), path.get_val('trajectory.alpha'), '-')
        ax[1,2].set_xlabel('t [s]')
        ax[1,2].set_ylabel(r'$\alpha$ [deg]')
        ax[1,2].spines['top'].set_visible(False)
        ax[1,2].spines['right'].set_visible(False)

        plt.subplots_adjust(hspace=0.37, wspace=0.27)
        plt.show()

        return None

