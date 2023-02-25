
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

        self.settings = settings

        self.aircraft = Aircraft(settings=settings)

        self.trajectory = Trajectory(settings=settings, mode=trajectory_mode)


    def plot_ipopt_convergence_data():
        pass

    def plot_trajectory(self):

        fig, ax = plt.subplots(2,3, figsize=(20, 8), dpi=100)
        plt.style.use('plot.mplstyle')

        ax[0,0].plot(self.trajectory.path.get_val('trajectory.x'), self.trajectory.path.get_val('trajectory.z'), '-', label='Take-off trajectory module', color='k')
        ax[0,0].set_xlabel('X [m]')
        ax[0,0].set_ylabel('Z [m]')
        ax[0,0].legend(loc='lower left', bbox_to_anchor=(0.0, 1.01), ncol=1, borderaxespad=0, frameon=False)
        ax[0,0].spines['top'].set_visible(False)
        ax[0,0].spines['right'].set_visible(False)

        ax[0,1].plot(self.trajectory.path.get_val('trajectory.t_s'), self.trajectory.path.get_val('trajectory.v'), '-', color='k')
        ax[0,1].set_xlabel('t [s]')
        ax[0,1].set_ylabel(r'$v$ [m/s]')
        ax[0,1].spines['top'].set_visible(False)
        ax[0,1].spines['right'].set_visible(False)

        ax[0,2].plot(self.trajectory.path.get_val('trajectory.t_s'), self.trajectory.path.get_val('trajectory.gamma'), '-', color='k')
        ax[0,2].set_xlabel('t [s]')
        ax[0,2].set_ylabel(r'$\gamma$ [deg]')
        ax[0,2].spines['top'].set_visible(False)
        ax[0,2].spines['right'].set_visible(False)

        ax[1,0].plot(self.trajectory.path.get_val('trajectory.t_s'), 1 / 1000. * self.trajectory.path.get_val('trajectory.F_n'), '-', color='k')
        ax[1,0].set_xlabel('t [s]')
        ax[1,0].set_ylabel(r'$F_n$ [kN]')
        ax[1,0].spines['top'].set_visible(False)
        ax[1,0].spines['right'].set_visible(False)

        ax[1,1].plot(self.trajectory.path.get_val('trajectory.t_s'), self.trajectory.path.get_val('trajectory.tau'), '-', color='k')
        ax[1,1].set_xlabel('t [s]')
        ax[1,1].set_ylabel(r'$TS$ [-]')
        ax[1,1].spines['top'].set_visible(False)
        ax[1,1].spines['right'].set_visible(False)

        ax[1,2].plot(self.trajectory.path.get_val('trajectory.t_s'), self.trajectory.path.get_val('trajectory.alpha'), '-', color='k')
        ax[1,2].set_xlabel('t [s]')
        ax[1,2].set_ylabel(r'$\alpha$ [deg]')
        ax[1,2].spines['top'].set_visible(False)
        ax[1,2].spines['right'].set_visible(False)

        plt.subplots_adjust(hspace=0.37, wspace=0.27)
        plt.show()

        return None

