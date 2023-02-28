from pyNA.pyna import pyna
import numpy as np
import unittest
import openmdao.api as om


class TestTrajectory(unittest.TestCase):

    def test_compare_nasa_stca(self):

        # Get NASA STCA trajectory from file
        py_stca = pyna()
        py_stca.trajectory.solve(py_stca.settings)

        # Compute NASA STCA trajectory using take-off model
        z_cb = py_stca.trajectory.path.get_val('z')[np.where(py_stca.trajectory.path.get_val('tau') < 0.65)[0][0]]
        v_max = py_stca.trajectory.path.get_val('v')[-1]
        x_max = py_stca.trajectory.path.get_val('x')[-1]

        py = pyna(case_name='stca',
                trajectory_mode='model',
                z_cb=z_cb,
                v_max=v_max,
                x_max=x_max,
                pkrot=True)

        tau={'groundroll':0.88, 'rotation':0.88, 'liftoff':0.88, 'vnrs':0.88, 'cutback':0.61}
        theta_flaps={'groundroll':10., 'rotation':10., 'liftoff':10., 'vnrs':10., 'cutback':10.}
        theta_slats={'groundroll':-6., 'rotation':-6., 'liftoff':-6., 'vnrs':-6., 'cutback':-6.}

        py.trajectory.solve(py.settings, py.aircraft, tau=tau, theta_flaps=theta_flaps, theta_slats=theta_slats)

        # Compute error between the trajectories
        t_end_min = np.min((py.trajectory.path.get_val('t_s')[-1], py_stca.trajectory.path.get_val('t_s')[-1]))

        n = np.size(py_stca.trajectory.path.get_val('t_s'))
        t_ip = np.linspace(0, t_end_min, n)

        for var in ['x', 'z', 'v', 'gamma', 'F_n', 'tau', 'alpha']:

            x_ip = np.interp(t_ip, py.trajectory.path.get_val('t_s'), py.trajectory.path.get_val(var)) / np.max(py_stca.trajectory.path.get_val(var))
            x_ip_stca = np.interp(t_ip, py_stca.trajectory.path.get_val('t_s'), py_stca.trajectory.path.get_val(var)) / np.max(py_stca.trajectory.path.get_val(var))
            
            # Check whether mean relative error between take-off model and NASA stca data is less than 5%
            mre = sum(abs(x_ip-x_ip_stca))/n
            self.assertLessEqual(mre, 0.05)


if __name__ == '__main__':
	unittest.main()