flight\_dynamics
================

class description
-----------------

.. automodule:: flight_dynamics
   :members:
   :undoc-members:
   :show-inheritance:

theory
------

A set of 2-dimensional flight dynamics equations is used in the trajectory module, describing the horizontal and vertical motion, :math:`(x, z)`, of the aircraft. The rate equations for the position vector, :math:`(x, z)`, are given by:

.. math::

    \begin{cases}
    \dot{x} = v \cos \gamma &\\ 
    \dot{z} = v \sin \gamma
    \end{cases}

The rate equation for the velocity, :math:`v`, is given by:

.. math::

    \dot{v} = 
    \begin{cases}
    \frac{1}{m}\left(F_n \cos \tilde{\alpha} - D - F_{fric} - mg \sin \gamma \right) & \textrm{for ground roll and rotation}  \\
    \frac{1}{m}\left(F_n \cos \tilde{\alpha} - D - mg \sin \gamma \right) & \textrm{for climb, vnrs and cutback}  \\
    \end{cases}

where the frictional force, :math:`F_{fric} = \mu (mg-L)`. The effective angle of attack, :math:`\tilde{\alpha}`, is given by the sum of the wing angle of attack, $\alpha$, the thrust inclination angle, :math:`i_{F_n}`, and the wing installation angle, :math:`\alpha_0$, i.e. $\tilde{\alpha} = \alpha + i_{F_n} - \alpha_0`. The rate equation for the climb angle, $\gamma$, is given by: 

.. math::

    \dot{\gamma} = 
    \begin{cases}
    \hspace{2cm} 0 & \textrm{for ground roll and rotation} \\ 
    \frac{1}{m v}\left(F_n  \sin \tilde{\alpha} + L - mg  \cos \gamma\right) & \textrm{for climb, vnrs and cutback}
    \end{cases}

The rate of change of the angle of attack, :math:`\frac{d\alpha}{dt} = cnst.`, is explicitly prescribed in the rotation phase and equal to 3.5deg/s. The equivalent airspeed is defined as :math:`v_{eas} = v \sqrt{\rho_0 / \rho_{sl}}`. The rate equation for the equivalent airspeed is given by: 

.. math::
    
    \dot{v_{eas}} = \dot{v} \sqrt{\frac{\rho_0}{\rho_{sl}}} + \frac{v\dot{z}}{2\sqrt{\rho_0\rho_{sl}}} \frac{d\rho_0}{dz}

The rate of change of ambient density with altitude, :math:`\frac{d\rho_0}{dz}`, is obtained by differentiating the density equation with respect to altitude. Finally, the aircraft landing gear is retracted at an altitude of :math:`z=55m`.

