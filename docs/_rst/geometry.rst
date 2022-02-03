geometry
========

class description
-----------------

.. automodule:: geometry
   :members:
   :undoc-members:
   :show-inheritance:

.. warning::
	
	The Julia version of the *Geometry* component, ``geometry.jl``, is not documented yet. However, it is very similar to the Python version, ``geometry.py``. More information and examples on how OpenMDAO.jl components work can be found `here <https://github.com/byuflowlab/OpenMDAO.jl/tree/master/examples>`_.

theory
------

Given the aircraft position along the flight trajectory as a function of source time, i.e. :math:`(x,y,z)(t_{s})`, the direction between the source and observer, :math:`\overrightarrow{r} = [x_{obs,x}, x_{obs,y}, -x_{obs,z}]^T-[x,y,-z]^T`, with magnitude :math:`r = || \overrightarrow{r} ||`. The normalized direction vector :math:`n_{so} = \frac{\overrightarrow{r}}{r}`. The elevation angle is given by:

.. math::

    \beta = \sin^{-1}(n_{so,z})


Euler transformation angles are used to change the normalized source-observer vector from the Earth-fixed axis system to the aircraft-fixed axis system:

.. math::

    \tilde{n_{so}} = \mathcal{T}_1(\Phi_B) \cdot \mathcal{T}_2(\Theta_B) \cdot \mathcal{T}_3(\Psi_B) \cdot n_{so}   


where :math:`\mathcal{T}_i` is the Euler transformation matrix around the i-th axis. For straight, horizontal flight, the angles :math:`\Phi_B, \Psi_B = 0` and :math:`\Theta_B = \alpha + \gamma`. The polar and azimuthal directivity angles are given by: 

.. math::

	\begin{aligned}
	\theta \quad &= \quad \cos^{-1}(\tilde{n_{so}}_x) \\ 
	\phi \quad &= \quad \tan^{-1} \left(\frac{\tilde{n_{so}}_y}{\tilde{n_{so}}_z}\right) \\
	\end{aligned}


The average speed of sound between the source and the observer, :math:`\bar{c}`, is computed taking a numerical average of the speed of sound, :math:`c_0`, at 11 intermediate altitudes :math:`z_{intermediate}` (including the endpoints). Finally, the observer time at each :math:`t_{o,i}` is calculated using:

.. math::

    t_{o,i} = t_{s,i} + \frac{r_i}{\bar{c}_i}    


