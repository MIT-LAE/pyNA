ground reflections
==================

class description
-----------------

.. automodule:: ground_reflections
	:members:
	:undoc-members:
	:show-inheritance:
	:exclude-members:

theory
------

Ground absorption and reflection effects}
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Chien-Soroka method is used to assess ground reflections and ground attenuation. The method accounts for the amplification and attenuation of noise when two signals, the direct and reflected signal, arrive at the microphone with a phase shift. Firstly, the path length of the reflected signal is computed: 

.. math::

    r_r = \sqrt{r^2 + 4 z_{\textrm{obs}}^2 + 4rz_{\textrm{obs}}\sin\beta}

where :math:`\beta` is the elevation angle of the source at the observer. The difference between the direct and the reflected wave path is denoted by :math:`\Delta r = r_r - r`. The wavenumber, :math:`k`, and dimensionless frequency, :math:`\eta`,  of the sound wave are computed using: 

.. math::

    \begin{aligned}
    k &= \frac{2\pi f_{sb}}{\bar{c}} \\ 
    \eta &= \frac{2\pi \rho_0 f_{sb}}{\sigma}
    \end{aligned}

where :math:`\sigma` is the specific flow resistance of ground. The cosine of the incidence angle of the reflected signal is:

.. math::

    \cos \theta = \frac{r \sin \beta + 2  z_{\textrm{obs}}}{r_r}

The empirical complex specific ground admittance, :math:`\nu`, is computed using: 

.. math::

    \nu = \frac{1}{\left[ 1 + (6.86\eta)^{-0.75}\right] + (4.36 \eta)^{-0.73} j}

where :math:`j` denotes the imaginary unit. The complex spherical wave reflection coefficient, :math:`Z`, is computed using:

.. math::

    Z = \Gamma + (1-\Gamma)F

where: 

.. math::

	\begin{aligned}
	    \Gamma &= \frac{\cos \theta - \nu}{\cos \theta + \nu} \\
	    F &= -2 \sqrt{\pi} U[-Re(\tau)] \tau \exp(\tau^2) + \frac{1}{2\tau^2} - \frac{3}{(2\tau^2)^2}
	\end{aligned}

The coefficient :math:`\tau` is given by: 

.. math::

    \tau = (\cos \theta + \nu) \sqrt{\frac{k r_r}{2j}} 

and the unit step function, :math:`U`, is given by:  

.. math::
    U = 
    \begin{cases}
    1 \quad & \textrm{if} \quad -\textrm{Re}(\tau) > 0 \\
    1/2 \quad & \textrm{if}\quad -\textrm{Re}(\tau) = 0 \\
    0 \quad & \textrm{if}\quad -\textrm{Re}(\tau) < 0 \\
    \end{cases}


The ground reflection factor, :math:`G`, is computed using: 

.. math::

    G = 1 + R^2 + 2R\exp\left(-(a_{coh} k \Delta r) ^ 2\right) \cos(\alpha + k \Delta r) \frac{\sin(\epsilon  k  \Delta r)}{\epsilon k \Delta r}

where :math:`R` and :math:`\alpha` are the magnitude and angle of the complex spherical wave reflection coefficient, i.e. :math:`R \exp(\alpha j) = Z`. The constants :math:`K` and :math:`\epsilon` are given by:

.. math::

	\begin{aligned}
	K &= 2^{1/(6N_b)} \\
	\epsilon &= K - 1 \\
	\end{aligned}

Finally, the ground absorption and reflection effects are applied to the mean-square acoustic pressure of each sub-band :math:`j` using: 

.. math::

    <p^2_{\textrm{ground-effects}}>_j \ = G <p^2_{\textrm{absorb}}>_j