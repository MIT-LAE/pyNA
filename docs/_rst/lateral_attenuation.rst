lateral attenuation
===================

class description
-----------------

.. automodule:: lateral_attenuation
	:members:
	:undoc-members:
	:show-inheritance:
	:exclude-members:

theory
------

The engine installation term, :math:`E_{\textrm{engine}}`, is given by: 

.. math::

    E_{\textrm{engine}} = 
    \begin{cases}
    10\log_{10}\left[\frac{(0.0039 \cos^2\phi_d + \sin^2\phi_d) ^ {0.062}}{(0.8786 * \sin^2(2 \phi_d) + \cos^2(2 \phi_d))}\right] & \textrm{if engines are mounted under the wing} \\
    10\log_{10}\left[(0.1225 \cos^2\phi_d + \sin^2\phi_d) ^ {0.329}\right] & \textrm{if engines are fuselage mounted} \\
    0 & \textrm{if propeller engines} \\
    \end{cases}

Assuming the aircraft is flying horizontally (zero bank angle), the depression angle, :math:`\phi_d`, is equal to the elevation angle, :math:`\beta`. The attenuation caused by ground and refraction-scattering effects, :math:`A_{\textrm{grs}}`, is given by: 

.. math::

    A_{\textrm{grs}} = 
    \begin{cases}
    1.137 - 0.0229 \beta + 9.72 \exp(-0.142 \beta) & \textrm{if} \quad \beta \leq 50 \textrm{deg}\\
    0& \textrm{else}\\
    \end{cases}

The overall lateral ground attenuation, :math:`g`, over a lateral length, :math:`l`, is given by:

.. math::

    g = 
    \begin{cases}
        11.83 (1 - \exp(-0.00274l))  &  \textrm{if} \quad 0. \leq l \leq 914\textrm{m} \\
        10.86  &  \textrm{if} \quad l > 914\textrm{m} \\
    \end{cases}

The lateral attenuation factor, :math:`\Lambda`, is given by:

.. math::

    \Lambda = 10^{0.1(E_{\textrm{engine}} - \frac{g A_{\textrm{grs}}}{10.86})}


Note that the lateral attenuation factor is 0 for observers underneath the flight path. Finally, the lateral attenuation are applied to the  mean-square acoustic pressure of each sub-band :math:`j` using: 

.. math::

    <p^2_{\textrm{lateral-attenuation}}>_j \ = \Lambda <p^2_{\textrm{ground-effects}}>_j

To avoid double book-keeping of the empirical lateral attenuation effects and the ground reflection effects, the ground reflection effects should only be applied from the noise source up to the center-line. The ground reflection and lateral attenuation section can then be combined using: 

.. math::

    <p^2_{\textrm{lateral-attenuation}}>_j \ = \Lambda G_{\textrm{center-line}} <p^2_{\textrm{absorb}}>_j
