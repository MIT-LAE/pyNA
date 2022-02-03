levels
======

class description
-----------------

.. automodule:: levels
   :members:
   :undoc-members:
   :show-inheritance:

.. warning::
	
	The Julia version of the *Levels* component, ``levels.jl``, is not documented yet. However, it is very similar to the Python version, ``levels.py``. More information and examples on how OpenMDAO.jl components work can be found `here <https://github.com/byuflowlab/OpenMDAO.jl/tree/master/examples>`_.

theory
------

Sound pressure level
^^^^^^^^^^^^^^^^^^^^
The sound pressure level (SPL) is calculated using:

.. math::

    SPL(\theta,\phi,f) = <p_{\textrm{prop}}^2>^* + \ 20\log_{10}\left[\frac{\rho_0 c_0^2}{p_{ref}}\right]
    \label{eq:SPL}

The reference pressure :math:`p_{\textrm{ref}} = 2\cdot 10^{-5}`.

Overall sound pressure level
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The overall sound pressure level (OASPL) is calculated using: 

.. math::
    OASPL(\theta,\phi) = 10\log_{10}\left[ \sum_{i=1}^{N_{freq}}\left( 10^{\frac{SPL(\theta,\phi,f)}{10}} \right) \right]

Tone-corrected perceived noise level
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The tone-corrected perceived noise level (PNLT) is calculated using the method described in ICAO Annex 16, Volume I \cite{ICAO2017}. The PNLT calculation at each time step requires an SPL spectrum as a function of frequency. In this section, the SPL at the i-th one-third octave frequency band is denoted by :math:`SPL[i]`. 
Firstly, the SPL is converted into perceived noisiness: 

.. math::

    n[i] =
    \begin{cases}
    10^{m_c(SPL[i] - SPL_c[i])} \quad & \quad \textrm{if}\quad SPL_a[i] \leq SPL[i]\\
    10^{m_b(SPL[i] - SPL_b[i])} \quad & \quad \textrm{if}\quad SPL_b[i] \leq SPL[i] \leq SPL_a[i]\\
    0.3 \cdot 10^{m_e(SPL[i] - SPL_e[i])} \quad & \quad \textrm{if}\quad SPL_e[i] \leq SPL[i] \leq SPL_b[i]\\
    0.1 \cdot 10^{m_d(SPL[i] - SPL_d[i])} \quad & \quad \textrm{if}\quad SPL_d[i] \leq SPL[i] \leq SPL_e[i]\\
    0 \quad & \quad \textrm{otherwise}\\
    \end{cases}

The Noy tables (i.e. :math:`SPL_a`, :math:`SPL_b`, :math:`SPL_c`, :math:`SPL_d`, :math:`SPL_e`, :math:`m_b`, :math:`m_c`, :math:`m_d`, :math:`m_d`) are tabulated online. The perceived noisiness values are combined to total perceived noisiness using: 

.. math::

    N = \max(n) + 0.15\left( \sum_{i=1}^{24} - \max(n) \right)

The perceived noise level is computed using:

.. math::
    
    PNL = 40 \ + \ \frac{10}{\log_{10} 2} \log_{10} N

The corrections for spectral irregularities on the perceived noise level involve several steps. *Step 1*: compute the slopes of the SPL spectrum using:

.. math::

    s[i] = 
    \begin{cases}
    N.A. \quad & \textrm{if} \quad i \leq 3\\
    SPL[i] - SPL[i-1] \quad & \textrm{if} \quad \textrm{otherwise} \\
    \end{cases}

*Step 2*: Encircle the values of the slopes in the spectrum with a value larger than 5, i.e. :math:`|s[i] - s[i-1]| > 5`.  *Step 3*: In the spectrum of SPL, encircle: 

.. math::

    \begin{cases}
    \textrm{SPL[i]}     \quad & \textrm{if} \quad \textrm{encircled value of slope} s[i] > 0 \ \textrm{and}\ s[i] > s[i-1] \\ 
    \textrm{SPL[i-1]}   \quad & \textrm{if} \quad \textrm{encircled value of slope} s[i] \leq 0 \ \textrm{and}\ s[i-1] > 0 \\
    \textrm{nothing}    \quad & \textrm{otherwise} \\
    \end{cases}

*Step 4*: Compute adjusted :math:`SPL`, i.e. :math:`SPL'`:

.. math::
    SPL'[i] = 
    \begin{cases}
    SPL[i]  \quad & \textrm{if} \quad \textrm{SPL[i] is not encircled}\\
    0.5(SPL[i-1]+SPL[i+1])  \quad & \textrm{if} \quad \textrm{SPL[i] is encircled}\\
    SPL[23] + s[23]  \quad & \textrm{if} \quad \textrm{i=24 and SPL[i] is encircled}\\
    \end{cases}

*Step 5*: Compute adjusted slopes :math:`s'`, including an 'imaginary' 25-th frequency band:

.. math::

    s'[i] = 
    \begin{cases}
    s'[4] \quad & \textrm{if} \quad \textrm{i = 3}\\
    SPL'[i+1]-SPL'[i]\quad & \textrm{if} \quad \textrm{i $\in $[4,24]}\\
    s[24] \quad & \textrm{if} \quad \textrm{i = 25}\\
    0 \quad & \textrm{if} \quad \textrm{otherwise}\\
    \end{cases}

*Step 6*: Compute average of adjacent adjusted slopes:

.. math::

    \bar{s} = 
    \begin{cases}
        \frac{1}{3}(s'[i] + s'[i+1] + s'[i+2])\quad & \textrm{if} \quad \textrm{i $\in$ [3,23]}\\
        0 \quad & \textrm{if} \quad \textrm{otherwise}\\
    \end{cases}

*Step 7*: Compute final adjusted SPL, i.e. :math:`SPL"`:

.. math::

    SPL"[i] = 
    \begin{cases}
    SPL[i] \quad & \textrm{if} \quad \textrm{i=3}\\
    SPL"[i-1] + \bar{s}[i-1] \quad & \textrm{if} \quad \textrm{i > 3}\\
    0 \quad & \textrm{if} \quad \textrm{otherwise}\\
    \end{cases}

*Step 8*: Compute differences :math:`F[i] = SPL[i]-SPL[i]"`. *Step 9*: Compute the tone corrections, :math:`C[i]`, using:

============================== =================================== =========================
f[i] [Hz]                      F[i] (Step 8)                       C[i]                     
============================== =================================== =========================
:math:`f[i]\in [50, 500)`      :math:`3/2 \leq F[i] < 3`           :math:`F[i]/3 - 1/2`
:math:`.`                      :math:`3<F[i]<20`                   :math:`F/6`
:math:`.`                      :math:`20 \leq F[i]`                :math:`10/3`
:math:`f[i]\in [500, 5000)`    :math:`3/2 \leq F[i] < 3`           :math:`F[i]/3 - 1`
:math:`.`                      :math:`3 \leq F[i] < 20`            :math:`F[i]/3`
:math:`.`                      :math:`20 \leq F[i]`                :math:`20/3`
:math:`f[i]\in [5000, 10000]`  :math:`3/2 \leq F[i] < 3`           :math:`F/3 - 1/2`
:math:`.`                      :math:`3 \leq F[i] < 20`            :math:`F[i]/6`
:math:`.`                      :math:`20 \leq F[i]`                :math:`10/3`
============================== =================================== =========================

*Step 10*: Compute the largest tone-correction and add it to the perceived noise level to obtain the tone-corrected perceived noise level:

.. math::

    PNLT = PNL + \max_{i\in \mathcal{I}} C 

When computing the maximum tone correction, the range of frequencies below 800Hz is ignored when enabling the ``TCF8001`` flag.
