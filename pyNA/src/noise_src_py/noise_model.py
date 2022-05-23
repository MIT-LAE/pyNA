import pdb
import os
import numpy as np
import openmdao.api as om
from pyNA.src.data import Data
from pyNA.src.settings import Settings
from pyNA.src.aircraft import Aircraft
from pyNA.src.noise_src_py.normalization_engine_variables import NormalizationEngineVariables
from pyNA.src.noise_src_py.geometry import Geometry
from pyNA.src.noise_src_py.source import Source
from pyNA.src.noise_src_py.shielding import Shielding
from pyNA.src.noise_src_py.propagation import Propagation
from pyNA.src.noise_src_py.levels import Levels
from pyNA.src.noise_src_py.levels_int import LevelsInt

class NoiseModel(om.Group):
    """
    Noise model group. The noise model group connects the following components:

    * Geometry:     compute geometry parameters of the noise problem
    * Source:       compute source mean-square acoustic pressure
    * Propagation:  compute the propagated mean-square acoustic pressure at the observer
    * Levels:       compute the noise levels at the observer
    * LevelsInt:    compute the integrated noise levels at the observer

    The Geometry, Source, Propagation, Levels and LevelsInt components are available both in Python or Julia. The Julia versions allows for a speed-up of the noise model evaluation as well as partial derivative information of the different submodules (which is not available in the Python versions).

    """

    def initialize(self):
        self.options.declare('settings', types=Settings)
        self.options.declare('data', types=Data)
        self.options.declare('ac', types=Aircraft)
        self.options.declare('n_t', types=int, desc='Number of time steps in trajectory')
        self.options.declare('mode', types=str)

    def setup(self):
        # Load options
        settings = self.options['settings']
        n_t = self.options['n_t']
        data = self.options['data']
        ac = self.options['ac']

        # Normalize engine inputs
        self.add_subsystem(name='normalize_engine',
                           subsys=NormalizationEngineVariables(settings=settings, n_t=n_t),
                           promotes_inputs=[],
                           promotes_outputs=[])

        # Geometry module
        self.add_subsystem(name='geometry',
                           subsys=Geometry(settings=settings, n_t=n_t, mode=self.options['mode']),
                           promotes_inputs=[],
                           promotes_outputs=['t_o'])
        
        # Source module
        self.add_subsystem(name='shielding',
                           subsys=Shielding(settings=settings, data=data, n_t=n_t),
                           promotes_inputs=[],
                           promotes_outputs=[])

        # Source module
        self.add_subsystem(name='source',
                           subsys=Source(n_t=n_t, settings=settings, data=data, ac=ac),
                           promotes_inputs=[],
                           promotes_outputs=[])
        self.connect('geometry.theta', 'source.theta')
        self.connect('geometry.phi', 'source.phi')
        self.connect('shielding.shield', 'source.shield')
        if settings.jet_mixing and settings.jet_shock == False:
            self.connect('normalize_engine.V_j_star', 'source.V_j_star')
            self.connect('normalize_engine.rho_j_star', 'source.rho_j_star')
            self.connect('normalize_engine.A_j_star', 'source.A_j_star')
            self.connect('normalize_engine.Tt_j_star', 'source.Tt_j_star')
        elif settings.jet_shock and settings.jet_mixing == False:
            self.connect('normalize_engine.V_j_star', 'source.V_j_star')
            self.connect('normalize_engine.A_j_star', 'source.A_j_star')
            self.connect('normalize_engine.Tt_j_star', 'source.Tt_j_star')
            self.connect('normalize_engine.M_j', 'source.M_j')
        elif settings.jet_shock and settings.jet_mixing:
            self.connect('normalize_engine.V_j_star', 'source.V_j_star')
            self.connect('normalize_engine.rho_j_star', 'source.rho_j_star')
            self.connect('normalize_engine.A_j_star', 'source.A_j_star')
            self.connect('normalize_engine.Tt_j_star', 'source.Tt_j_star')
        if settings.core:
            if settings.method_core_turb == 'GE':
                self.connect('normalize_engine.mdoti_c_star', 'source.mdoti_c_star')
                self.connect('normalize_engine.Tti_c_star', 'source.Tti_c_star')
                self.connect('normalize_engine.Ttj_c_star', 'source.Ttj_c_star')
                self.connect('normalize_engine.Pti_c_star', 'source.Pti_c_star')
                self.connect('normalize_engine.DTt_des_c_star', 'source.DTt_des_c_star')
            elif settings.method_core_turb == 'PW':
                self.connect('normalize_engine.mdoti_c_star', 'source.mdoti_c_star')
                self.connect('normalize_engine.Tti_c_star', 'source.Tti_c_star')
                self.connect('normalize_engine.Ttj_c_star', 'source.Ttj_c_star')
                self.connect('normalize_engine.Pti_c_star', 'source.Pti_c_star')
                self.connect('normalize_engine.rho_te_c_star', 'source.rho_te_c_star')
                self.connect('normalize_engine.c_te_c_star', 'source.c_te_c_star')
                self.connect('normalize_engine.rho_ti_c_star', 'source.rho_ti_c_star')
                self.connect('normalize_engine.c_ti_c_star', 'source.c_ti_c_star')
        if settings.fan_inlet or settings.fan_discharge:
            self.connect('normalize_engine.mdot_f_star', 'source.mdot_f_star')
            self.connect('normalize_engine.N_f_star', 'source.N_f_star')
            self.connect('normalize_engine.DTt_f_star','source.DTt_f_star')
            self.connect('normalize_engine.A_f_star', 'source.A_f_star')
            self.connect('normalize_engine.d_f_star', 'source.d_f_star')

        if self.options['mode'] == 'trajectory':
            # Propagation module
            self.add_subsystem(name='propagation',
                            subsys=Propagation(n_t=n_t, settings=settings, data=data),
                            promotes_inputs = [],
                            promotes_outputs = [])
            self.connect('geometry.c_bar', 'propagation.c_bar')
            self.connect('geometry.r', 'propagation.r')
            self.connect('geometry.beta', 'propagation.beta')
            self.connect('source.msap_source', 'propagation.msap_source')

            # Levels module
            if settings.bandshare:
                promotes_outputs_lst = ['spl', 'oaspl', 'pnlt', 'C']
            else:
                promotes_outputs_lst = ['spl', 'oaspl', 'pnlt']
            self.add_subsystem(name='levels',
                            subsys=Levels(n_t=n_t, settings=settings, data=data),
                            promotes_inputs=[],
                            promotes_outputs=promotes_outputs_lst)
            self.connect('propagation.msap_prop', 'levels.msap_prop')

            # Integrated levels module
            if settings.levels_int_metric == 'ioaspl':
                promotes_inputs_lst = ['t_o', 'oaspl']
            elif settings.levels_int_metric == 'ipnlt':
                promotes_inputs_lst = ['t_o', 'pnlt']
            elif settings.levels_int_metric == 'epnl':
                if settings.bandshare:
                    promotes_inputs_lst = ['t_o', 'pnlt', 'C']
                else:
                    promotes_inputs_lst = ['t_o', 'pnlt']
            self.add_subsystem(name='levels_int',
                            subsys=LevelsInt(n_t=n_t, settings=settings),
                            promotes_inputs=promotes_inputs_lst,
                            promotes_outputs=[settings.levels_int_metric])

        elif self.options['mode'] == 'distribution':
            # Levels module
            if settings.bandshare:
                promotes_outputs_lst = ['spl', 'oaspl', 'pnlt', 'C']
            else:
                promotes_outputs_lst = ['spl', 'oaspl', 'pnlt']
            self.add_subsystem(name='levels',
                            subsys=Levels(n_t=n_t, settings=settings, data=data),
                            promotes_inputs=[],
                            promotes_outputs=promotes_outputs_lst)
            self.connect('source.msap_source', 'levels.msap_prop')