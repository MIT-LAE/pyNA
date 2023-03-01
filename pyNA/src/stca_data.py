import pandas as pd


class StcaData:

    def __init__(self) -> None:
        
        # self.shield = dict()
        self.levels_time_history = dict()
        self.spl_distribution = dict()
        self.spl_distribution_supp = dict()


    # def load_shielding_time_history(self, settings):
    #     """
    #     Load shielding time history (Berton et al., 2019).
    #     """

    #     if settings['case_name'] == 'nasa_stca_standard':
    #         for observer in settings['observer_lst']:
    #             if observer == 'lateral':
    #                 file_name = 'shielding_l.csv'
    #             elif observer == 'flyover':
    #                 file_name = 'shielding_f.csv'
    #             elif observer == 'approach':
    #                 file_name = 'shielding_a.csv'
    #             self.shield[observer] = pd.read_csv(settings['pyna_directory']+'/cases/'+settings['case_name'] + '/shielding/'+file_name).values[:,1:]
    #     else:
    #         raise ValueError('No shielding time history available for the "' + settings['case_name'] + '" case' )


    def load_levels_time_history(self, settings) -> None:
        """
        Loads the verification data of the NASA STCA noise assessment (Berton et al., 2019).
        """

        # Source: verification noise assessment data set of NASA STCA (Berton et al., 2019)
        if settings['case_name'] == 'nasa_stca_standard':

            if settings['all_sources']:
                for observer in settings['observer_lst']:
                    if observer in ['lateral', 'flyover']:
                        self.levels_time_history[observer] = pd.read_excel(settings['pyna_directory'] + '/cases/' + settings['case_name'] + '/verification/55t-Depart-Standard-Total.xlsx', sheet_name=observer, engine="openpyxl")
                    elif observer == 'approach':
                        self.levels_time_history[observer] = pd.read_excel(settings['pyna_directory'] + '/cases/' + settings['case_name'] + '/verification/Approach Levels.xlsx', sheet_name='total', engine="openpyxl")

            if settings['jet_mixing_source']:
                for observer in settings['observer_lst']:
                    if observer in ['lateral', 'flyover']:
                        self.levels_time_history[observer] = pd.read_excel(settings['pyna_directory'] + '/cases/' + settings['case_name'] + '/verification/55t-Depart-Standard-Jet CaseA-D.xlsx', sheet_name=observer, engine="openpyxl")
                    elif observer == 'approach':
                        self.levels_time_history[observer] = pd.read_excel(settings['pyna_directory'] + '/cases/' + settings['case_name'] + '/verification/Approach Levels.xlsx', sheet_name='jet', engine="openpyxl")

            if settings['core_source']:
                for observer in settings['observer_lst']:
                    if observer in ['lateral', 'flyover']:
                        self.levels_time_history[observer] = pd.read_excel(settings['pyna_directory'] + '/cases/' + settings['case_name'] + '/verification/55t-Depart-Standard-Core CaseA-D.xlsx', sheet_name=observer, engine="openpyxl")
                    elif observer == 'approach':
                        self.levels_time_history[observer] = pd.read_excel(settings['pyna_directory'] + '/cases/' + settings['case_name'] + '/verification/Approach Levels.xlsx', sheet_name='core', engine="openpyxl")

            if settings['airframe_source']:
                for observer in settings['observer_lst']:
                    if observer in ['lateral', 'flyover']:
                        self.levels_time_history[observer] = pd.read_excel(settings['pyna_directory'] + '/cases/' + settings['case_name'] + '/verification/55t-Depart-Standard-Airframe CaseA-D.xlsx', sheet_name=observer, engine="openpyxl")
                    elif observer == 'approach':
                        self.levels_time_history[observer] = pd.read_excel(settings['pyna_directory'] + '/cases/' + settings['case_name'] + '/verification/Approach Levels.xlsx', sheet_name='airframe', engine="openpyxl")

            if settings['fan_inlet_source']:
                for observer in settings['observer_lst']:
                    if observer in ['lateral', 'flyover']:
                        self.levels_time_history[observer] = pd.read_excel(settings['pyna_directory'] + '/cases/' + settings['case_name'] + '/verification/55t-Depart-Standard-Fan Inlet CaseA-D.xlsx', sheet_name=observer, engine="openpyxl")
                    elif observer == 'approach':
                        self.levels_time_history[observer] = pd.read_excel(settings['pyna_directory'] + '/cases/' + settings['case_name'] + '/verification/Approach Levels.xlsx', sheet_name='fan inlet', engine="openpyxl")

            if settings['fan_discharge_source']:
                for observer in settings['observer_lst']:
                    if observer in ['lateral', 'flyover']:
                        self.levels_time_history[observer] = pd.read_excel(settings['pyna_directory'] + '/cases/' + settings['case_name'] + '/verification/55t-Depart-Standard-Fan Discharge CaseA-D.xlsx', sheet_name=observer, engine="openpyxl")
                    elif observer == 'approach':
                        self.levels_time_history[observer] = pd.read_excel(settings['pyna_directory'] + '/cases/' + settings['case_name'] + '/verification/Approach Levels.xlsx', sheet_name='fan discharge', engine="openpyxl")

        else:
            raise ValueError('No shielding time history available for the "' + settings['case_name'] + '" case' )

        return None


    def load_spl_distribution(self, settings) -> None:
        
        """
        Load verification data for source SPL spectral and directional distributions.
        
        :param settings
        :type settings: 

        :param components: list of components to run
        :type components: list

        """

        if settings['fan_inlet_source']:
            self.spl_distribution['fan_inlet_source'] = pd.read_excel(settings['pyna_directory'] + '/cases/' + settings['case_name'] + '/verification/Fan Module Source.xlsx', sheet_name='Fan Inlet Full').values
            self.spl_distribution_supp['fan_inlet_source'] = pd.read_excel(settings['pyna_directory'] + '/cases/' + settings['case_name'] + '/verification/Fan Module Source.xlsx', sheet_name='Fan Inlet Suppressed').values

        if settings['fan_discharge_source']:
            self.spl_distribution['fan_discharge_source'] = pd.read_excel(settings['pyna_directory'] + '/cases/' + settings['case_name'] + '/verification/Fan Module Source.xlsx', sheet_name='Fan Discharge Full').values
            self.spl_distribution_supp['fan_discharge_source'] = pd.read_excel(settings['pyna_directory'] + '/cases/' + settings['case_name'] + '/verification/Fan Module Source.xlsx', sheet_name='Fan Discharge Suppressed').values

        if settings['core_source']:
            self.spl_distribution['core_source'] = pd.read_excel(settings['pyna_directory'] + '/cases/' + settings['case_name'] + '/verification/Core Module Source.xlsx', sheet_name='Full').values
            self.spl_distribution_supp['core_source'] = pd.read_excel(settings['pyna_directory'] + '/cases/' + settings['case_name'] + '/verification/Core Module Source.xlsx', sheet_name='Suppressed').values

        if settings['jet_mixing_source']:
            self.spl_distribution['jet_mixing_source'] = pd.read_excel(settings['pyna_directory'] + '/cases/' + settings['case_name'] + '/verification/Jet Module Source.xlsx', sheet_name='Full').values
            self.spl_distribution_supp['jet_mixing_source'] = pd.read_excel(settings['pyna_directory'] + '/cases/' + settings['case_name'] + '/verification/Jet Module Source.xlsx', sheet_name='Suppressed').values

        if settings['airframe_source']:
            self.spl_distribution['airframe_source'] = pd.read_excel(settings['pyna_directory'] + '/cases/' + settings['case_name'] + '/verification/Airframe Module Source.xlsx', sheet_name='Full').values
            self.spl_distribution_supp['airframe_source'] = pd.read_excel(settings['pyna_directory'] + '/cases/' + settings['case_name'] + '/verification/Airframe Module Source.xlsx', sheet_name='Suppressed').values

        return None