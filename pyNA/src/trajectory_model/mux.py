import pdb
import numpy as np
import openmdao
import openmdao.api as om
import pyNA


class Mux(om.ExplicitComponent):
    """
    Mux trajectory phases.

    """

    def __init__(self, **kwargs):
        """
        Instantiate Mux and populate private members.
        """
        super().__init__(**kwargs)

        self._vars = {}
        self._input_names = {}

    def initialize(self):
        """
        Declare options.
        """

        # self.options.declare('gauss_transcription_order', default=3, desc='transcription_order')
        self.options.declare('input_size_array', default=np.array([20, 20]), desc='Size of input arrays to be muxed')
        self.options.declare('output_size', default=2, desc='Size of the muxed array')
        self.options.declare('settings', types=dict)
        self.options.declare('objective', types=str)

    def add_var(self, name:str, val=1.0, units=None, desc=''):
        """
        Add variable to the mux component.

        :param name: variable name
        :type name: str
        :param val: variable value
        :type val: np.ndarray
        :param units: variable units
        :type units: str
        :param desc: variable description
        :type desc: str

        :return: None
        """

        # Load options
        input_size_array = self.options['input_size_array']
        output_size = self.options['output_size']
        mux_num = len(input_size_array)

        self._vars[name] = {'val': val, 'units': units, 'desc': desc}
        options = self._vars[name]
        kwgs = self._vars[name]

        # Add inputs for the number of arrays to be muxed for the variable name
        self._input_names[name] = []
        
        n_output = 0
        for i in range(mux_num):
            # Add input names to options dict
            in_name = '{0}_{1}'.format(name, i)
            self._input_names[name].append(in_name)
            
            # Add inputs to component
            self.add_input(name=in_name, shape=(input_size_array[i],), **kwgs)

            # Add partials
            if i < mux_num-1:
                ro = np.arange(n_output, n_output + input_size_array[i]-1)
            else:
                ro = np.arange(n_output, n_output + input_size_array[i])

            # Delete duplicates
            # cols = np.arange(shapes[i])
            if i < mux_num-1:
                co = np.arange(input_size_array[i]-1)
            else:
                co = np.arange(input_size_array[i])

            # Declare partials
            self.declare_partials(of=name, wrt=in_name, rows=ro, cols=co)

            # Add to output size
            if i < mux_num - 1:
                n_output = n_output + (input_size_array[i] - 1)
            else:
                n_output = n_output + input_size_array[i]

        # Add output variable
        self.add_output(name=name,
                        val=np.zeros(int(output_size),),
                        units=options['units'],
                        desc=options['desc'])

    def compute(self, inputs: openmdao.vectors.default_vector.DefaultVector, outputs: openmdao.vectors.default_vector.DefaultVector):

        # Load options
        input_size_array = self.options['input_size_array']
        mux_num = len(input_size_array)

        # Iterate over the variables in the mux component
        for var in self._vars:

            # Select input variable name 
            invar = self._input_names[var]

            # Append inputs of same variable name to vals
            output_vals=[]
            for i in range(mux_num):    
                # Extract input array
                input_array = inputs[invar[i]]

                if i < mux_num-1:
                    output_vals.append(input_array[np.arange(input_size_array[i]-1)])
                else:
                    output_vals.append(input_array[np.arange(input_size_array[i])])

            # Write stack of vals to outputs
            outputs[var] = np.hstack(output_vals)

            if var in ['tau', 'theta_flaps', 'alpha']:
                if self.options['objective'] == 'noise':
                    # Write TS to file
                    f = open(pyNA.__path__.__dict__["_path"][0] + '/' + self.options['settings']['case_name'] + '/output/' + self.options['settings']['output_directory_name'] + '/' + 'inputs_' + var + '.txt' , 'a')
                    f.write(str(outputs[var]) + '\n')
                    f.close()

    def compute_partials(self, inputs:openmdao.vectors.default_vector.DefaultVector, partials: openmdao.vectors.default_vector.DefaultVector):

        # Load options
        input_size_array = self.options['input_size_array']
        mux_num = len(input_size_array)

        for var in self._vars:
            invar = self._input_names[var]

            for i, iv in enumerate(invar):

                if i < mux_num-1:
                    partials[var, iv]= np.ones(input_size_array[i]-1)
                else:
                    partials[var, iv]= np.ones(input_size_array[i])

                


