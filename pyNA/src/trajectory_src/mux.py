import pdb
import numpy as np
import openmdao
import openmdao.api as om


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
        self.options.declare('size_inputs', default=np.array([20, 20]), desc='Size of input arrays to be muxed')
        self.options.declare('size_output', default=2, desc='Size of the muxed array')

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
        size_inputs = self.options['size_inputs']
        size_output = self.options['size_output']
        mux_num = len(size_inputs)

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
            self.add_input(name=in_name, shape=(size_inputs[i],), **kwgs)

            # Add partials
            if i < mux_num-1:
                ro = np.arange(n_output, n_output + size_inputs[i]-1)
            else:
                ro = np.arange(n_output, n_output + size_inputs[i])

            # Delete duplicates
            # cols = np.arange(shapes[i])
            if i < mux_num-1:
                co = np.arange(size_inputs[i]-1)
            else:
                co = np.arange(size_inputs[i])

            # Declare partials
            self.declare_partials(of=name, wrt=in_name, rows=ro, cols=co)

            # Add to output size
            if i < mux_num - 1:
                n_output = n_output + (size_inputs[i] - 1)
            else:
                n_output = n_output + size_inputs[i]

        # Add output variable
        self.add_output(name=name,
                        val=np.zeros(int(size_output),),
                        units=options['units'],
                        desc=options['desc'])

    def compute(self, inputs: openmdao.vectors.default_vector.DefaultVector, outputs: openmdao.vectors.default_vector.DefaultVector):

        # Load options
        size_inputs = self.options['size_inputs']
        mux_num = len(size_inputs)

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
                    output_vals.append(input_array[np.arange(size_inputs[i]-1)])
                else:
                    output_vals.append(input_array[np.arange(size_inputs[i])])

            # Write stack of vals to outputs
            outputs[var] = np.hstack(output_vals)

    def compute_partials(self, inputs:openmdao.vectors.default_vector.DefaultVector, partials: openmdao.vectors.default_vector.DefaultVector):

        # Load options
        size_inputs = self.options['size_inputs']
        mux_num = len(size_inputs)

        for var in self._vars:
            invar = self._input_names[var]

            for i, iv in enumerate(invar):

                if i < mux_num-1:
                    partials[var, iv]= np.ones(size_inputs[i]-1)
                else:
                    partials[var, iv]= np.ones(size_inputs[i])

                


