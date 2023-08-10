"""
OpenDSS Data Structure
@author: Samuel Talkington and Jorge Fernandez 
MIT License
October 2021

"""
import numpy as np
import pandas as pd
import opendssdirect as dss
import warnings


ELEMENT_CLASSES = {
    'Load': dss.Loads,
    'PV': dss.PVsystems,
    'Generator': dss.Generators,
    'Line': dss.Lines,
    'Xfmr': dss.Transformers,
}
LINE_CLASSES = ['Line', 'Xfmr']


class DSS_Data:
    """
    Data class for OpenDSS network models.
    """

    def __init__(self, redirects, verbose=True):
        """
        Initializes an OpenDSS network model. 
        
        Params:s
            - redirects (list): List of strings of filepaths to .dss files
            - verbose (boolean): whether or not to print verbose logs
        """
        self.dss = dss #dss object
        # Initialize model attributes 
        self.verbose=verbose
        self.redirects = None
        self.Y_net = None #internal ybus
        self.currents_dict = dict() #Internal (nodal current injections) currents_dict (static) at a single timestep
        self.voltages_dict = dict() #Internal voltages_dict (static) at a single timestep
        self.powers_dict = dict() #Internal complex powers dict (static) at a single timestep
        self.line_currents_dict = dict() #Internal line currents dict (static) at a single timestep
        self.xfmer_currents_dict = dict() #Internal transformer currents dict (static) at a single timestep
        #Compile all redirect files
        self.compile_dss(redirects)

    def compile_dss(self,redirects):
        """Compiles the DSS redirect files input into the object"""
        if not isinstance(redirects, list):
            self.redirects = [redirects]
        else:
            self.redirects = redirects
        for redirect in self.redirects:
            self.redirect(redirect)
            self.run_command('solve')
        if self.verbose:
            print(f'DSS Compiled Circuit: {self.dss.Circuit.Name()}')
    
    @staticmethod
    def run_command(cmd):
        """Runs any string command"""
        status = dss.run_command(cmd)
        if status:
            print(f'DSS Status ({cmd}): {status}')

    def redirect(self, filename):
        if self.verbose:
            print(f'DSS Running file: {filename}')
        self.run_command(f'redirect "{filename}"')

    
    ##Get methods
    @staticmethod
    def get_all_elements(element='Load'):
        if element in ELEMENT_CLASSES:
            cls = ELEMENT_CLASSES[element]
            df = dss.utils.to_dataframe(cls)
        else:
            df = dss.utils.class_to_dataframe(element, transform_string=lambda x: pd.to_numeric(x, errors='ignore'))
            # df = dss.utils.class_to_dataframe(element)
        return df

    def get_node_base_voltages(self):
        """
        Returns a pandas series of the per-unit base voltages for every node in the network.
        """
        # Create a dictionary to store the voltages
        node_base_voltages = dict()
        #Get all buses
        buses = self.dss.Circuit.AllBusNames()
        # Get the y node order (all nodes in the system) and their appropriate names. We will check for consistency later.
        y_node_order = self.dss.Circuit.YNodeOrder()
        # Iterate through all buses
        for bus in buses:
            # Set the active bus
            self.dss.Circuit.SetActiveBus(bus)
            # Get the base voltage
            bus_base_voltage = self.dss.Bus.kVBase() # V_line to neutral Vbase for 1phase buses
            # Get the nodes at this bus
            bus_nodes = self.dss.Bus.Nodes()
            # Iterate through all nodes at this bus
            for node in bus_nodes:
                # Set the name of the node to be the bus name plus the node number
                node = bus + f'.{node}'
                # Check if the node is in the y node order
                if node not in y_node_order:
                    # Make the node name consistent with the y node order.
                    # For now, if the node name is not in the y node order, assume that it should be captilized and fix this.
                    node = node.upper() #make the node name upper case
                    # Check if the node is in the y node order
                    if node not in y_node_order:
                        #TODO: raise an error here, because this node is not in the y node order. This should never happen.
                        #TODO: add other methods to fix this issue if it does happen.
                        continue
                # Save the base voltage for this node as the bus base voltage.
                node_base_voltages[node] = bus_base_voltage
        # Convert the dictionary to a pandas series
        node_base_voltages = pd.Series(node_base_voltages)
        return  node_base_voltages
    
    def get_node_voltages_mag_pu(self):
        """
        Gets a static dictionary of all of the nodal voltage magnitudes in per unit in the system at a single time step
        """
        voltages_dict = dict()
        
        nodes = self.dss.Circuit.YNodeOrder()
        volts = self.dss.Circuit.YNodeVArray()
        v_bases = self.get_node_base_voltages()

        # organize the voltage for testing
        Volts = np.asarray(volts)
        V = Volts[0::2] +  1j*Volts[1::2]

        for i, node in enumerate(nodes):
            voltages_dict[node] = np.abs(V[i])/(v_bases[node]*1000) #NOTE: this is in per unit, and the base voltages are in kV.

        self.voltages_dict = voltages_dict

        return voltages_dict

    
    def get_node_voltages(self):
        """
        Get static dictionary of all node voltages in the system at a single timestep

        #---- LOOK INTO NODES THAT HAVE ZERO VOLTAGE
        """
        voltages_dict = dict()
        
        nodes = self.dss.Circuit.YNodeOrder()
        volts = self.dss.Circuit.YNodeVArray()

        # organize the voltage for testing 
        Volts = np.asarray(volts)
        V = Volts[0::2] +  1j*Volts[1::2]
        
        for i, node in enumerate(nodes):
            
            # err = self.dss.Circuit.SetActiveElement(node)
            # if(err != ''):
            #    print(err)
            
            voltages_dict[node] = V[i]
        
        self.voltages_dict = voltages_dict

        return voltages_dict 
    
    def get_bus_voltages_pu(self):
        """Gets the static voltages for all buses
        
        TODO - add error handling, ensure that bus phases and nodes are returned.
        """
        err = self.dss.run_command('solve')
        if(not err==""):
            print(err)
        voltages = self.dss.Circuit.AllBusMagPu()
        return voltages,err
    
    def get_node_currents(self):
        """
        Get static dictionary of all node currents in the system at a single timestep
        """
        currents_dict = dict()
        
        nodes = self.dss.Circuit.YNodeOrder()
        currents = self.dss.Circuit.YCurrents()

        #orange the current for testing
        Currents = np.asarray(currents)
        I = Currents[0::2] +  1j*Currents[1::2]

        for i, node in enumerate(nodes):
            self.dss.Circuit.SetActiveElement(node)
            currents_dict[node] = I[i]
        
        self.currents_dict = currents_dict

        return currents_dict

    def get_node_complex_powers(self):
        """
        Get static dictionary of all nodal complex power injections in the system at a single timestep
        """
        powers_dict = dict()
        nodes = self.dss.Circuit.YNodeOrder()
        volts = self.dss.Circuit.YNodeVArray()
        currents = self.dss.Circuit.YCurrents()

        #orange the current for testing
        
        # organize the phasors for dataset construction 
        Volts = np.asarray(volts)
        V = Volts[0::2] +  1j*Volts[1::2] # voltage phasors
        Currents = np.asarray(currents) 
        I = Currents[0::2] +  1j*Currents[1::2] # current phasors 
        S = V*np.conjugate(I) # S=VI* complex power

        for i, node in enumerate(nodes):
            powers_dict[node] = S[i]
        
        self.powers_dict = powers_dict
        
        return powers_dict


    def get_node_ybus(self, init):
        """
        Gets the Ybus matrix for the network at the current timestep/solution.
        """
        warnings.warn("This method is deprecated. It is recommended to use the internal yadi.dss.YMatrix module instead.")
        if init == True:
            self.__initialization()

        # initialize OpenDSS solver
        self.dss.run_command("solve")

        # extract the voltages from the initial setup
        vtd = self.get_node_voltages()
                
        # required to obtain the Ybus without load and generators equivalents
        self.dss.run_command("vsource.source.enabled=no")
        #self.dss.run_command("batchedit load..* enabled=no")
        #self.dss.run_command("batchedit transformer..* enabled=no")
        self.dss.run_command("solve")
        
        nodes = self.dss.Circuit.YNodeOrder()
        
        volts = [vtd[i] for i in nodes] 
        
        #extract the Ybus
        Ytmp = self.dss.Circuit.SystemY()
        
        # get all the nodes
        n = len(self.dss.Circuit.AllNodeNames())
        
        # organize the Ybus to "normal order"
        Ymatrixtmp=np.asarray(Ytmp).reshape((2*n,n), order="F")
        Ymatrixtmp=Ymatrixtmp.T
    
        # OpenDSS rearrange Y bus components
        Y_net = Ymatrixtmp[:,0::2] + 1j*Ymatrixtmp[:,1::2]
    
        #Set the internal Y network
        self.Y_net = Y_net
        
        return Y_net, volts

    def get_line_data(self):
        """
        Returns dictionaries of line data, specifically:
            -BusNames: Array of strings. Get Bus definitions to which each terminal is connected. 0-based array.
            -NumTerminals: Number of Terminals this Circuit Element
            -NumConductors: Number of Conductors per Terminal
            -NodeOrder: Array of integer containing the node numbers (representing phases, for example) for each conductor of each terminal.
        """
        data_lines = {}
        names_lines = self.dss.Lines.AllNames()
        line_idx,line = 0,self.dss.Lines.First()
        while line:
            name_line = names_lines[line_idx] #get name of line
            # Get line data
            line_data = {
                'BusNames': self.dss.CktElement.BusNames(),
                'NumTerminals': self.dss.CktElement.NumTerminals(),
                'NumConductors': self.dss.CktElement.NumConductors(),
                'NodeOrder': self.dss.CktElement.NodeOrder(),
                'Phases': self.dss.Lines.Phases(), #number of phases
                'NormAmps': self.dss.Lines.NormAmps(), #normal ampere rating
                'EmergAmps': self.dss.Lines.EmergAmps(), #emergency ampere rating
            }
            data_lines[name_line] = line_data # Save the data for this line      
            line = self.dss.Lines.Next() #increment line
            line_idx += 1 #increment index
        return data_lines
    
    def get_line_ratings(self):
        """
        Returns a dictionary of the nominal and emergency ratings for each line.
        """
        ratings_lines = {}
        names_lines = self.dss.Lines.AllNames()
        line_idx,line = 0,self.dss.Lines.First()
        while line:
            name_line = names_lines[line_idx]
            # Get line ratings
            line_ratings = {
                'NormAmps': self.dss.Lines.NormAmps(), #normal ampere rating
                'EmergAmps': self.dss.Lines.EmergAmps(), #emergency ampere rating
            }
            ratings_lines[name_line] = line_ratings # Save the ratings for this line
            line = self.dss.Lines.Next() #increment line
            line_idx += 1 #increment index
        return ratings_lines

    def get_line_currents(self,structure="matrix"):
        """
        Gets the lines currents in the system at the current timestep/solution. 
        The first key is always the line name.

        Parameters
        ----------
        structure : str, optional
            The structure of the output. The default is "matrix".
            - If structure == "dict": 
                - Then the value is a dictionary, where:
                - The first key is the terminal number (1,2), 
                    - i.e., whether to extract the current flowing from terminal 1->2 or 2->1.
                - The second key is the phase (a,b,c)
                - The value is the from-to current for that terminal and phase
            - If structure == "matrix":
                - Then the value is a 2x3 matrix, where:
                - The rows are the terminals (from or to) and the columns are the phases \in {a,b,c}
        Note that I_{n1,n2}^{(\phi)} != I_{n2,n1}^{(\phi)} in general.
        """
        
        network_line_currents = {}
        names_lines = self.dss.Lines.AllNames()
        line_idx,line = 0,self.dss.Lines.First()

        while line: #iterate over lines
            line_label = names_lines[line_idx] #get name of line
            R2n_line_currents = np.asarray(self.dss.CktElement.Currents()) #get line currents at this line (dimensionality R^{2n})
            line_currents = R2n_line_currents[0::2] + 1j*R2n_line_currents[1::2] #convert to complex
            num_terminals = self.dss.CktElement.NumTerminals() #get number of terminals
            num_phases = self.dss.Lines.Phases() #get number of phases

            # Check that the number of terminals and phases are valid
            if num_terminals < 2:
                raise Exception("There is a floating line, check that each line has two terminals.")
            if num_phases == 0:
                raise Exception("Invalid line specification. There are no phases in this line.")

            # Get line currents in the appropriate structure
            if structure == "matrix":
                f_currents = line_currents[0:num_phases]
                t_currents = line_currents[num_phases:]        
                # 2xnum_phases matrix, where the rows are the terminals (from or to) and the columns are the phases \in {a,b,c}
                network_line_currents[line_label] = np.vstack((f_currents,t_currents)) 
            elif structure == "dict":
                terminal_currents = {} # dictionary of dictionaries of phase currents for each terminal
                for term_idx in range(num_terminals): #terminal index (0,1)
                    phase_currents = {} # dictionary of currents by phase for the current terminal terminal
                    terminal_label = str(term_idx+1) #terminal label (1,2)
                    for ph_idx in range(num_phases): #phase index (0,1,2)
                        phase_number = self.dss.CktElement.NodeOrder()[ph_idx] #phase number: (0,1,2) -> (1,2,3)
                        phase_label =  self.__make_phase_label(phase_number) #phase_label: (1,2,3) -> (a,b,c)
                        phase_currents[phase_label] = line_currents[term_idx*num_phases+ph_idx]
                    terminal_currents[terminal_label] = phase_currents
                network_line_currents[line_label] = terminal_currents
            else:
                raise Exception("Invalid structure. Must be 'matrix', or 'dict'.")

            line = self.dss.Lines.Next() #increment line
            line_idx += 1 #increment index
        self.line_currents_dict = network_line_currents
        return network_line_currents
    

    def get_xfmr_currents(self,structure='matrix',include_neutral=True):
        """
        Gets the transformer currents in the system at the current timestep/solution. 
        The first key is always the transformer name.

        Parameters:
            - include_neutral : Bool - whether or not to include the neutral phase in the extracted currents. Defaults to true.
        """
        network_xfmr_currents = {}
        names_xfmrs = self.dss.Transformers.AllNames()
        xfmr_idx,xfmr = 0,self.dss.Transformers.First()
        while xfmr:
            xfmr_label = names_xfmrs[xfmr_idx] #get name of xfmr
            # get the transformer phase currents --- note that this includes a subset of {a,b,c} and {0}, where {0} is the neutral
            R2n_xfmr_currents = np.asarray(self.dss.CktElement.Currents()) #get line currents at this line (dimensionality R^{2n})
            xfmr_currents = R2n_xfmr_currents[0::2] + 1j*R2n_xfmr_currents[1::2] #convert to complex
            num_terminals = self.dss.CktElement.NumTerminals() #get number of terminals
            num_phases = self.dss.CktElement.NumConductors() #get number of phases --- note this includes the Neutral conductor.
            if structure == 'dict':
                warnings.warn("Dict structure not yet implemented for transformer currents. Returning matrix structure.")
                structure = 'matrix'
            if structure == 'matrix':
                if include_neutral: #include all phases in the node order.    
                    f_currents = xfmr_currents[0:num_phases]
                    t_currents = xfmr_currents[num_phases:]        
                    # 2xnum_phases matrix, where the rows are the terminals (from or to) and the columns are the phases \in {a,b,c}
                    network_xfmr_currents[xfmr_label] = np.vstack((f_currents,t_currents)) 
                else: #do not include the neutral phase of the transformers. Warning: this may cause issues with relating to the Ybus order.
                    if self.dss.CktElement.NodeOrder()[num_phases-1] == 0: #check that the neutral phase is the last phase
                        f_currents = xfmr_currents[0:num_phases-1] # -1 to exclude the neutral phase
                        t_currents = xfmr_currents[num_phases:-1] # -1 to exclude the neutral phase
                        # 2x(num_phases-1) matrix, where the rows are the terminals (from or to) and the columns are the phases \in {a,b,c}
                        network_xfmr_currents[xfmr_label] = np.vstack((f_currents,t_currents))
                    else:
                        raise Exception(
                            "Invalid specification of transformer {x}. The neutral phase is not the last phase. Please re-order the phases".format(
                                x=xfmr_label
                            ))
            else:
                raise Exception("Invalid structure. Must be 'matrix', or 'dict'.")
            xfmr = self.dss.Transformers.Next() 
            xfmr_idx += 1 #increment index
        self.xfmer_currents_dict = network_xfmr_currents # Save the network transformer current dictionary
        return network_xfmr_currents

    def get_xfmr_ratings(self):
        """
        Returns a dictionary of the nominal and emergency ratings for each transformer.
        """
        ratings_xfmrs = {}
        names_xfmrs = self.dss.Transformers.AllNames()
        xfmr_idx,xfmr = 0,self.dss.Transformers.First()
        while xfmr:
            name_xfmr = names_xfmrs[xfmr_idx]
            # Get xfmr ratings
            ratings_xfmrs[name_xfmr] = self.dss.Transformers.kVA()
            xfmr = self.dss.Transformers.Next() #increment xfmr
            xfmr_idx += 1 #increment index
        return ratings_xfmrs

    def get_xfmr_data(self):
        """
        Returns dictionaries of xfmer data, specifically:
            -isDelta: bool: Whether the transormer is delta or wye
            -NumTerminals: Number of Terminals this Circuit Element
            -NumConductors: Number of Conductors per Terminal
            -NodeOrder: Array of integer containing the node numbers (representing phases, for example) for each conductor of each terminal.
        """
        data_xfmrs = {}
        names_xfmrs = self.dss.Transformers.AllNames()
        xfmr_idx,xfmr = 0,self.dss.Transformers.First()
        while xfmr:
            name_xfmr = names_xfmrs[xfmr_idx]
            # Get xfmr data
            xfmr_data = {
                'isDelta': self.dss.Transformers.IsDelta(),
                'NumWindings' : self.dss.Transformers.NumWindings(),
                'MinTap' : self.dss.Transformers.MinTap(), #minimum tap position
                'MaxTap' : self.dss.Transformers.MaxTap(), #maximum tap position
                'NumTerminals': self.dss.CktElement.NumTerminals(),
                'NumConductors': self.dss.CktElement.NumConductors(),
                'NodeOrder': self.dss.CktElement.NodeOrder(),
            }
            data_xfmrs[name_xfmr] = xfmr_data # Save the data for this xfmr
            xfmr = dss.Transformers.Next() #increment xfmr
            xfmr_idx += 1 #increment index
        return data_xfmrs

    #TODO: add a method to get the line and xfmr conductor+phase labels to conductor index
    def get_xfmr_conductor_idx_map(self,flow_direction,include_neutral=True):
        """
        Makes a nested dictionary mapping the conductor terminal->phase->conductor index.
        """
        #-----
        warnings.warn("This method is not yet implemented, doing nothing.")
        pass
        #-----
        term_idx = {} # dictionary of dictionaries of phase currents for each terminal
        for term_idx in range(num_terminals): #terminal index (0,1)
            phase_currents = {} # dictionary of currents by phase for the current terminal terminal
            terminal_label = str(term_idx+1) #terminal label (1,2)
            for ph_idx in range(num_phases): #phase index (0,1,2)
                phase_number = self.dss.CktElement.NodeOrder()[ph_idx] #phase number: (0,1,2) -> (1,2,3)
                phase_label =  self.__make_phase_label(phase_number) #phase_label: (1,2,3) -> (a,b,c)
                phase_currents[phase_label] = line_currents[term_idx*num_phases+ph_idx]
            terminal_currents[terminal_label] = phase_currents
        network_line_currents[line_label] = terminal_currents

    def __make_phase_label(
        phase_number : int, # 1,2,3
        mapping = {
            1: 'a',
            2: 'b',
            3: 'c'
        }
    ):
        """
        Makes a phase label ('a','b','c') from a phase number (1,2,3)
        """
        if phase_number not in mapping.keys():
            if phase_number == 0:
                warnings.warn("Phase number is 0. Assuming you meant '1' -> 'a'")
                return mapping[1]
            else:    
                raise Exception("Invalid phase number. Must be in {1,2,3}")
        else:
            return mapping[phase_number]

    def __initialization(self):
        """Initializies basic DSS parameters"""
        # set maxiterations number
        self.dss.run_command("Set Maxiterations=600")
        # disable the default regulator
        self.dss.run_command("Set controlmode=Off") 
