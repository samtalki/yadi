"""
OpenDSS Data Structure
@author: Samuel Talkington and Jorge Fernandez 
MIT License
October 2021

"""
import numpy as np
import pandas as pd
import opendssdirect as dss


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
        self.currents_dict = dict() #Internal currents_dict (static) at a single timestep
        self.voltages_dict = dict() #Internal voltages_dict (static) at a single timestep
        self.powers_dict = dict() #Internal complex powers dict (static) at a single timestep
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


    
    def get_node_voltages(self):
        """
        Get static dictionary of all node voltages in the system at a single timestep
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
        Get static dictionary of all node complex powers in the system at a single timestep
        """
        powers_dict = dict()
        if(self.currents_dict == None or self.voltages_dict == None):
            raise Exception('No internal currents dict or voltages dict found')
        for node in self.dss.Circuit.YNodeOrder():
            powers_dict[node] = self.voltages_dict[node]*np.conjugate(self.currents_dict[node]) #S=VI*
        self.powers_dict = powers_dict
        return powers_dict


    def get_node_ybus(self, init):
        
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


    def __initialization(self):
        """Initializies basic DSS parameters"""
        # set maxiterations number
        self.dss.run_command("Set Maxiterations=600")
        # disable the default regulator
        self.dss.run_command("Set controlmode=Off") 
