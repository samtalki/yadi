"""
OpenDSS Data Structure
@author: Samuel Talkington and Jorge Fernandez 
MIT License
October 2021

"""
import numpy as np
import pandas as pd
import opendssdirect as dss
import math

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
    @author: Samuel Talkington and Jorge Fernandez
    Data class for OpenDSS network models.
    """

    def __init__(self, redirects, verbose=True):
        """
        @author: Samuel Talkington
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


    ##
    # Setter methods
    ##

    def set_loads(self, loadP, loadQ, loadNames):
        """
        @author: Jorge Fernandez
        """

        # rename nodes to explicit load names
        loadP = loadP.loc[loadNames.index].rename(index=loadNames)
        loadQ = loadQ.loc[loadNames.index].rename(index=loadNames)
                
        # modify loads
        self.__modify_loads(loadP, loadQ)


    def __modify_loads(self, instDemandP, instDemandQ):
        """
        @author: Jorge Fernandez
        Method to modify loads from a DSS file according to dispatch
        """
        
        elems = self.dss.circuit_all_element_names()
        for i, elem in enumerate(elems):
            self.dss.circuit_set_active_element(elem)
            if "Load" in elem:   
                # extract load name
                loadName = elem.split(".")[1]
                # write load name
                self.dss.loads_write_name(loadName)
                # read kvar
                kvar = instDemandQ[loadName]
                # save kw
                kw = instDemandP[loadName]
                if math.isnan(kw):
                    breakpoint()
                self.__modify_load(kw, kvar,  loadName)

    def __modify_load(self, kw, kvar, load):
        self.dss.text(f"edit load.{load} "
                      f"kw={kw} "
                      f"kvar={kvar}")

    ##
    # Getter methods
    ##

    def get_all_elements(self,element='Load'):
        if element in ELEMENT_CLASSES:
            cls = ELEMENT_CLASSES[element]
            df = self.dss.utils.to_dataframe(cls)
        else:
            df = self.dss.utils.class_to_dataframe(element, transform_string=lambda x: pd.to_numeric(x, errors='ignore'))
            # df = dss.utils.class_to_dataframe(element)
        return df

    def get_node_voltages(self):
        """
        @author: Samuel Talkington
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
        @author: Samuel Talkington
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
        @author: Samuel Talkington
        Get static dictionary of all node complex powers in the system at a single timestep
        """

        self.powers_dict = dict()

        if(self.currents_dict == None or self.voltages_dict == None):
            raise Exception('No internal currents dict or voltages dict found')
        
        for node in self.dss.Circuit.YNodeOrder():
            self.powers_dict[node] = self.voltages_dict[node]*np.conjugate(self.currents_dict[node]) #S=VI*
        
        return self.powers_dict


    def get_node_ybus(self, init):
        """
        @author: Samuel Talkington and Jorge Fernandez
        """
        
        if init == True:
            self.__initialize_dss()

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


    def get_line_flows(self, lname):
        """
        @author: Jorge Fernandez
        Given a list of line names lname, return 4 tuple (Pjk,Qjk,pjk,qjk) 
        """
        # get line flows
        lines, lines_KW_power, lines_KVAR_power = self.__get_circuit_line_flows()
        
        # organize line flows for node-based analysis
        Pjk, Qjk, pjk, qjk = self.__read_line_flows(lines, lines_KW_power, lines_KVAR_power, lname)
        return Pjk, Qjk, pjk, qjk
    
    def __read_line_flows(self, lines, lines_KW_power, lines_KVAR_power, lname):
        """
        @author: Jorge Fernandez
        """
        # active flows
        Pjk = np.zeros([len(lname)])  # containing flows Pkm
        Pkj = np.zeros([len(lname)])  # containing flows Pmk
        # reactive flows
        Qjk = np.zeros([len(lname)])  # containing flows Pkm
        Qkj = np.zeros([len(lname)])  # containing flows Pmk
        # aggregated per line
        pjk = np.zeros([len(lines)])  # containing flows Pkm
        qjk = np.zeros([len(lines)])  # containing flows Pkm
        cont = 0
        for l, line in enumerate(lines):

            p = np.asarray(lines_KW_power[line])
            q = np.asarray(lines_KVAR_power[line])

            Pjk[cont: cont + int(len(p)/2)] = p[:int(len(p)/2)]
            Pkj[cont: cont + int(len(p)/2)] = p[int(len(p)/2):]

            Qjk[cont: cont + int(len(p)/2)] = q[:int(len(p)/2)]
            Qkj[cont: cont + int(len(p)/2)] = q[int(len(p)/2):]

            # total line flow
            pjk[l] = sum(p[:int(len(p)/2)])
            qjk[l] = sum(q[:int(len(p)/2)])

            cont += int(len(p)/2)

        return Pjk, Qjk, pjk, qjk

    #TODO: Verify accuracy of changes in  PyDSS interface calls to OpenDSSDirect.
    def __get_circuit_line_flows(self):
        """
        Get the line flows for all elements in the circuit.
        @author: Jorge Fernandez and Samuel Talkington
        """
        # prelocate 
        lines = list()
        lines_KW_dict = dict()
        lines_KVAR_dict = dict()
        # dss elements
        elements = self.dss.Circuit.AllElementNames()
        for i, elem in enumerate(elements):
            self.dss.Circuit.SetActiveElement(elem)
            if "Line" in elem:
                lines.append(elem)
                lines_KW_dict[elem] = self.dss.CktElement.Powers()[0::2]
                lines_KVAR_dict[elem] = self.dss.CktElement.Powers()[1::2]

            elif "Transformer" in elem:
                lines.append(elem)
                if elem == 'Transformer.xfm1':
                    lines_KW_dict[elem] = self.dss.CktElement.Powers()[0::2]
                    lines_KVAR_dict[elem] = self.dss.CktElement.Powers()[1::2]
                    del lines_KW_dict[elem][3]
                    del lines_KW_dict[elem][-1]
                    del lines_KVAR_dict[elem][3]
                    del lines_KVAR_dict[elem][-1]
                else: 
                    lines_KW_dict[elem] = [i for i in self.dss.CktElement.Powers()[0::2] if i != 0]
                    lines_KVAR_dict[elem] = [i for i in self.dss.CktElement.Powers()[1::2] if i != 0]
        return lines, lines_KW_dict, lines_KVAR_dict

    def __initialize_dss(self):
        """
        @author: Samuel Talkington
        Initializies basic DSS parameters
        """
        # set maxiterations number
        self.dss.run_command("Set Maxiterations=600")
        # disable the default regulator
        self.dss.run_command("Set controlmode=Off") 
