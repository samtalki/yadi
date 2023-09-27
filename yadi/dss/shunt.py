
import numpy as np
import pandas as pd
import yadi.yadi.dss.transformer as transformer 
import os


class DSS_Shunt(transformer.DSS_Transformer):

    def __init__(self, redirects, precompile, verbose=False):
        """"
        Class for handling shunts in OpenDSS.

        """

        super().__init__(redirects, redirects, precompile)
    
    def write_PMD_shunt(self):
        
        # initialize structure
        shunt = {}

        # get all bus names
        shunt_names = self.dss.Capacitors.AllNames()

        # main loop
        for sn in shunt_names:

            # activate capacitor
            self.dss.Capacitors.Name(sn)

            # get bus from capacitor
            f_bus, t_bus = self.dss.CktElement.BusNames()
            f_bus = f_bus.split(".")[0]
            t_bus = t_bus.split(".")[0]

            # get configuration
            if self.dss.Capacitors.IsDelta() == True:
                configuration = "DELTA" 
            else:
                configuration = "WYE" 

            # connections
            connections = self.bus[f_bus]["terminals"]

        return shunt
