
import numpy as np
import pandas as pd
import yadi.yadi.dss.shunt as shunt 
import os


class DSS_Load(shunt.DSS_Shunt):

    def __init__(self, redirects, precompile, verbose=False):
        """"
        Class for handling loads in OpenDSS.

        """

        super().__init__(redirects, redirects, precompile)

    def create_loads(self):

        # initialize load container 
        self.loads = [] 

        # set first load as active 
        self.dss.Loads.First()

        while True:
            # set load as active
            ln = self.dss.Loads.Name()


            # build dictionary with required data for visualization    
            load = {
                "uid": ln,
                "p": [],
                "q": [],
            }

            # append to container
            self.loads.append(load)

            if not self.dss.Loads.Next() > 0:
                break

    def read_load_power(self):

        for load in self.loads:

            # Kw and Kvar of the load
            p = self.dss.Loads.kW()
            q = self.dss.Loads.kvar()

            # get load uid
            uid = load["uid"]

            # set active load
            self.dss.Circuit.SetActiveElement(f"Load.{uid}")

            # get power
            load["p"].append(p)
            load["q"].append(q)

    def get_loadBusAndVln(self):
        "Method to extract load buses from a feeder"
        loadBusDict = dict()
        loadVlnDict = dict()
        loads = self.dss.Loads.AllNames()
        for load in loads:
            # set load as active
            self.dss.Loads.Name(load)
            # name = self.dss.CktElement.Name()  # sanity check
            # get bus name
            buses = self.dss.CktElement.BusNames()
            bus = buses[0]
            # save name
            loadBusDict[load] = bus
            loadVlnDict[load] = self.dss.Loads.kV()
        return pd.Series(loadBusDict), pd.Series(loadVlnDict)
    