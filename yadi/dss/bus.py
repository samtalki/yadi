import numpy as np
import pandas as pd
import yadi.yadi.dss.load_shape as load_shape 
import os


class DSS_Bus(load_shape.DSS_LoadShape):

    def __init__(self, redirects, precompile, verbose=False):
        """"
        Class for handling buses in OpenDSS.

        """

        super().__init__(redirects, redirects, precompile)

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