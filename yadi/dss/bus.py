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
    
    def write_PMD_bus(self):

        # initialize bus structure
        self.bus = {}

        # get all bus names
        bus_names = self.dss.Circuit.AllBusNames()

        # main loop
        for bn in bus_names:

            # set bus as active
            self.dss.Circuit.SetActiveBus(bn)

            # get bus nodes/terminals
            terminals = self.dss.Bus.Nodes()

            # if bus has more than 4 nodes, assume node 4 is grounded
            if len(terminals) > 4:
                grounded = [4]
                rg = [0.0]
                xg = [0.0]
            else:
                grounded = []
                rg = []
                xg = []

            # create structure
            self.bus[bn] = {
            "terminals"   : terminals,
            "grounded"    : grounded,
            "rg"          : rg,
            "xg"          : xg,
            "status"      : "ENABLED",
            "time_series" : {},
            }

