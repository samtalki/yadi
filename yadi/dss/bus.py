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
    
    def create_buses(self):

        # initialize bus container 
        self.buses = [] 

        # iterate over all bus names
        for bn in self.dss.Circuit.AllBusNames():

            # set active bus
            self.dss.Circuit.SetActiveBus(bn)

            # number of nodes
            num_nodes = self.dss.Bus.NumNodes()

            # get xfmr phases
            num_phases = self.dss.CktElement.NumPhases()

            # get bus coordinates
            x = self.dss.Bus.X()
            y = self.dss.Bus.Y()

            # build dictionary with required data for visualization    
            bus = {
                "uid": bn,
                "x": x,
                "y": y,
                "nodes": num_nodes,
                "phases": num_phases, 
            }

            # get bus nodes/terminals
            nodes = self.dss.Bus.Nodes()

            # create voltage magnitude container for each bus-terminal combination
            for node in nodes:
                bus[f"vm.{node}"] = []

            # append to container
            self.buses.append(bus)

    def read_bus_voltages(self):

        for bus in self.buses:

            # set active bus
            self.dss.Circuit.SetActiveBus(bus["uid"])

            # get current bus voltages
            voltages = self.dss.Bus.VMagAngle()

            # get voltage magnitudes
            vm = voltages[0::2] 

            # get bus nodes/terminals
            terminals = self.dss.Bus.Nodes()

            # create voltage magnitude container for each bus-terminal combination
            for i, node in enumerate(terminals):
                bus[f"vm.{node}"].append(vm[i]) 

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

