
import numpy as np
import pandas as pd
import yadi.yadi.dss.line as line 
import os


class DSS_Transformer(line.DSS_Line):

    def __init__(self, redirects, precompile, verbose=False):
        """"
        Class for handling Transformers in OpenDSS.

        """
        super().__init__(redirects, redirects, precompile)

    def create_xfmrs(self):

        # set first xfmr as active 
        xfmr_idx, xfmr = 0, self.dss.Transformers.First()

        while xfmr:
            # set xfmr as active
            tn = self.dss.Transformers.Name()

            # from and to buses
            f_bus_name, t_bus_name = self.dss.CktElement.BusNames()

            # from and to buses
            f_bus_name = f_bus_name.split(".")[0]
            t_bus_name = t_bus_name.split(".")[0]

            # read buses data
            f_bus = self.names_to_buses[f_bus_name]
            t_bus = self.names_to_buses[t_bus_name]

            # nodes
            f_bus_nodes = f_bus["nodes"]
            t_bus_nodes = t_bus["nodes"]
            nodes = []

            for ni, nj in zip(f_bus_nodes, t_bus_nodes):
                nodes.append(f"{ni}-{nj}")

            # build dictionary with required data for visualization    
            xfmr = {
                "uid": tn,
                "Transformer": True,
                "nodes": nodes,
                "source": f_bus_name,
                "target": t_bus_name,
                "pij": {},
                "pji": {},
                "qij": {},
                "qji": {},
            }

            # create voltage magnitude container for each xfmr-terminal combination
            for node in nodes:
                xfmr["pij"][f"{node}"] = []
                xfmr["pji"][f"{node}"] = []
                xfmr["qij"][f"{node}"] = []
                xfmr["qji"][f"{node}"] = []

            # append to container
            self.branches.append(xfmr)

            xfmr = self.dss.Transformers.Next() 
            xfmr_idx += 1 #increment index

    def read_xfmr_power(self):

        for xfmr in self.branches:

            if xfmr["Transformer"]:

                # get xfmr uid
                uid = xfmr["uid"]

                # set active xfmr
                self.dss.Circuit.SetActiveElement(f"Transformer.{uid}")

                # get xfmr active powers
                if "reg" in uid:
                    p = [i for i in self.dss.cktelement_powers()[0::2] if i != 0]
                    q = [i for i in self.dss.cktelement_powers()[1::2] if i != 0]
                else:
                    p = self.dss.CktElement.Powers()[0::2]
                    q = self.dss.CktElement.Powers()[1::2]
                
                # get number of nodes 
                nodes = xfmr["nodes"] 

                # create voltage magnitude container for each xfmr-terminal combination
                for n, node in enumerate(nodes):
                    xfmr["pij"][f"{node}"].append(p[n]) 
                    xfmr["pji"][f"{node}"].append(p[int(len(p)/2) + n]) 
                    xfmr["qij"][f"{node}"].append(q[n]) 
                    xfmr["qji"][f"{node}"].append(q[int(len(q)/2) + n]) 

    def get_trafoEAmps(self):
        "Method to extract transformers emergency amps"
        trafos = self.dss.Transformers.AllNames()
        thermalLimitDict = dict()
        for trafo in trafos:
            self.dss.Transformers.Name(trafo)
            # name = self.dss.CktElement.Name()  # sanity check
            thermalLimitDict[trafo] = self.dss.CktElement.NormalAmps()
        return pd.Series(thermalLimitDict)

    def get_trafoPerPhaseEAmps(self):
        "Method to extract transformers emergency amps"
        trafos = self.dss.Transformers.AllNames()
        thermalLimitDict = dict()
        for trafo in trafos:
            self.dss.Transformers.Name(trafo)
            phases = self.dss.CktElement.NumPhases()
            if phases > 1:
                for ph in range(phases):
                    thermalLimitDict[trafo + f".{ph + 1}"] = self.dss.CktElement.NormalAmps()
            else:
                thermalLimitDict[trafo] = self.dss.CktElement.NormalAmps()
        return pd.Series(thermalLimitDict)