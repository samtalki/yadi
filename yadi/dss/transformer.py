
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
            f_bus, t_bus = self.dss.CktElement.BusNames()

            # get xfmr phases
            num_phases = self.dss.CktElement.NumPhases()

            # build dictionary with required data for visualization    
            xfmr = {
                "uid": tn,
                "Transformer": True,
                "phases": num_phases,
                "source": f_bus.split(".")[0],
                "target": t_bus.split(".")[0],
            }

            # create voltage magnitude container for each xfmr-terminal combination
            for ph in range(num_phases):
                xfmr[f"p_ij.{ph+1}"] = []
                xfmr[f"p_ji.{ph+1}"] = []
                xfmr[f"q_ij.{ph+1}"] = []
                xfmr[f"q_ji.{ph+1}"] = []

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

                # get xfmr phases
                num_phases = self.dss.CktElement.NumPhases()

                # create voltage magnitude container for each xfmr-terminal combination
                for ph in range(num_phases):
                    xfmr[f"p_ij.{ph+1}"].append(p[ph]) 
                    xfmr[f"p_ji.{ph+1}"].append(p[int(len(p)/2) + ph]) 
                    xfmr[f"q_ij.{ph+1}"].append(q[ph]) 
                    xfmr[f"q_ji.{ph+1}"].append(q[int(len(q)/2) + ph]) 

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