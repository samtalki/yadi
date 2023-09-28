
import numpy as np
import pandas as pd
import yadi.yadi.dss.bus as bus 
import os


class DSS_Line(bus.DSS_Bus):

    def __init__(self, redirects, precompile, verbose=False):
        """"
        Class for handling lines in OpenDSS.

        """

        super().__init__(redirects, redirects, precompile)

    def create_lines(self):

        # initialize line container 
        self.lines = [] 

        # set first line as active 
        self.dss.Lines.First()

        while True:
            # set line as active
            ln = self.dss.Lines.Name()

            # from and to buses
            f_bus = self.dss.Lines.Bus1()
            t_bus = self.dss.Lines.Bus2()

            # build dictionary with required data for visualization    
            line = {
                "uid": ln,
                "f_bus": f_bus.split(".")[0],
                "t_bus": t_bus.split(".")[0],
            }

            # get line phases
            num_phases = self.dss.Lines.Phases()

            # create voltage magnitude container for each line-terminal combination
            for ph in range(num_phases):
                line[f"p_ij.{ph+1}"] = []
                line[f"p_ji.{ph+1}"] = []
                line[f"q_ij.{ph+1}"] = []
                line[f"q_ji.{ph+1}"] = []

            # append to container
            self.lines.append(line)

            if not self.dss.Loads.Next() > 0:
                break

    def read_line_power(self):

        for line in self.lines:

            # get line uid
            uid = line["uid"]

            # set active line
            self.dss.Circuit.SetActiveElement(f"Line.{uid}")

            # get line active powers
            p = self.dss.CktElement.Powers()[0::2]
            q = self.dss.CktElement.Powers()[1::2]

            # get line phases
            num_phases = self.dss.Lines.Phases()

            # create voltage magnitude container for each line-terminal combination
            for ph in range(num_phases):
                line[f"p_ij.{ph+1}"].append(p[ph]) 
                line[f"p_ji.{ph+1}"].append(p[int(len(p)/2) + ph]) 
                line[f"q_ij.{ph+1}"].append(q[ph]) 
                line[f"q_ji.{ph+1}"].append(q[int(len(q)/2) + ph]) 

    def get_lineEAmps(self):
        "Method to extract line emergency amps"
        lines = self.dss.Lines.AllNames()
        thermalLimitDict = dict()
        for line in lines:
            self.dss.Lines.Name(line)
            thermalLimitDict[line] = self.dss.Lines.NormAmps()
        return pd.Series(thermalLimitDict)

    def get_linePerPhaseEAmps(self):
        "Method to extract line emergency amps"
        lines = self.dss.Lines.AllNames()
        thermalLimitDict = dict()
        for line in lines:
            self.dss.Lines.Name(line)
            phases = self.dss.Lines.Phases()
            if phases > 1:
                for ph in range(phases):
                    thermalLimitDict[line + f".{ph + 1}"] = self.dss.Lines.NormAmps()
            else:
                thermalLimitDict[line] = self.dss.Lines.NormAmps()
        return pd.Series(thermalLimitDict)