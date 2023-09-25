
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