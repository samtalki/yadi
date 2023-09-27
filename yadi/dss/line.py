
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

    def write_PMD_line(self):

        # initialize line structure
        self.line = {}

        # get all bus names
        line_names = self.dss.Lines.AllNames()

        # main loop
        for ln in line_names:

            # set line as active
            self.dss.Lines.Name(ln)

            # buses
            f_bus = self.dss.Lines.Bus1
            t_bus = self.dss.Lines.Bus2

            if not "sw" in ln:
                # create structure
                self.line[ln] = {
                    "f_bus"         : f_bus,
                    "t_bus"         : t_bus,
                    "f_connections" : self.bus[f_bus]["terminals"],
                    "t_connections" : self.bus[t_bus]["terminals"],
                    "linecode"      : self.dss.Lines.LineCode(),
                    "length"        : self.dss.Lines.Length(),
                    "source_id"     : f"line.{ln}",
                    "status"        : "ENABLED",
                    "time_series"   : {},
                }
            else:

                rs = self.dss.Lines.RMatrix()
                xs = self.dss.Lines.XMatrix()

                # create structure
                self.line[ln] = {
                    "f_bus"         : f_bus,
                    "t_bus"         : t_bus,
                    "f_connections" : self.bus[f_bus]["terminals"],
                    "t_connections" : self.bus[t_bus]["terminals"],
                    "rs"            : rs,
                    "xs"            : xs,
                    "g_fr"          : np.zeros_like(rs),
                    "b_fr"          : np.zeros_like(rs),
                    "g_to"          : np.zeros_like(rs),
                    "b_to"          : np.zeros_like(rs),
                    "length"        : self.dss.Lines.Length(),
                    "source_id"     : f"line.{ln}",
                    "status"        : "ENABLED",
                    "time_series"   : {},
                }

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