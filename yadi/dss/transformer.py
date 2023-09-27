
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

    def write_PMD_transformer(self):

        # initialize line structure
        self.transformer = {}

        # get all bus names
        transformer_names = self.dss.Transformers.AllNames()

        # main loop
        for tn in transformer_names:

            # set line as active
            self.dss.Transformers.Name(f"Transformer.{tn}")


            # number of phases from transformer
            num_phases = self.dss.CktElement.NumPhases()

            # buses
            f_bus, t_bus = self.dss.CktElement.BusNames()
            f_bus = f_bus.split(".")[0]
            t_bus = t_bus.split(".")[0]

            # connections
            connections = [self.bus[f_bus]["terminals"], self.bus[t_bus]["terminals"]]

            num_windings = self.dss.Transformers.NumWindings()

            # configuration
            configuration = []

            # set tap ratio for each winding 
            tm_set = []

            # KV rating for each winding
            vm_nom = []

            # KVA rating for each winding
            sm_nom = []

            for i in range(num_windings):
                # set active winding
                self.dss.Transformers.Wdg(i + 1)

                # write configuration
                if self.Transformers.IsDelta() == True:
                   configuration.append("DELTA") 
                else:
                   configuration.append("WYE") 

                # write tap ratio
                tm_set.append(np.ones(num_phases))

                # write KV rating
                vm_nom.append(self.dss.Transformers.kV())

                # write KVA rating
                sm_nom.append(self.dss.Transformers.kVA())
            
            xfmrcode = self.dss.Transformers.XfmrCode()

            # list of short circuit reactances between each pair of windings (upper triangular matrix)
            xsc = np.zeros((num_windings, (num_windings - 1) // 2))

            # active power loss due to resistance of each winding
            rw = np.zeros(num_windings)

            # nominal tap ratio for the transformer
            tm_nom = np.ones(num_windings)

            if not "reg" in tn:
                # create normal transformer structure
                self.transformer[tn] = {
                    "bus"           : [f_bus, t_bus],
                    "connections"   : connections,
                    "configuration" : configuration,
                        "xsc"       : xsc,
                        "rw"        : rw,
                    "cmag"          : 0.0,
                    "noloadloss"    : 0.0,
                    "tm_nom"        : tm_nom,
                    "tm_set"        : tm_set,
                    "polarity"      : [1, 1],
                    "vm_nom"        : vm_nom,
                    "sm_nom"        : sm_nom,
                    "source_id"     : tn,
                    "status"        : "ENABLED",
                    "time_series"   : {},
                }
            else:
                # create regulator transformer structure

                regulator_names = self.dss.RegControls.AllNames()
                for rn in regulator_names:

                    # activate regulator by name
                    self.dss.RegControls.Name(rn)

                    # trnasformer associated to this regulator
                    tn_rn = self.dss.RegControls.Transformer()

                    if tn == tn_rn:

                        # primary current rating
                        ctprim = self.dss.RegControls.CTPrimary()

                        # resistnace setting for line drop compensator
                        r = self.dss.RegControls.ForwardR()

                        # reactance setting for line drop compensator
                        x = self.dss.RegControls.ForwardX()

                        # voltage ratio of potential transformer 
                        ptratio = self.dss.RegControls.PTRatio()

                        # voltage bandwidth
                        band = self.dss.RegControls.FowardBand()

                        # voltage setpoint
                        vreg = self.dss.RegControls.ForwardVreg() 

                        # create structure
                        self.transformer[tn] = {
                            "bus"           : [f_bus, t_bus],
                            "connections"   : connections,
                            "configuration" : configuration,
                                "xsc"       : xsc,
                                "rw"        : rw,
                            "cmag"          : 0.0,
                            "noloadloss"    : 0.0,
                            "tm_nom"        : tm_nom,
                            "tm_set"        : tm_set,
                            "polarity"      : [1, 1],
                            "vm_nom"        : vm_nom,
                            "sm_nom"        : sm_nom,
                            "controls"      : {"ctprim" :ctprim, 
                                            "x"      :x, 
                                            "r"      :r, 
                                            "ptratio":ptratio, 
                                            "band"   :band, 
                                            "vreg"   :vreg},
                            "source_id"     : tn,
                            "status"        : "ENABLED",
                            "time_series"   : {},
                        }

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