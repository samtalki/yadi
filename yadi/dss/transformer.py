import math

import pandas as pd

import yadi.dss.line as line


class DSS_Transformer(line.DSS_Line):
    def __init__(self, redirects, precompile, verbose=False):
        """ "
        Class for handling Transformers in OpenDSS.

        """
        super().__init__(redirects, redirects, precompile)

    def get_regulators(self):

        # initialize line container
        self.controlled_xfmrs = []

        # set first xfmr as active
        reg_idx, reg = 0, self.dss.RegControls.First()

        while reg:
            # get name of controlled xfmr
            tn = self.dss.RegControls.Transformer()

            # push name to list
            self.controlled_xfmrs.append(tn)

            # move to next regulator
            reg = self.dss.RegControls.Next()
            reg_idx += 1  # increment index

    def create_xfmrs(self):

        # define regulator structure
        self.get_regulators()

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

            # winding reactance (in percentage)
            xhl = self.dss.Transformers.Xhl()

            # winding KVA rating
            s_base = self.dss.Transformers.kVA()

            # wye primitive
            yprim = self.dss.CktElement.YPrim()

            # get number of nodes including reference
            n = len(self.dss.CktElement.NodeOrder())

            # per winding quantities
            kV, R = self.get_perWindingQuantities()

            for ni, nj in zip(f_bus_nodes, t_bus_nodes):
                nodes.append(f"{ni}-{nj}")

            # build dictionary with required data for visualization
            xfmr = {
                "uid": tn,
                "transformer": True,
                "nodes": nodes,
                "source": f_bus_name,
                "target": t_bus_name,
                "R": R,
                "xhl": xhl,
                "s_base": s_base,
                "v_base": kV,
                "yprim": yprim,
                "n": n,
                "ir_ij": {},
                "ii_ij": {},
                "ir_ji": {},
                "ii_ji": {},
                "pij": {},
                "pji": {},
                "qij": {},
                "qji": {},
            }

            # create voltage magnitude container for each xfmr-terminal combination
            for node in nodes:
                xfmr["ir_ij"][f"{node}"] = []
                xfmr["ii_ij"][f"{node}"] = []
                xfmr["ir_ji"][f"{node}"] = []
                xfmr["ii_ji"][f"{node}"] = []
                xfmr["pij"][f"{node}"] = []
                xfmr["pji"][f"{node}"] = []
                xfmr["qij"][f"{node}"] = []
                xfmr["qji"][f"{node}"] = []

            # append to container
            self.branches.append(xfmr)

            xfmr = self.dss.Transformers.Next()
            xfmr_idx += 1  # increment index

    def get_perWindingQuantities(self):

        # winding voltage bases
        num_windings = self.dss.Transformers.NumWindings()

        kV = []
        R = 0
        for i in range(num_windings):
            # activate winding
            self.dss.Transformers.Wdg(i + 1)

            # winding voltage base
            kV.append(self.dss.Transformers.kV())

            # winding resistance (in percentage)
            R += self.dss.Transformers.R()

        return kV, R

    def read_xfmr_power(self):

        for xfmr in self.branches:
            if xfmr["transformer"]:
                # get xfmr uid
                uid = xfmr["uid"]

                # set active xfmr
                self.dss.Circuit.SetActiveElement(f"Transformer.{uid}")
                ir = [
                    i
                    for i in self.dss.CktElement.Currents()[0::2]
                    if not math.isclose(i, 0.0, rel_tol=1e-20)
                ]
                ii = [
                    i
                    for i in self.dss.CktElement.Currents()[1::2]
                    if not math.isclose(i, 0.0, rel_tol=1e-20)
                ]

                # get xfmr active powers
                p = [
                    i
                    for i in self.dss.CktElement.Powers()[0::2]
                    if not math.isclose(i, 0.0, rel_tol=1e-20)
                ]
                q = [
                    i
                    for i in self.dss.CktElement.Powers()[1::2]
                    if not math.isclose(i, 0.0, rel_tol=1e-20)
                ]

                # get number of nodes
                nodes = xfmr["nodes"]

                # create voltage magnitude container for each xfmr-terminal combination
                for n, node in enumerate(nodes):
                    xfmr["ir_ij"][f"{node}"].append(ir[n])
                    xfmr["ii_ij"][f"{node}"].append(ii[n])
                    xfmr["ir_ji"][f"{node}"].append(ir[int(len(p) / 2) + n])
                    xfmr["ii_ji"][f"{node}"].append(ii[int(len(p) / 2) + n])
                    xfmr["pij"][f"{node}"].append(p[n])
                    xfmr["qij"][f"{node}"].append(q[n])
                    xfmr["pji"][f"{node}"].append(p[int(len(p) / 2) + n])
                    xfmr["qji"][f"{node}"].append(q[int(len(q) / 2) + n])

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
