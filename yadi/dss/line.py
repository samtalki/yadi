import pandas as pd

import yadi.dss.bus as bus


class DSS_Line(bus.DSS_Bus):
    def __init__(self, redirects, precompile: bool = True, verbose: bool = True) -> None:
        super().__init__(redirects, precompile=precompile, verbose=verbose)

    def create_lines(self):

        # initialize line container
        self.branches = []

        # set first line as active
        line_idx, line = 0, self.dss.Lines.First()

        while line:
            # set line as active
            ln = self.dss.Lines.Name()

            # from and to buses
            f_bus_name = self.dss.Lines.Bus1().split(".")[0]
            t_bus_name = self.dss.Lines.Bus2().split(".")[0]

            # read buses data
            f_bus = self.names_to_buses[f_bus_name]
            t_bus = self.names_to_buses[t_bus_name]

            # wye primitive
            yprim = self.dss.CktElement.YPrim()

            # get number of nodes including reference
            n = len(self.dss.CktElement.NodeOrder())

            # Resistance Matrix [ohm/unit length]
            Rm = self.dss.Lines.RMatrix()

            # Reactance Matrix [ohm/unit length]
            Xm = self.dss.Lines.XMatrix()

            # nodes (active nodes from each bus in OpenDss)
            f_bus_nodes = f_bus["nodes"]
            t_bus_nodes = t_bus["nodes"]
            nodes = []
            for ni, nj in zip(f_bus_nodes, t_bus_nodes):
                nodes.append(f"{ni}-{nj}")

            # build dictionary with required data for visualization
            line = {
                "uid": ln,
                "transformer": False,
                "length": self.dss.Lines.Length(),
                "nodes": nodes,
                "source": f_bus_name,
                "target": t_bus_name,
                "yprim": yprim,
                "Rm": Rm,
                "Xm": Xm,
                "n": n,
                "pij": {},
                "pji": {},
                "qij": {},
                "qji": {},
                "ir_ij": {},
                "ii_ij": {},
                "ir_ji": {},
                "ii_ji": {},
            }

            # create voltage magnitude container for each line-terminal combination
            for node in nodes:
                line["pij"][f"{node}"] = []
                line["pji"][f"{node}"] = []
                line["qij"][f"{node}"] = []
                line["qji"][f"{node}"] = []
                line["ir_ij"][f"{node}"] = []
                line["ii_ij"][f"{node}"] = []
                line["ir_ji"][f"{node}"] = []
                line["ii_ji"][f"{node}"] = []

            # append to container
            self.branches.append(line)

            line = self.dss.Lines.Next()
            line_idx += 1

    def read_line_power(self):

        for line in self.branches:
            if not line["transformer"]:
                # get line uid
                uid = line["uid"]

                # set active line
                self.dss.Circuit.SetActiveElement(f"Line.{uid}")

                # openDSS currents
                ir = self.dss.CktElement.Currents()[0::2]
                ii = self.dss.CktElement.Currents()[1::2]

                # get line active powers
                p = self.dss.CktElement.Powers()[0::2]
                q = self.dss.CktElement.Powers()[1::2]

                # get line phases
                nodes = line["nodes"]

                # create voltage magnitude container for each line-terminal combination
                for n, node in enumerate(nodes):
                    line["pij"][f"{node}"].append(p[n])
                    line["pji"][f"{node}"].append(p[int(len(p) / 2) + n])
                    line["qij"][f"{node}"].append(q[n])
                    line["qji"][f"{node}"].append(q[int(len(q) / 2) + n])
                    line["ir_ij"][f"{node}"].append(ir[n])
                    line["ii_ij"][f"{node}"].append(ii[n])
                    line["ir_ji"][f"{node}"].append(ir[int(len(p) / 2) + n])
                    line["ii_ji"][f"{node}"].append(ii[int(len(p) / 2) + n])

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
