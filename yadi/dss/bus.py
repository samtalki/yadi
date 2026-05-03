import yadi.dss.load_shape as load_shape


class DSS_Bus(load_shape.DSS_LoadShape):
    def __init__(self, redirects, precompile, verbose=False):
        """ "
        Class for handling buses in OpenDSS.

        """

        super().__init__(redirects, redirects, precompile)

    def create_buses(self):

        # initialize bus container
        self.buses = []
        self.names_to_buses = {}

        # iterate over all bus names
        for bn in self.dss.Circuit.AllBusNames():
            # set active bus
            self.dss.Circuit.SetActiveBus(bn)

            # get bus coordinates
            x = self.dss.Bus.X()
            y = self.dss.Bus.Y()

            # base LN voltage base
            kv_base = self.dss.Bus.kVBase()

            # get bus nodes/terminals
            nodes = self.dss.Bus.Nodes()

            # build dictionary with required data for visualization
            bus = {
                "uid": bn,
                "x": x,
                "y": y,
                "nodes": nodes,
                "kV_base": kv_base,
                "vm": {},
                "va": {},
            }

            # create voltage magnitude container for each bus-terminal combination
            for node in nodes:
                bus["vm"][f"{node}"] = []
                bus["va"][f"{node}"] = []

            # append to container
            self.buses.append(bus)
            self.names_to_buses[bn] = bus

    def read_bus_voltages(self):

        for bus in self.buses:
            # get name
            uid = bus["uid"]

            # set active bus
            self.dss.Circuit.SetActiveBus(uid)

            # get current bus voltages
            voltages = self.dss.Bus.VMagAngle()

            # get voltage magnitudes and angles
            vm = voltages[0::2]
            va = voltages[1::2]

            # get bus nodes/terminals
            terminals = self.dss.Bus.Nodes()

            # create voltage magnitude container for each bus-terminal combination
            for i, node in enumerate(terminals):
                bus["vm"][f"{node}"].append(vm[i])
                bus["va"][f"{node}"].append(va[i])

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
                "terminals": terminals,
                "grounded": grounded,
                "rg": rg,
                "xg": xg,
                "status": "ENABLED",
                "time_series": {},
            }
