import pandas as pd

import yadi.dss.shunt as shunt


class DSS_Load(shunt.DSS_Shunt):
    def __init__(self, redirects, precompile: bool = True, verbose: bool = True) -> None:
        super().__init__(redirects, precompile=precompile, verbose=verbose)

    def create_loads(self):

        # initialize load container
        self.loads = []

        # set first load as active
        load_idx, load = 0, self.dss.Loads.First()

        while load:
            # set load as active
            ln = self.dss.Loads.Name()

            # build dictionary with required data for visualization
            load = {
                "uid": ln,
                "p": [],
                "q": [],
            }

            # append to container
            self.loads.append(load)

            load = self.dss.Loads.Next()
            load_idx += 1  # increment index

    def read_load_power(self):

        for load in self.loads:
            # get load uid
            uid = load["uid"]

            # set active load
            self.dss.Loads.Name(uid)

            # Kw and Kvar of the load
            p = self.dss.Loads.kW()
            q = self.dss.Loads.kvar()

            # get power
            load["p"].append(p)
            load["q"].append(q)

    def get_loadBusAndVln(self):
        "Method to extract load buses from a feeder"
        loadBusDict = dict()
        loadVlnDict = dict()
        loads = self.dss.Loads.AllNames()
        for load in loads:
            # set load as active
            self.dss.Loads.Name(load)
            # name = self.dss.CktElement.Name()  # sanity check
            # get bus name
            buses = self.dss.CktElement.BusNames()
            bus = buses[0]
            # save name
            loadBusDict[load] = bus
            loadVlnDict[load] = self.dss.Loads.kV()
        return pd.Series(loadBusDict), pd.Series(loadVlnDict)
