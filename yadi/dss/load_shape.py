import os
from calendar import monthrange

import numpy as np
import pandas as pd

import yadi.dss.monitor as monitor


class DSS_LoadShape(monitor.DSS_Monitor):
    def __init__(self, redirects, precompile, verbose=False):
        """ "
        Class for handling LoadShapes in OpenDSS.

        """
        super().__init__(redirects, redirects, precompile)

    def set_loadshape(self, loadshape_path, loadshape_name="loadshape1"):
        """
        Sets a loadshape for all loads
        """
        self.dss.Text.Command(f"Redirect {loadshape_path}")
        self.dss.Text.Command(f"batchedit load..*  yearly={loadshape_name} ")  # change all loads

    def setAllLoadShapes(self, kwLoadShapes, kvarLoadShapes):
        "Method to modify loadShapes from a DSS file"
        loadShapeNames = self.dss.LoadShape.AllNames()
        for n, loadShapeName in enumerate(loadShapeNames):
            if loadShapeName == "default":
                continue
            # extract profiles
            if loadShapeName in kwLoadShapes.columns:
                Pmult = tuple(kwLoadShapes.loc[:, loadShapeName].values)
            else:
                Pmult = None
            if loadShapeName in kvarLoadShapes.columns:
                Qmult = tuple(kvarLoadShapes.loc[:, loadShapeName].values)
            else:
                Qmult = None

            # actually modify load
            if (Qmult is not None) and (Pmult is not None):
                self.__modifyLoadShapePQ(Pmult, Qmult, loadShapeName)
            else:  # not a load
                self.dss.LoadShape.Name(loadShapeName)
                Pmult = self.dss.LoadShape.PMult()
                self.__modifyLoadShapeP(tuple(Pmult), loadShapeName)
            # check load modification
            # self.dss.LoadShape.Name(loadShapeName)
            # Pmult = self.dss.LoadShape.PMult()
            # Qmult = self.dss.LoadShape.QMult()

    def __modifyLoadShapeP(self, Pmult, name):
        self.dss.Text.Command(
            f"edit loadshape.{name} npts={len(Pmult)} mult={Pmult} UseActual=False"
        )

    def __modifyLoadShapePQ(self, Pmult, Qmult, name):
        self.dss.Text.Command(
            f"edit loadshape.{name} npts={len(Pmult)} mult={Pmult} qmult={Qmult} UseActual=True"
        )

    def __extract_loadShapes(self):
        """extract loadshapes from dss file"""
        loadShapeNames = self.dss.LoadShape.AllNames()
        kwLoadShape_dict = dict()
        kvarLoadShape_dict = dict()
        for n, loadShapeName in enumerate(loadShapeNames):
            if loadShapeName == "default":
                continue
            # set active loadshape using its name
            self.dss.LoadShape.Name(loadShapeName)
            # checkName = self.dss.LoadShape.Name()
            # get Pmult and Qmult
            Pmult = self.dss.LoadShape.PMult()
            Qmult = self.dss.LoadShape.QMult()
            # Pmult_len = len(Pmult)
            if len(Pmult) != 1:
                kwLoadShape_dict[loadShapeName] = np.asarray(Pmult)
            if len(Qmult) != 1:
                kvarLoadShape_dict[loadShapeName] = np.asarray(Qmult)
        kwLoadShapes = pd.DataFrame().from_dict(kwLoadShape_dict)
        kvarLoadShapes = pd.DataFrame().from_dict(kvarLoadShape_dict)
        return (kwLoadShapes, kvarLoadShapes)

    def __split_loadShapes(self, loadShapes):
        """split loadshapes by months"""
        kwLoadShapes = loadShapes[0]
        kvarLoadShapes = loadShapes[1]
        skipRows = 0
        monthsForIter = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
        for it, monthIter in enumerate(monthsForIter):
            daysInMonth = monthrange(2019, int(monthIter))
            hoursInMonth = 24 * daysInMonth[1]  # number of hours in month
            # define the name of the monthly demand file
            kwDemand_path = os.path.join(self.monthlyDemand_dir, f"month_{monthIter}_kwProfile.pkl")
            kvarDemand_path = os.path.join(
                self.monthlyDemand_dir, f"month_{monthIter}_kvarProfile.pkl"
            )
            if not os.path.isfile(kwDemand_path):
                if monthIter == "12":
                    kwDemand = kwLoadShapes.iloc[skipRows:, :]
                    kvarDemand = kvarLoadShapes.iloc[skipRows:, :]
                    # create index date range
                    time = pd.date_range(start=f"2019-{monthIter}-01", end="2020-01-01", freq="h")
                else:
                    kwDemand = kwLoadShapes.iloc[skipRows : skipRows + hoursInMonth, :]
                    kvarDemand = kvarLoadShapes.iloc[skipRows : skipRows + hoursInMonth, :]
                    # create index date range
                    time = pd.date_range(
                        start=f"2019-{monthIter}-01",
                        end=f"2019-{monthsForIter[it + 1]}-01",
                        freq="h",
                    )
                # transform string index into datetime index
                kwDemand.index = pd.to_datetime(time[: len(kwDemand)])
                kvarDemand.index = pd.to_datetime(time[: len(kvarDemand)])
                # call method for processing series
                kwDemand.to_pickle(kwDemand_path)
                kvarDemand.to_pickle(kvarDemand_path)
                skipRows += hoursInMonth
            else:
                skipRows += hoursInMonth

    def __load_LoadShapePerMonth(self, month):
        """load demand per month"""
        # extract demand
        kwDemand_path = os.path.join(self.monthlyDemand_dir, f"month_{month}_kwProfile.pkl")
        kvarDemand_path = os.path.join(self.monthlyDemand_dir, f"month_{month}_kvarProfile.pkl")
        if not os.path.isfile(kwDemand_path):
            self.__split_loadShapes(self.__extract_loadShapes())
        kwDemand = pd.read_pickle(kwDemand_path)
        kvarDemand = pd.read_pickle(kvarDemand_path)
        # self.simulation_steps = len(kwDemand)
        # self.kwDemand = kwDemand
        # self.kvarDemand = kvarDemand
        return kwDemand, kvarDemand
