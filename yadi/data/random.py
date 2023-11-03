import yadi.dss.model as model
import opendssdirect as dss
import numpy as np

class DSS_Random(model.DSS_Data):
    """
    Generates random data for an OpenDSS model
    Uses the existing loadshapes to run a short-term time-series, and random perturbs the load 
    """

    def __init__(self, redirects, verbose=True, precompile=True):
        super().__init__(redirects, verbose, precompile)