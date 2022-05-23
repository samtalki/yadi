"""
Model-derived HC computation scheme.
@author: Samuel Talkington
04/03/2022
"""
import numpy as np
import mohca_cl.dss.model as dss_model
import opendssdirect as dss
from numba import jit

class ModelHC(dss_model.DSS_Model):

    def __init__(self,redirects) -> None:
        super().__init__(redirects)

    def get_hc(self):
        """
        Get the vector of hosting capacities for every bus with a load
        """
        loads = self.get_all_elements('loal')
        pass

    def get_max_kw(self,bus,pf):
        """
        Gets the maximum kW for a given bus given a power factor setting pf
        """
        def increment_kw(bus,pf):
            """
            Internal kW incrementing function
            """
            pass
        pass

    @staticmethod
    def get_q_constraints(pf):
        """
        Computes q constraints based on vector of pf settings
        """
        pass