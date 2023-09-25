
import numpy as np
import pandas as pd
import yadi.yadi.dss.shunt as shunt 
import os


class DSS_Load(shunt.DSS_Shunt):

    def __init__(self, redirects, precompile, verbose=False):
        """"
        Class for handling loads in OpenDSS.

        """

        super().__init__(redirects, redirects, precompile)