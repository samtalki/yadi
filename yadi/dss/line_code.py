
import numpy as np
import pandas as pd
import yadi.dss.load as load 
import os


class DSS_LineCode(load.DSS_Load):

    def __init__(self, redirects, precompile, verbose=False):
        """"
        Class for handling line codes in OpenDSS.

        """

        super().__init__(redirects, redirects, precompile)