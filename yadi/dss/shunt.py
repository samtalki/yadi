
import numpy as np
import pandas as pd
import yadi.yadi.dss.transformer as transformer 
import os


class DSS_Shunt(transformer.DSS_Transformer):

    def __init__(self, redirects, precompile, verbose=False):
        """"
        Class for handling shunts in OpenDSS.

        """

        super().__init__(redirects, redirects, precompile)