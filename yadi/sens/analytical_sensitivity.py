"""
OpenDSS Sensitivity-Based Linear Approximation Model
Perturb and observe to form a linear approximation of the power flow equations (first order taylor expansion)
@author: Samuel Talkington
MIT License
October 6th, 2021
"""

import numpy as np
import yadi.dss.model as model
import pandas as pd
import warnings