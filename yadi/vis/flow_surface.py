import matplotlib.pyplot as plt
import numpy as np
import yadi.dss.model as dss_model


class FlowSurface(dss_model.DSS_Data):

    def __init__(self,redirects, verbose=True):
        """
        Flow surface visualization object
        """
        super().__init__(redirects=redirects,verbose=verbose)
        self.Pjk, self.Qjk, self.pjk, self.qjk = None,None,None,None

    def get_flow_surface_point(self,lname):
        self.Pjk,self.Qjk,self.pjk,self.qjk = self.get_line_flows(lname)
        
        

    def plot_line_surface(self,lname):
        self.Pjk,self.Qjk,self.pjk,self.qjk = self.get_line_flows(lname)
    
    # def get_loads(self,):
    #     pass

    # def get_gens(self,):
    #     pass
