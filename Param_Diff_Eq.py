import tensorflow as tf
import numpy as np

class DiffEq():
    """
    This class defines the different differential
    equations used.
    """

    def __init__(self, diffeq, x, y, dydx, d2ydx2):
        """
        diffeq : name of the differential equation used (ex: diffeq = "first order ode")
        """
        self.diffeq = diffeq
        self.x = x
        self.y = y
        self.dydx = dydx
        self.d2ydx2 = d2ydx2

        if self.diffeq == "trivialbsp":
            if self.x[0].shape[0] == 2:
                self.eq = self.dydx[:,0] -self.y[:,0]* self.x[:,1]
            elif self.x[0].shape[0] == 3:
                self.eq = self.x[:,2]*(self.dydx[:,0] -self.y[:,0]* self.x[:,1]) #Unterschied „*self.x[:,2]“
             
