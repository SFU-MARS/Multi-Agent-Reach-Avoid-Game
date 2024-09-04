import heterocl as hcl
import numpy as np
import math
from odp.computeGraphs.CustomGraphFunctions import my_abs

class UAV_6D:
    ##  _init_ needs modification based on UAV dynamics.
    # Adjusted dmin and dmax to the correct size: 2. (The reason it had four was because it had two control bounds and two planner bounds). 
    def __init__(self, x=[0, 0, 0, 0, 0, 0], uMin=np.array([-6, -15]), uMax=np.array([6, 15]), pMin=np.array([-0.2, -0.15]), pMax=np.array([0.2, 0.15]),
                 dMin=np.array([-0.02, -0.02]), dMax=np.array([0.02, 0.02]),
                 dims=6, uMode="min", dMode="max"):
        
        self.x = x
        self.uMode = uMode
        self.dMode = dMode

        # Object properties
        self.x    = x

        # Control bounds
        self.uMin = uMin
        self.uMax = uMax

        # Planner bounds maybe is optional? 
        #self.pMin = pMin
        #self.pMax = pMax

        # Disturbance bounds
        self.dMin = dMin
        self.dMax = dMax

        self.dims = dims


        # Constants in the equations 
        self.CD_v   = 1     # translation drag coefficient
        self.m      = 10    # quadrotor's mass
        self.g      = 9.81  # gravity
        self.Iyy    = 1     # moment of inertia 
        self.CD_phi = 1     # rotational drag coefficient 
        self.l      = 1     # half-length
         

    def opt_ctrl(self, t, state, spat_deriv): # Two controls -> Two Thursters
        uOpt1 = hcl.scalar(0, "uOpt1")
        uOpt2 = hcl.scalar(0, "uOpt2")
        # Need to have two spare ones to fit the size. Just create and pass back, even though they're not used
        in3   = hcl.scalar(0, "in3")
        in4   = hcl.scalar(0, "in4")

        #parSum1 = hcl.scalar(0, "parSum1")
        #parSum2 = hcl.scalar(0, "parSum2")

        with hcl.if_(self.uMode == "min"):
            with hcl.if_(spat_deriv[0] > 0): # If some derivative is larger than zero, then u1 is negative umax? 
                uOpt1[0] = self.uMin[0]
            with hcl.elif_(spat_deriv[0] < 0):
                uOpt1[0] = self.uMax[0]

            with hcl.if_(spat_deriv[1] > 0):
                uOpt2[0] = self.uMin[1]
            with hcl.elif_(spat_deriv[1] < 0):
                uOpt2[0] = self.uMax[1]

        return (uOpt1[0], uOpt2[0], in3, in4)


    def opt_dstb(self, t, state, spat_deriv): # Two Input with Two distutbance? I guess the amount of dis should be the same as input
        dOpt1 = hcl.scalar(0, "dOpt1")
        dOpt2 = hcl.scalar(0, "dOpt2")
        d3 = hcl.scalar(0, "d3")
        d4 = hcl.scalar(0, "d4")

        with hcl.if_(self.dMode == "max"):
            with hcl.if_(spat_deriv[0] > 0):
                dOpt1[0] = self.dMin[0]
            with hcl.elif_(spat_deriv[0] < 0):
                dOpt1[0] = self.dMax[0]

            with hcl.if_(spat_deriv[1] > 0):
                dOpt2[0] = self.dMin[1]
            with hcl.elif_(spat_deriv[1] < 0):
                dOpt2[0] = self.dMax[1]


        with hcl.elif_(self.dMode == "min"):
            with hcl.if_(spat_deriv[0] > 0):
                dOpt1[0] = self.dMax[0]
            with hcl.elif_(spat_deriv[0] < 0):
                dOpt1[0] = self.dMin[0]

            with hcl.if_(spat_deriv[1] > 0):
                dOpt2[0] = self.dMax[1]
            with hcl.elif_(spat_deriv[1] < 0):
                dOpt2[0] = self.dMin[1]

        return (dOpt1[0], dOpt2[0], d3, d4)


        # state questions for UAVs
        # x1_dot = x2
        # x2_dot = (-1/m)*(CD_v*x2)+(T1/m)*sin(x5)+(T2/m)*sin(x5)
        # x3_dot = x4
        # x4_dot = (-1/m)*(m*g+CD_v*x4)+(T1/m)*cos(x5)+(T2/m)*cos(x5)
        # x5_dot = x6
        # x6_dot = (-1/Iyy)*CD_phi*omega+(l/Iyy)*T1-(l/Iyy)*T2

    def dynamics(self, t, state, uOpt, dOpt): # uOpt is control: T1 -> u1, T2 -> u2
        # States in system
        # x1=x, x3=z, x5=phi    -- it denotes the planar positional coordinates and pitch angle
        # x2=Vx, x4=Vz, x6=omega     -- it denotes their time derivatives resepctively 

        x1_dot = hcl.scalar(0, "x1_dot")
        x2_dot = hcl.scalar(0, "x2_dot")
        x3_dot = hcl.scalar(0, "x3_dot")
        x4_dot = hcl.scalar(0, "x4_dot")
        x5_dot = hcl.scalar(0, "x5_dot")
        x6_dot = hcl.scalar(0, "x6_dot")

        x1_dot[0] = state[1]
        x2_dot[0] = (-1/self.m)*(self.CD_v*state[1])+(uOpt[0]/self.m)*hcl.sin(state[4])+(uOpt[1]/self.m)*hcl.sin(state[4])
        x3_dot[0] = state[3]
        x4_dot[0] = (-1/self.m)*(self.m*self.g+self.CD_v*state[3])+(uOpt[0]/self.m)*hcl.cos(state[4])+(uOpt[1]/self.m)*hcl.cos(state[4])
        x5_dot[0] = state[5]
        x6_dot[0] = (-1/self.Iyy)*self.CD_phi*state[5]+(self.l/self.Iyy)*uOpt[0]-(self.l/self.Iyy)*uOpt[1]
        return (x1_dot[0], x2_dot[0], x3_dot[0], x4_dot[0], x5_dot[0], x6_dot[0])

    def Hamiltonian(self, t_deriv, spatial_deriv):
        return t_deriv[0] * spatial_deriv[0] + t_deriv[1] * spatial_deriv[1] + t_deriv[2] * spatial_deriv[2] + t_deriv[3] * spatial_deriv[3] \
                + t_deriv[4] * spatial_deriv[4] + t_deriv[5] * spatial_deriv[5]
