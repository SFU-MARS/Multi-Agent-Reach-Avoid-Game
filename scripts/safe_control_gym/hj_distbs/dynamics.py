import heterocl as hcl
import numpy as np
import math
import imp
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import scipy.integrate
import random

from odp.computeGraphs.CustomGraphFunctions import my_abs
from odp.Grid import Grid  # Utility functions to initialize the problem
from odp.Shapes import *
from odp.Plots import PlotOptions # Plot options
from odp.solver import HJSolver  # Solver core
from odp.solver import TTRSolver
from scipy.integrate import solve_ivp
import argparse


class UAV6D:
    def __init__(self, 
                x=[0, 0, 0, 0, 0, 0], 
                uMin=np.array([-5.3*10**-3, -5.3*10**-3, -1.43*10**-4]), 
                uMax=np.array([5.3*10**-3,  5.3*10**-3,  1.43*10**-4]),
                dims=6, 
                uMode="min", 
                dMode="max"):

        # mode        
        self.uMode = uMode
        self.dMode = dMode

        # state
        self.x = x

        # Control bounds
        self.uMin = uMin
        self.uMax = uMax
    
        # Hanyang: Disturbance bounds, change the magnitude to calculate different distb level value functions
        self.distb_level = 0.0
        self.dMin = self.distb_level * uMin
        self.dMax = self.distb_level * uMax

        # Dimension 
        self.dims = dims

        # Constants in the equations
        self.Ixx = 1.65*10**-5  # moment of inertia from ETH page 57
        self.Iyy = 1.65*10**-5  # moment of inertia
        self.Izz = 2.92*10**-5  # moment of inertia

    def opt_ctrl(self, t, state, spat_deriv):
        uOpt1 = hcl.scalar(0, "uOpt1")
        uOpt2 = hcl.scalar(0, "uOpt2")
        uOpt3 = hcl.scalar(0, "uOpt3")
        # Just create and pass back, even though they're not used
        in4 = hcl.scalar(0, "in4")


        if (self.uMode == "min"):
            with hcl.if_(spat_deriv[3] > 0):
                uOpt1[0] = self.uMin[0]
            with hcl.elif_(spat_deriv[3] < 0):
                uOpt1[0] = self.uMax[0]

            with hcl.if_(spat_deriv[4] > 0):
                uOpt2[0] = self.uMin[1]
            with hcl.elif_(spat_deriv[4] < 0):
                uOpt2[0] = self.uMax[1]
            
            with hcl.if_(spat_deriv[5] > 0):
                uOpt3[0] = self.uMin[2]
            with hcl.elif_(spat_deriv[5] < 0):
                uOpt3[0] = self.uMax[2]

        elif (self.uMode == "max"):
            with hcl.if_(spat_deriv[3] > 0):
                uOpt1[0] = self.uMax[0] 
            with hcl.elif_(spat_deriv[3] < 0):
                uOpt1[0] = self.uMin[0]

            with hcl.if_(spat_deriv[4] > 0):
                uOpt2[0] = self.uMax[1] 
            with hcl.elif_(spat_deriv[4] < 0):
                uOpt2[0] = self.uMin[1] 

            with hcl.if_(spat_deriv[5] > 0):
                uOpt3[0] = self.uMax[2]
            with hcl.elif_(spat_deriv[5] < 0):
                uOpt3[0] = self.uMin[2] 
        
        else:
            raise ValueError("undefined uMode ...")

        return (uOpt1[0], uOpt2[0], uOpt3[0], in4)
    
    
    def opt_dstb(self, t, state, spat_deriv):
        # Graph takes in 4 possible inputs, by default, for now
        dOpt1 = hcl.scalar(0, "dOpt1")
        dOpt2 = hcl.scalar(0, "dOpt2")
        dOpt3 = hcl.scalar(0, "dOpt3")
        # Just create and pass back, even though they're not used
        d4 = hcl.scalar(0, "d4")

        if (self.dMode == "min"):
            with hcl.if_(spat_deriv[3] > 0):
                dOpt1[0] = self.dMax[0] 
            with hcl.elif_(spat_deriv[3] < 0):
                dOpt1[0] = self.dMin[0]

            with hcl.if_(spat_deriv[4] > 0):
                dOpt2[0] = self.dMax[1]
            with hcl.elif_(spat_deriv[4] < 0):
                dOpt2[0] = self.dMin[1] 
            
            with hcl.if_(spat_deriv[5] > 0):
                dOpt3[0] = self.dMax[2]
            with hcl.elif_(spat_deriv[5] < 0):
                dOpt3[0] = self.dMin[2] 

        elif (self.dMode == "max"):
            with hcl.if_(spat_deriv[3] > 0):
                dOpt1[0] = self.dMin[0] 
            with hcl.elif_(spat_deriv[3] < 0):
                dOpt1[0] = self.dMax[0] 

            with hcl.if_(spat_deriv[4] > 0):
                dOpt2[0] = self.dMin[1] 
            with hcl.elif_(spat_deriv[4] < 0):
                dOpt2[0] = self.dMax[1]

            with hcl.if_(spat_deriv[5] > 0):
                dOpt3[0] = self.dMin[2] 
            with hcl.elif_(spat_deriv[5] < 0):
                dOpt3[0] = self.dMax[2] 

        return (dOpt1[0], dOpt2[0], dOpt3[0], d4)
        
    """
    :: Dynamics of 6D full quadrotor, refer to https://www.kth.se/polopoly_fs/1.588039.1600688317!/Thesis%20KTH%20-%20Francesco%20Sabatino.pdf
    """
    def dynamics(self, t, state, uOpt, dOpt):

        phi_dot = hcl.scalar(0, "phi_dot")
        theta_dot = hcl.scalar(0, "theta_dot")
        psi_dot = hcl.scalar(0, "psi_dot")
        p_dot = hcl.scalar(0, "p_dot")
        q_dot = hcl.scalar(0, "q_dot")
        r_dot = hcl.scalar(0, "r_dot")

        phi = state[0] # roll
        theta = state[1] # pitch 
        psi = state[2] # yaw
        p = state[3]
        q = state[4]
        r = state[5]
        tau_x = uOpt[0]
        tau_y = uOpt[1]
        tau_z = uOpt[2]

        # some constants
        I_xx = self.Ixx
        I_yy = self.Iyy
        I_zz = self.Izz
        tau_wx = dOpt[0]
        tau_wy = dOpt[1]
        tau_wz = dOpt[2]

        # state dynamics equation
        phi_dot[0] = p + r * hcl.cos(phi) * (hcl.sin(theta)/hcl.cos(theta)) + q * hcl.sin(phi) * (hcl.sin(theta)/hcl.cos(theta))
        theta_dot[0] = q * hcl.cos(phi) - r * hcl.sin(phi)
        psi_dot[0] = r * hcl.cos(phi)/hcl.cos(theta) + q * hcl.sin(phi)/hcl.cos(theta)

        p_dot[0] = ((I_yy - I_zz)/I_xx) * r * q + ((tau_x+tau_wx)/I_xx)
        q_dot[0] = ((I_zz - I_xx)/I_yy) * p * r + ((tau_y+tau_wy)/I_yy)
        r_dot[0] = ((I_xx - I_yy)/I_zz) * p * q + ((tau_z+tau_wz)/I_zz)

        return (phi_dot[0], theta_dot[0], psi_dot[0], p_dot[0], q_dot[0], r_dot[0])
    
    def Hamiltonian(self, t_deriv, spatial_deriv):
        return t_deriv[0] * spatial_deriv[0] + t_deriv[1] * spatial_deriv[1] + t_deriv[2] * spatial_deriv[2] \
            + t_deriv[3] * spatial_deriv[3] + t_deriv[4] * spatial_deriv[4] + t_deriv[5] * spatial_deriv[5]


    def opt_ctrl_non_hcl(self, t, state, spat_deriv):

        uOpt1, uOpt2, uOpt3 = self.uMax[0], self.uMax[1], self.uMax[2]
        
       
        if self.uMode == "min":
                if spat_deriv[3] > 0:
                    uOpt1 = self.uMin[0]

                if spat_deriv[4] > 0:
                    uOpt2 = self.uMin[1]
                
                if spat_deriv[5] > 0:
                    uOpt3 = self.uMin[2]

        elif (self.uMode == "max"):
                if spat_deriv[3] < 0:
                    uOpt1 = self.uMin[0]

                if spat_deriv[4] < 0:
                    uOpt2 = self.uMin[1]

                if spat_deriv[5] < 0:
                    uOpt3 = self.uMin[2]
            
        else:
                raise ValueError("undefined uMode ...")
        
        #print (spat_deriv[3],spat_deriv[4],spat_deriv[5])
        
        return (uOpt1, uOpt2, uOpt3)


    def opt_dstb_non_hcl(self, t, state, spat_deriv):

        dOpt1,dOpt2,dOpt3 = self.dMax[0], self.dMax[1], self.dMax[2] 
        # add predefined values to avoid "none" type input when spat-deriv = 0

        if (self.dMode == "min"):

                if spat_deriv[3] < 0:
                    dOpt1 = self.dMin[0]

                if spat_deriv[4] < 0:
                    dOpt2 = self.dMin[1]
                
                if spat_deriv[5] < 0:
                    dOpt3 = self.dMin[2]

        elif (self.dMode == "max"):

                if spat_deriv[3] > 0:
                    dOpt1 = self.dMin[0]
                if spat_deriv[4] > 0:
                    dOpt2 = self.dMin[1]
                if spat_deriv[5] > 0:
                    dOpt3 = self.dMin[2]


        return (dOpt1, dOpt2, dOpt3)

    def dynamics_non_hcl(self, t, state, uOpt, dOpt):
        phi_dot, theta_dot, psi_dot, p_dot, q_dot, r_dot = None, None, None, None, None, None


        phi = state[0] # roll
        theta = state[1] # pitch 
        psi = state[2] # yaw
        p = state[3]
        q = state[4]
        r = state[5]
        tau_x = uOpt[0]
        tau_y = uOpt[1]
        tau_z = uOpt[2]

        # some constants
        I_xx = self.Ixx
        I_yy = self.Iyy
        I_zz = self.Izz
        tau_wx = dOpt[0]
        tau_wy = dOpt[1]
        tau_wz = dOpt[2]

        # state dynamics equation
        phi_dot = p + r * np.cos(phi) * np.tan(theta) + q * np.sin(phi) * np.tan(theta)
        theta_dot = q * np.cos(phi) - r * np.sin(phi)
        psi_dot = r * np.cos(phi)/np.cos(theta) + q * np.sin(phi)/np.cos(theta)

        p_dot = ((I_yy - I_zz)/I_xx) * r * q + ((tau_x + tau_wx)/I_xx)
        q_dot = ((I_zz - I_xx)/I_yy) * p * r + ((tau_y + tau_wy)/I_yy)
        r_dot = ((I_xx - I_yy)/I_zz) * p * q + ((tau_z + tau_wz)/I_zz)

        return (phi_dot, theta_dot, psi_dot, p_dot, q_dot, r_dot)



class CartPole4D:
    def __init__(self, x=[0, 0, 0, 0], uMax=10, uMin=-10, dMax=10, dMin=-10, uMode="min", dMode="max", distb_level=0.0):
        self.x = x
        self.uMax = uMax
        self.uMin = uMin
        self.uMode = uMode
        self.dMode = dMode
        # Hanyang: Disturbance bounds, change the magnitude to calculate different distb level value functions
        self.distb_level = distb_level
        self.dMin = self.distb_level * dMin
        self.dMax = self.distb_level * dMax
        # Physical parameters
        self.l = 0.5
        self.mc = 1.0
        self.mp = 0.1
        self.g = 9.8
    
    
    def dynamics(self, t, state, uOpt, dOpt):
        # Reference: https://coneural.org/florian/papers/05_cart_pole.pdf
        # Set of differential equations describing the system
        # x1_dot = x2
        # x2_dot = (uOpt + mp*l*(x4^2*sin(x3) - x4_dot*cost(x2))/(mc + mp) 
        # x3_dot = x4
        # x4_dot = (g*sin(x3) + cos(x3)*(-uOpt - mp*l*x4^2*sin(x3))/(mc + mp))/(l*(4/3 - mp*cos(x3)^2/(mc + mp)))
        x1_dot = hcl.scalar(0, "x1_dot")
        x2_dot = hcl.scalar(0, "x2_dot")
        x3_dot = hcl.scalar(0, "x3_dot")
        x4_dot = hcl.scalar(0, "x4_dot")

        x1_dot[0] = state[1]
        x4_dot[0] = (self.g*hcl.sin(state[2]) + hcl.cos(state[2])*(-uOpt[0] - dOpt[0] - self.mp*self.l*state[3]*state[3]*hcl.sin(state[2])) / (self.mc+ self.mp))/(self.l*(4/3 - self.mp*hcl.cos(state[2])*hcl.cos(state[2])/(self.mc + self.mp)))
        x2_dot[0] = (uOpt[0] + dOpt[0] + self.mp*self.l*(state[3]*state[3]*hcl.sin(state[2]) - x4_dot[0]*hcl.cos(state[2])))/(self.mc + self.mp)
        x3_dot[0] = state[3]
        
        return (x1_dot[0], x2_dot[0], x3_dot[0], x4_dot[0])


    def opt_ctrl(self, t, state, spat_deriv):
        opt_u = hcl.scalar(self.uMax, "opt_u")
        # Just create and pass back, even though they're not used
        in2 = hcl.scalar(0, "in2")
        in3 = hcl.scalar(0, "in3")
        in4 = hcl.scalar(0, "in4")
        u_coefficient = hcl.scalar(0, "u_coefficient")

        u_coefficient = spat_deriv[1]/(self.mc + self.mp) - (spat_deriv[3]*hcl.cos(state[2]))/(self.l*(4/3*(self.mp+self.mc) - self.mp*hcl.cos(state[2])*hcl.cos(state[2])))
        with hcl.if_(u_coefficient >= 0):
            with hcl.if_(self.uMode == "min"):
                opt_u[0] = -opt_u[0]
        with hcl.elif_(u_coefficient < 0):
            with hcl.if_(self.uMode == "max"):
                opt_u[0] = -opt_u[0]
        return (opt_u[0], in2[0], in3[0], in4[0])
    

    def opt_dstb(self, t, state, spat_deriv):
        opt_d = hcl.scalar(self.dMax, "opt_d")
        # Just create and pass back, even though they're not used
        d2 = hcl.scalar(0, "d2")
        d3 = hcl.scalar(0, "d3")
        d4 = hcl.scalar(0, "d4")
        d_coefficient = hcl.scalar(0, "d_coefficient")

        d_coefficient = spat_deriv[1]/(self.mc + self.mp) - (spat_deriv[3]*hcl.cos(state[2]))/(self.l*(4/3*(self.mp+self.mc) - self.mp*hcl.cos(state[2])*hcl.cos(state[2])))
        with hcl.if_(d_coefficient >= 0):
            with hcl.if_(self.dMode == "min"):
                opt_d[0] = -opt_d[0]
        with hcl.elif_(d_coefficient < 0):
            with hcl.if_(self.dMode == "max"):
                opt_d[0] = -opt_d[0]
        return (opt_d[0], d2[0], d3[0], d4[0])
    

    def opt_ctrl_non_hcl(self, t, state, spat_deriv):
        opt_a = self.uMax
        u_coefficient = spat_deriv[1]/(self.mc + self.mp) - (spat_deriv[3]*np.cos(state[2]))/(self.l*(4/3*(self.mp+self.mc) - self.mp*np.cos(state[2])*np.cos(state[2])))

        if u_coefficient >= 0:
            if self.uMode == "min":
                opt_a = -opt_a
        else:
            if self.uMode == "max":
                opt_a = -opt_a

        return opt_a
    

    def opt_dstb_non_hcl(self, t, state, spat_deriv):
        opt_d = self.dMax
        d_coefficient = spat_deriv[1]/(self.mc + self.mp) - (spat_deriv[3]*np.cos(state[2]))/(self.l*(4/3*(self.mp+self.mc) - self.mp*np.cos(state[2])*np.cos(state[2])))

        if d_coefficient >= 0:
            if self.dMode == "min":
                opt_d = -opt_d
        else:
            if self.dMode == "max":
                opt_d = -opt_d

        return opt_d
    
    def dynamics_non_hcl(self, t, state, uOpt, dOpt):
        x1_dot, x2_dot, x3_dot, x4_dot = None, None, None, None

        x1 = state[0]
        x2 = state[1]
        x3 = state[2]
        x4 = state[3]
        u = uOpt[0]
        d = dOpt[0]

        x1_dot = x2
        x4_dot = (self.g*np.sin(x3) + np.cos(x3)*(-u - d - self.mp*self.l*x4*x4*np.sin(x3)) / (self.mc+ self.mp))/(self.l*(4/3 - self.mp*np.cos(x3)*np.cos(x3)/(self.mc + self.mp)))
        x2_dot = (u + d + self.mp*self.l*(x4*x4*np.sin(x3) - x4_dot*np.cos(x3)))/(self.mc + self.mp)
        x3_dot = x4

        return (x1_dot, x2_dot, x3_dot, x4_dot)