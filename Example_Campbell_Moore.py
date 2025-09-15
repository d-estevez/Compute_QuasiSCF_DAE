# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 13:49:41 2025

@author: schwarz
"""

import numpy as np
import sympy as sp

from Generate_QuasiSCF_Steps import generate_QuasiSCF_DAE,latex_matrix

def Campbell_Moore():
    """
    Example from S. L. Campbell and E. Moore. 
    Constraint preserving integrators for general nonlinear higher index DAEs. 
    Numer. Math., 69(4):383–399, 1995.
    """
    
    m=7
    
    ls=[1,1,1]
    
    t = sp.symbols('t')

    # Matrix E
    E=sp.Matrix.eye(m)
    E[-1,-1]=0
    
    

    # Abbreviations
    c=sp.cos(t)
    s=sp.sin(t)
    
    a = sp.symbols('alpha')
    F=sp.Matrix([
     [0, 0, 0, -1, 0, 0, 0],
     [0, 0, 0, 0, -1, 0, 0],
     [0, 0, 0, 0, 0, -1, 0],
     [0, 0, s, 0,1, -c, -a*c*c],
     [0, 0,-c,-1, 0, -s, -a*s*c],
     [0, 0, 1, 0, 0, 0, a*s],
     [a*c*c, a*s*c, -a*s, 0, 0, 0, 0]])
    
    
    P=sp.zeros(m,m)
    I3=sp.eye(3)
    

    P[:3,3:6]=I3
    P[3:6,:3]=I3
    P[-1,-1]=1
    #sp.pprint(P)
    F=P@F@P
    
    # Construct K0    
    KT=sp.Matrix(np.zeros([7,7]))

    v1=sp.Matrix([c*c,s*c,-s])
    v2=sp.Matrix([s*c,s*s,c])
    v3=sp.Matrix([s,-c,0])

    #For Hessenberg
    KT[0,:3]=v2.T
    KT[1,:3]=v3.T
    KT[2,3:6]=v2.T
    KT[3,3:6]=v3.T
    
    KT[4,-1]=1
    KT[5,:3]=v1.T
    KT[6,3:6]=v1.T
    
    K0=KT.T 
    
    L0=sp.Matrix(np.zeros([7,7]))
    L0[0,:3]=v2.T
    L0[1,:3]=v3.T
    L0[2,3:6]=v2.T
    L0[3,3:6]=v3.T
    
    L0[4,:3]=v1.T
    L0[5,3:6]=v1.T
    L0[6,-1]=1
    
    return E,F,K0,L0,ls,t 

if __name__ == "__main__":
    
    Eend,Fend, Ks, Ls=generate_QuasiSCF_DAE(*Campbell_Moore())

    latex_matrix(Eend,'E_{end}')
    latex_matrix(Fend,'F_{end}')
    
    K0=Ks[0]
    K1=Ks[1]
    K2=Ks[2]

    latex_matrix(K0,'K_0')
    latex_matrix(K1@K2,'K1*K2')
    #The expression for Pi_can is very long.