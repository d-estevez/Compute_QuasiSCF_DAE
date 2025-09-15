# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 16:29:25 2025

@author: schwarz
"""

import numpy as np
import sympy as sp

from Generate_QuasiSCF_Steps import generate_QuasiSCF_DAE,compute_Pican_AB,latex_matrix

def E_3D_HMM():
    """
    Example from M. Hanke, E. Izquierdo Macana, and R. M¨arz. 
    On asymptotics in case of linear index-2 differential-algebraic equations. 
    SIAM J. Numer. Anal., 35(4):1326–1346, 1998.
    """
    
    m=3
    
    ls=[1,1]
    
    t = sp.symbols('t')
    lam = sp.symbols('lambda')
    eta  = sp.symbols('eta')
    
    nt=eta*t

    E=sp.Matrix([
     [1, 0, 0],
     [0, 1, 0],
     [0,0,0]])
    
    F=sp.Matrix([
     [lam, -1, -1],
     [nt*(1-nt)-eta, lam, -nt],
     [1-nt,1,0]])

    c2=sp.Matrix([1-nt,1,0])
    c=sp.sqrt((1-nt)**2+1)
    v2=c2/c
    v1=sp.Matrix(sp.Matrix([1,-1+nt,0]))/c
    
    # Construct K0 (orthogonal)    
    K0T=sp.Matrix(np.zeros([m,m]))
    K0T[0,:]=v1.T
    K0T[1,-1]=1
    K0T[2,:]=v2.T
    K0=K0T.T
    
    # Construct L0 (orthogonal)
    L0=sp.Matrix(np.zeros([m,m]))
    L0[0,:]=v1.T
    L0[1,:]=v2.T
    L0[2,-1]=1
   
    
    return E,F,K0,L0,ls,t 

if __name__ == "__main__":
    
    E,F,K,L,ls,t=E_3D_HMM()
    
    latex_matrix(E,'E')
    latex_matrix(F,'F')
    print('ls',ls)
    
    Eend,Fend,Ks,Ls=generate_QuasiSCF_DAE(*E_3D_HMM())
    latex_matrix(Eend,'E_{end}')
    latex_matrix(Fend,'F_{end}')
    
    Pican=compute_Pican_AB(Ks,ls)
    latex_matrix(Pican,'Pi_{can}')

  
        