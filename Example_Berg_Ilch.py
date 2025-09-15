# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 08:42:21 2025

@author: schwarz
"""

import numpy as np
import sympy as sp
#import copy

from Generate_QuasiSCF_Steps import generate_QuasiSCF_DAE,compute_Pican_AB,latex_matrix

def Berg_Ilch():
    """
    Example from  T. Berger and A. Ilchmann. 
    On the standard canonical form of time-varying linear DAEs. 
    Quarterly of Applied Mathematic, LXXI(1):69–87, 2013.
    """
    m=3
    
    ls=[1,1]
    
    t = sp.symbols('t')


    # Abbreviations
    c=sp.cos(t)
    s=sp.sin(t)
    
    E=sp.Matrix([
     [s, c, 0],
     [0, 0, 0],
     [-c*s,s*s,0]])
 
    F=-sp.Matrix([
     [s-c, c+s, 0],
     [-c, s, 0],
     [-s*s, -s*c, t*t+1]])
    
    L0=sp.Matrix([
     [s*s, 0, -c],
     [c, 0, 1],
     [0,1,0]])
    
    K0=sp.Matrix([
     [1/s, 0, 0],
     [0, 0, 1],
     [0,1,0]])
    
    return E,F,K0,L0,ls,t 

if __name__ == "__main__":
    E,F,K0,L0,ls,t=Berg_Ilch()
    latex_matrix(E,'E')
    latex_matrix(F,'F')
    print('ls',ls)
    
    Eend,Fend, Ks, Ls=generate_QuasiSCF_DAE(*Berg_Ilch())
    
    Pican=compute_Pican_AB(Ks,ls)
    latex_matrix(Pican,'Pi_{can}')