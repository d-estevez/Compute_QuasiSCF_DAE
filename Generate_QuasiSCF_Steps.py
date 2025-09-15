# -*- coding: utf-8 -*-
"""
Created on Mon Sep 08 09:35:47 2025

@author: Estévez Schwarz

This program computes a QuasiSCF via a PreSCF for 
regular linear time varying DAEs and provides the 
projector Pi_can accordingly to:

Diana Estévez Schwarz, René Lamour, and Roswitha März: 
   "Computing standard canonical forms of regular linear
   time varying DAEs via a preliminary stage"
 

"""
import numpy as np
import sympy as sp


def generate_QuasiSCF_DAE(E,F,K0,L0,ls,t,col=True):
    """
    Compute QuasiSCF via  PreSCF for linear, time varying DAEs
    E,F pair of matrix functions
    
    INPUT:
    E,F: matrix pair of DAE
    K0,L0: transformation matrices for a PreSCF
    ls canonical characteristics in the order of col=True
    col=True :  N  from SUT_comumns  (default)
    col=False:  N  from SUT_rows
    """
    m=E.shape[0]
    I=sp.Matrix.eye(m)
    Esscf=E_sscf(m,ls,col)
    
    #################################################
    # STEP 0: Compute PreSCF with given L0 and K0   #
    #################################################
   
    # L0, K0 lead to the PreSCF
    E,F= equivalence(E,F,L0,K0,t)
    
    #Check if now E is E_scf:
        
    E0=E.applyfunc(int) # If here an error appears, then 
    # E0 is not constant and therefore not correct
    
    if E0!=Esscf:  #Check if constant E0 is correkt
        print('E_0:')
        sp.pprint(E0)
        print('E_sscf:')
        sp.pprint(Esscf)
        raise ValueError("ERROR: E_0!=E_sscf, check LO and K0")
        return 
    
    ################################
    #STEP 1: Compute pure ODE      #
    ################################
    
    K1,L1=I.copy(), I.copy()
    E,F,K1,L1=reduce_F12(E,F,ls,t)
    
             
    ###############################
    # STEP 2: Compute pure DAE    #
    ###############################
            
    K2,L2=I.copy(),I.copy()
    E,F,K2,L2=reduce_F21(E,F,ls,t)
     
    # Prepare Output:
    Ks=[K0,K1,K2]
    Ls=[L0,L1,L2]
    
    return E,F,Ks,Ls

def N_Ec(ls):
    """
    Construct N_Ec with block sizes from ls
    """
    n=sum(ls)
    l1=ls[0]
    thetas=ls[1:]
    matrix_size = (n, n)  
    matrix = np.zeros(matrix_size)  

    row, col, lk = 0, l1, l1
    
    for theta in thetas:
      # Insert an identity matrix of size theta at the specified block
      matrix[row:row + theta, col:col + theta] = np.eye(theta)
      
      # Update indices for the next block
      row += lk
      col += theta
      lk = theta 
        
    return matrix

def generate_N(m,ls,col=True):
    """
    Construct elementary nilpotent matrix N_Ec or N_Er
    """

    N=N_Ec(ls) 
    
    if col==False:
        Nh = N[::-1].T
        N=Nh[::-1].copy() # provide N_Er
    return N

def E_sscf(m,ls,col):
    """
    Construct constant block structured matrix E from SSCF
    """
    
    l=int(sum(ls))
    N= generate_N(m,ls,col)
    E=np.eye(m)
    E[-l:,-l:]=N
    Esscf=sp.Matrix(E.astype(int))
    
    return Esscf

def equivalence(E,F,L,K,t):
    """
    Time varying equivalence transformation of pair E,F 
    with tranformation matrices L,K
    """
    
    Kdot=sp.simplify(sp.diff(K, t))
    
    Fs=L@F@K + L@E@Kdot
    Es=L@E@K
    Es=sp.simplify(Es)
    Fs=sp.simplify(Fs)
  
    return Es,Fs

def trace_K(Kold,Knew):
    Knewp=sp.simplify(Kold@Knew)
    return Knewp

def trace_L(Lold,Lnew):
    Lnewp=sp.simplify(Lnew@Lold)
    return Lnewp
    
def reduce_F12(E,F,ls,t):
    """
    Equivalence transformations to obtain a pure ODE
    	"""
    m=E.shape[0]
    d=m-sum(ls)
    index=len(ls)
    I=sp.Matrix.eye(m)
    Kprod, Lprod = I.copy(),I.copy()
    E22=E[-(m-d):,-(m-d):,].copy()
    
    for i in range(index):
        F22=F[-(m-d):,-(m-d):].copy()
        F12=F[:d,-(m-d):].copy()        
        iF22=F22.inv()
        
        Li,Ki=I.copy(),I.copy()
        
        Li[:d,-(m-d):]=-sp.simplify(F12@iF22)
        Ki[:d,-(m-d):]=sp.simplify(F12@iF22@E22)
        
        E,F= equivalence(E,F,Li,Ki,t)
        
        Kprod=trace_K(Kprod,Ki)
        Lprod=trace_L(Lprod,Li)
        
    return E,F,Kprod,Lprod

def reduce_F21(E,F,ls,t):
    """
    Equivalence transformations to obtain a pure DAE
    	"""
    m=E.shape[0]
    d=m-sum(ls)
    index=len(ls)
    I=sp.Matrix.eye(m)
    Kprod, Lprod = I.copy(),I.copy()
    E22=E[-(m-d):,-(m-d):,].copy()
    
    for i in range(index):
         F22=F[-(m-d):,-(m-d):].copy()
         F21=F[-(m-d):,:d].copy()
         iF22=F22.inv()
         
         Li,Ki=I.copy(),I.copy()
         
         Ki[-(m-d):,:d]=-sp.simplify(iF22@F21)
         Li[-(m-d):,:d]=sp.simplify(E22@iF22@F21)
         
         E,F= equivalence(E,F,Li,Ki,t)
         
         Kprod=trace_K(Kprod,Ki)
         Lprod=trace_L(Lprod,Li)
         
    return E,F,Kprod,Lprod



def compute_Pican_AB(Ks,ls):
    """
    Computation of the canonical projector Pi_can
    """
    
    K0=Ks[0]
    K1=Ks[1]
    K2=Ks[2]
    
    m=K0.shape[0]
    d=m-sum(ls)
    
    A=K1[:d,-(m-d):].copy()  
    B=K2[-(m-d):,:d].copy()
    
    AB=A@B
    ABA=AB@A
    BA=B@A
    
    Pi_K1K2=sp.zeros(m,m)
    Pi_K1K2[:d,:d]=sp.eye(d)+AB
    Pi_K1K2[:d,-(m-d):]=-A-ABA
    Pi_K1K2[-(m-d):,:d]=B
    Pi_K1K2[-(m-d):,-(m-d):]=-BA
    
    Pi_K1K2=sp.simplify(Pi_K1K2)
    
    Pi_can=sp.simplify(K0@Pi_K1K2@K0.inv())
    

    return Pi_can

    
def latex_matrix(A,name):
    print(r'\[')
    print(name+'=')
    latex_code = sp.latex(sp.simplify(A))
    print(latex_code)
    print(r',\quad ')
    print(r'\]')
    
    return 

    
if __name__ == "__main__":
    
# Example from Berger/Ilchmann for sin(t)!=0        
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
    
    

    Eend,Fend,Ks,Ls=generate_QuasiSCF_DAE(E,F,K0,L0,ls,t)

    
    latex_matrix(Eend,'E_{end}')
    latex_matrix(Fend,'F_{end}')
    
    Pican=compute_Pican_AB(Ks,ls)
    latex_matrix(Pican,'Pi_{can}')
    
    
    
    
        