# -*- coding: utf-8 -*-
"""
Methods for approximate matrix factorization using incomplete data

@author: dario
"""
import numpy as np
from numpy.linalg import *
from sys import stdout

def ALS(A, k, reg):
    '''
    Alternating least-squares for rank-k approximation to a sparse matrix A
    
    A ~ UV
    
    where A is N_u x N_i, U is N_u x K and V is K x N_i
    
    The optimization problem is
    
    ||A - UV||^2 + reg (||U||^2_F+||V||^2_F) }
    
    where the matrix norm runs only over observed elements.
    
    Input:
    
    - A: Matrix to factorize
    - K: Rank of the approximation
    - reg: Regularization parameter
    '''
    
    # Data size
    Nu, Ni = A.shape
    
    # Random initialization
    U = np.random.randn(Nu, k)
    V = np.random.randn(k, Ni)
    
    while 1:
        # Single LS step
        # 1 - Fix U and optimize V
        pass
        

    
def SGD(A, k, reg, NUM_ITER = 100, TOL = 0.001, lr = 1e-2, t0 = 100, BACKOFF_RATE = 0.5, MIN_LR = 1e-3):
    '''
    Utterly basic stochastic gradient descent for low-rank factorization
    of spare matrices
    '''

    # Data size
    Nu, Ni = A.shape
    
    # Remove avg
    # A.data = A.data - A.data.mean()

    # Random initialization
    U = np.random.randn(Nu, k)
    V = np.random.randn(k, Ni)
    
    support = np.vstack(A.nonzero())    
    err_old = np.inf
    
    for t in range(NUM_ITER):
        U_old = np.array(U)
        V_old = np.array(V)
        
#        stdout.write('.')
        for n in range(support.shape[1]):
            if (A.data[n]==0):
                continue
            u,i = support[:,n]
            U[u,:] += -reg*lr*U[u,:]+lr*V[:,i]*(A.data[n]-np.dot(U[u,:],V[:,i]))
            V[:,i] += -reg*lr*V[:,1]+lr*U[u,:]*(A.data[n]-np.dot(U[u,:],V[:,i]))
            
        err = sp_norm(A,np.dot(U,V))
        print 'Iteration %i\tError %f\tLearning rate %f' % (t,err, lr)
        # Stoppping condition
        if (abs((err-err_old)/err_old)<TOL):
            print 'Convergence reached'         
            break
        elif (err>err_old):
            print 'Error increasing, backing off'
            U = np.array(U_old)
            V = np.array(V_old)
            lr = lr/anneal_rate
            lr = lr*anneal_rate*BACKOFF_RATE
            continue
        
        anneal_rate = np.sqrt(t0)/np.sqrt(t0+t)
        lr = max(MIN_LR,lr*anneal_rate)
        err_old = err
           
    return U,V
    
def sp_norm(A,approx):
    return norm(A.data[A.data!=0]-approx[A.nonzero()])/norm(A.data)
