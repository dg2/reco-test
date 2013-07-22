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
    U = np.random.randn(Nu, k)/reg
    V = np.random.randn(k, Ni)/reg
    
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
        elif (np.isnan(err) or (err>err_old)):
            print 'Backing off'
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

def bias_correction(A, reg = 1, THRESH = 1e-2, MAX_ITER = 5):
    '''
    Perform bias correction by taking into account per-user and per-item biases

    Minimizes \sum_{(u,i) \in S} (r_{ui}-(\mu+b_u+b_i))^2 + \lambda(\sum_u b_u^2+ \sum_i b_i^2
    '''
    
    N_u, N_i = A.shape
    
    A = A.copy()
    A.data = A.data-A.data.mean()
    
    ## Fixed-point iteration for finding user and item biases
    # Initialize
    b_u = np.zeros(N_u)
    b_i = np.zeros(N_i)
    t = 0

    while 1:
        t += 1
        old_bu = np.array(b_u)
        old_bi = np.array(b_i)

        ## Update user biases        
        for n in range(N_u):
            aux = A.nonzero()[1][A.nonzero()[0]==n]
            if (len(aux)==0):
                continue
            r = A[n,aux].A
            bias = old_bi[aux]
            b_u[n] = np.sum(r - bias)/(reg+len(aux))
        
        print np.max(b_u)
        print np.min(b_u)

        # Update item biases

        for n in range(N_i):
            aux = A.nonzero()[0][A.nonzero()[1]==n]
            if (len(aux)==0):
                continue
            r = A[aux,n].A
            bias = old_bu[aux]
            b_i[n] = np.sum(r-bias)/(reg+len(aux))

        print np.max(b_i)
        print np.max(b_i)

        if ((norm(b_u-old_bu)<THRESH) and (norm(b_i-old_bi)<THRESH)):
            print 'Convergence reached'
            return (b_u, b_i)

        if (t >= MAX_ITER):
            print 'Maximum number of iterations reached'
            return (b_u, b_i)

def bias_correction_sgd(A, reg = 1, THRESH = 1e-3, MAX_ITER = 50, lr = 0.01):
    '''
    Perform bias correction by taking into account per-user and per-item biases

    Minimizes \sum_{(u,i) \in S} (r_{ui}-(\mu+b_u+b_i))^2 + \lambda(\sum_u b_u^2+ \sum_i b_i^2
    '''
    
    N_u, N_i = A.shape
    
    A = A.copy()
    A.data = A.data-A.data.mean()
    
    ## Stochastic Gradient Descent
    # Initialize
    b_u = np.zeros(N_u)
    b_i = np.zeros(N_i)
    t = 0
    err_est = np.inf

    support = np.vstack(A.nonzero())
    N = support.shape[1]
    err_vec = np.zeros(N)

    while 1:
        t += 1
        old_bu = np.array(b_u)
        old_bi = np.array(b_i)
        old_err = err_est
        
        ## Update user biases        
        err_est = 0
        for n in range(support.shape[1]):
            u,i = support[:,n]
            pred = b_u[u] + b_i[i]
            err = A.data[n]-pred
            err_vec[n] = err
            err_est += err**2
            b_u[u] += lr*(err-reg*b_u[u])
            b_i[i] += lr*(err-reg*b_i[i])

        err_est = err_est/N
        print 'Iteration %i\tEstimated error: %f' % (t,err_est)

        if (err_est > old_err):
            print 'Backing off ...'
            b_u = np.array(old_bu)
            b_i = np.array(old_bi)
            lr = lr*0.75
            continue

        if (np.abs(err_est-old_err)<THRESH):
            print 'Convergence reached'
            A.data = err_vec
            return (b_u, b_i, A)

        if (t >= MAX_ITER):
            print 'Maximum number of iterations reached'
            A.data = err_vec
            return (b_u, b_i, A)

