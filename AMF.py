# -*- coding: utf-8 -*-
"""
Methods for approximate matrix factorization using incomplete data

@author: dario
"""
# TODO: Online feature normalization (basically, running means and stds and updates every N_u samples)
# TODO: Allow incremental update. This requires encapsulating the SGD matrix factorization as a class
# TODO: Allow multipass for streaming inputs (e.g. a pipe from HDFS). This requires caching the input, like VW
#       That does not look good for potentially huge files.
#       As a simplification , we can create a wrapper for cat / hadoop -fs -cat that resets itselfs when it 
#       reaches the end of file


import numpy as np
from numpy.linalg import *
from scipy.sparse import coo_matrix
from sys import stdout
from math import log, exp
from collections import namedtuple
import time

SGD_result = namedtuple('SGD_result',['user_features','item_features','input_file','parameters'])

def _sgd_gradient(loss, reg_type):
    if (loss=='l2'):
        pass
    elif (loss=='huber'):
        pass
        
class SGD_MF():
    """
    Matrix factorization using Stochastic Gradient Descent
    """
    def __init__(self, k=10, reg_type = 'l2', loss = 'square_loss', reg_param = 1, NUM_ITER = 100, TOL = 0.001, lr = 1e-2, \
                 t0 = 100, BACKOFF_RATE = 0.5, MIN_LR = 1e-3):
        pass
    def online_fit(stream):
        pass
    def update(stream):
        pass

        
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
    of sparse matrices
    '''

    if (not isinstance(A, coo_matrix)):
        A = coo_matrix(A)

    # Data size
    Nu, Ni = A.shape
    Nnz = A.nnz

    # Remove avg
    # A.data = A.data - A.data.mean()

    # Random initialization
    U = np.random.randn(Nu, k)/reg
    V = np.random.randn(k, Ni)/reg
    
    err_old = np.inf
    anneal_rate = 1

    for t in range(NUM_ITER):
        U_old = np.array(U)
        V_old = np.array(V)
        
#        stdout.write('.')
        for n in range(Nnz):
            r = A.data[n]
            u = A.row[n]
            i = A.col[n]
            if (r==0):
                continue

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

def SGD_file(input_file, k, reg, indices=[0,1,2], sep=',', NUM_ITER = 100, TOL = 0.001, lr0 = 1e-2, t0 = 100, BACKOFF_RATE = 0.5, MIN_LR = 1e-3, U = None, V=None, mean_reg=0, transform = None, user_bias = False, item_bias = False):
    '''
    Basic stochastic gradient descent for low-rank factorization
    of sparse matrices. Read files in the format

    Inputs:

    - input_file: A file-like object or a string containint a filename. 
    - k: Dimensionality of the latent space
    - reg: L2 regularization parameter
    - sep [',']: The field delimiter for the input file
    - indices [0,1,2]: The indices for the user, item and ranking columns in the input file
    - U:
    - V:
    - mean_reg [0]: Parameter for the mean-based regularizer (i.e. a term that pushes individual latent space 
    representations to be close to the average) EXPERIMENTAL!
    - NUM_ITER
    - TOL
    - lr0
    - t0
    - transform: A function to be applied to the individual rankings (e.g. log)
    - user_bias [False]: Introduce a per-user bias term (EXPERIMENTAL)
    - item_bias [False]: Introduce a per-item bias term (EXPERIMENTAL)
    '''
    
    LINES_BETWEEN_INFO = 50000
    err_old = np.inf
    anneal_rate = 1
    lr = lr0

    # If we do not have any starting values for the customer/item features, initialize them as empty dictionaries
    if (U is None):
        U = dict()                
    if (V is None):
        V = dict()

    # Initialize the counters for number of users and items
    N_u = len(U)
    N_v = len(V)

    # If necesarry, update the user and item biases
    if (user_bias is True):
        print 'Initializing user bias'
        ub = dict()
    elif (isinstance(user_bias,dict)):
        print 'Loading user bias for %i users' % len(user_bias)
        ub = user_bias
        user_bias = True
    if (item_bias is True):        
        print 'Initializing item bias'
        vb = dict()
    elif (isinstance(item_bias,dict)):
        print 'Loading item bias for %i items' % len(item_bias)
        vb = item_bias
        item_bias = True   
    
    
    u_mean = 0
    v_mean = 0
    for u in U.itervalues():
        u_mean += u        
    for v in V.itervalues():
        v_mean += v
        
    
    reg = float(reg)

    # Handle both strings inputs (which we assume to be a file name) or file-like inputs
    if (isinstance(input_file,str)):
        h = open(input_file,'r')
    else:
        h = input_file

    line_number = 0
    cum_err = 0
    aux_cum_err = 0
    t_0 = time.time()
    t_block = time.time()

    for t in range(NUM_ITER):
        if (t>0):
            try:
                h.seek(0)            
            except:
                print 'Can not rewind the file. Is it a piped stream?'
                break
            
        iter_err = 0
        # Process a rating
        for line in h:
            line_number += 1
            values = line.split(sep)
            u_code = values[indices[0]]   # User
            i_code = values[indices[1]]   # Item
            r = float(values[indices[2]]) # Rating
            if (not transform is None):
                r = transform(r)

            # Check if it is the first time that we see a client or item
            if (not u_code in U):
                # Random initialization
                U[u_code] = np.random.randn(k)/reg
                # Update counter
                N_u += 1
                # Handle user biases
                if (user_bias is True):
                    ub[u_code] = np.random.randn(1)/reg

            # Do the same for items
            if (not i_code in V):
                V[i_code] = np.random.randn(k)/reg
                N_v += 1
                if (item_bias is True):
                    vb[i_code] = np.random.randn(1)/reg

            # Prediction for the current data point
            u = U[u_code]
            v = V[i_code]
                        
            pred = np.dot(u,v)
            
            if (user_bias is True):
                pred += ub[u_code]
            if (item_bias is True):
                pred += vb[i_code]
                
            # Prediction err
            err = r-pred

            # Sanity check
            if (np.isnan(err)):
                print line_number
                print pred
                print err
                print U[u_code]
                print V[i_code]
                break

            # Update cumulative error counters
            cum_err += abs(err)
            aux_cum_err += abs(err)
            iter_err += abs(err)

            # Gradient updates
            u_update = lr*v*err-reg*lr*u-mean_reg*lr*(u-u_mean)
            v_update = lr*u*err-reg*lr*v-mean_reg*lr*(v-v_mean)

            U[u_code] += u_update
            V[i_code] += v_update

            if (user_bias is True):
                ub[u_code] += lr*(err-reg*ub[u_code])
            if (item_bias is True):
                vb[i_code] += lr*(err-reg*vb[i_code])              


            u_mean += u_update/N_u
            v_mean += v_update/N_u
            

            # Show info
            if line_number % LINES_BETWEEN_INFO == 0:
                t_current = time.time()
                print ('Instance %i, %s:%s:%f:%f\nMean error: %.3f, ' + 
                'Mean error since last: %.3f, Nu: %i, Ni: %i','Time since last: %.3f','Total time: %.3f') % \
                (line_number, u_code, i_code, r, pred, cum_err /
                 line_number, aux_cum_err/LINES_BETWEEN_INFO, len(U), len(V),t_current-t_block,t_current-t_0)
                # Reset the error-between-updates counter and the timer
                aux_cum_err = 0
                t_block = t_current
        
        print 'Iteration %i\tError %f\tLearning rate %f' % (t,cum_err, lr)

        # Stoppping condition        
        if ((t>0) and (abs((iter_err-err_old)/err_old)<TOL)):
            print 'Convergence reached'         
            break
        elif (np.isnan(iter_err) or (iter_err>err_old)):
            print 'Backing off'
            # TODO: To allow for back-off we need to keep a copy of the U and V dictionaries.
            # Is it worth it? Maybe we should have a parameter to select whether we want
            # to use this idea or no
            
            #  U = np.array(U_old)
            #V = np.array(V_old)
            #lr = lr/anneal_rate
            #lr = lr*anneal_rate*BACKOFF_RATE
            #continue

        # Update learning rate
        anneal_rate = np.sqrt(t0)/np.sqrt(t0+t)        
        lr = max(MIN_LR,lr*anneal_rate)
        err_old = iter_err

    # We are done. Pack the results and end
    # TODO: Write the results to a file instead of passing them as an object?
    param = {'k':k, 'reg':reg, 'lr':lr0, 'num_iter':NUM_ITER, 'user_bias':user_bias, 'item_bias':item_bias}
    if (user_bias is True):
        param.update({'ub':ub})
    if (item_bias is True):
        param.update({'vb':vb})
    try:
        out = SGD_result(user_features = U, item_features = V, input_file = filename, parameters = param)
    except:
        out = (U,V)
    return out


def SGD_predict(input_file, U, V, indices = [0,1], sep = ',', output_file = None):
    if (output_file is None):
        output_file = input_file + '_pred'
    with open(input_file, 'r') as h_in:
        with open(output_file, 'w') as  h_out:
            for line in h_in:
                values = line.split(sep)
                u_code = values[indices[0]]
                i_code = values[indices[1]]
                pred = np.dot(U[u_code],V[i_code])
                h_out.write('%s,%s,%.3f\n' % (u_code, i_code, pred))         

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

