# -*- coding: utf-8 -*-
"""
Methods for approximate matrix factorization using incomplete data

@author: Dario
"""
# TODO: Allow using L1 regularization for coefficients others than w
# TODO: More loss functions (Huber, ...)
# TODO: Online feature normalization (basically, running means and stds and updates every N_u samples)
#       We more or less cover part of this with the per-user and per-item biases
# TODO: Allow incremental update. This requires encapsulating the SGD matrix factorization as a class
# TODO: Cache input. memmap?
# TODO: Accept a list of files as input
# TODO: Parallel version
# TODO: Smarter stopping. Instead of only evaluating error at the end of each iteration, look at some exponential
#       average of the per-block cumulative errors or something similar
# TODO: Add some Cython goodness
# TODO: We could initialize a feature vector by drawing from the distribution of feature vectors. That means simply choosing as the initial vector one at random from the pool of existing customers/items

import numpy as np
import time
import os
import sys
import re
import signal
import cPickle
from os.path import join
from numpy.linalg import *
from scipy.sparse import coo_matrix
from sys import stdout
from math import log, exp
from collections import namedtuple
from multiprocessing import Pool
import subprocess

# Namedtuple to hold results for the matrix factorization
#SGD_result = namedtuple('SGD_result',['user_features','item_features','user_bias','item_bias','input_file','parameters'])

#class EndOfPipe()
    
class LoopedPipe():
    '''
    Allows opening a read pipe from an external command (e.g. head, hadoop -fs -cat) that implements seek(0). 
    This class also supports wrapping over itself, i.e. the pipe gets re-started when it is consumed. In that
    case, the current_pass member variable indicates which is the current pass through the data.
    
    It is useful for having multiple SGD passes from a dataset that is streamed.
    '''
    def __init__(self,params, wrap=False, max_lines = None):
        if isinstance(params,str):
            params = params.split(' ')
            
        self.params = params
        self.wrap = wrap
        self.max_lines = max_lines
        # Open a pipe with the desired parameters
        self.reset_pipe()
        self.current_pass = 0
        self.current_line = 0
    def seek(self, idx):
        if (idx==0):
            self.reset_pipe()
            self.current_pass = 0
        else:
            raise ValueError('A looped pipe only allows seeking 0')

    def __iter__(self):
        return self
    
    def next(self):
        if (not self.max_lines is None) and (self.current_line>self.max_lines):
            if (self.wrap):         
                print 'Reopening the stream'
                self.reset_pipe()
                self.current_pass += 1
                self.current_line = 0
                return self.pipe.next()            
            else:
                raise StopIteration()

        try:
            out = self.pipe.next()
            self.current_line += 1
            return out
        except StopIteration: 
            if (self.wrap):         
                print 'Reopening the stream'
                self.reset_pipe()
                self.current_pass += 1
                self.current_line = 0
                return self.pipe.next()            
            else:
                raise StopIteration()

    def close(self):    
        self.pipe.close()
        self.process.kill()
        
    def reset_pipe(self):
        self.process = subprocess.Popen(self.params, stdout = subprocess.PIPE)
        self.pipe = self.process.stdout
        
    def __close__(self):
        self.pipe.close()
        self.process.kill()        
        
        #SGD_clustered_result = namedtuple('SGD_clustered_result',['user_features','latent_dictionary','item_features','user_bias','item_bias','input_file','parameters'])

class SGD_result():
    USER_FILENAME = 'user_features.csv'
    ITEM_FILENAME = 'item_features.csv'
    PARAM_FILENAME = 'params'
    DICT_FILENAME = 'latent_dict.csv'
    
    def __init__(self, user_features, item_features, parameters, latent_dictionary = None):
        self.user_features = user_features
        self.item_features = item_features
        self.params = parameters
        self.latent_dictionary = latent_dictionary

    def _save_array_dict(self, file,d):
        with open(file,'w') as h:
            for k,v in d.iteritems():
                h.write('%s,%s\n' % (k,','.join(map(str,v))))
        
    def save(self, folder):
        # Check if folder exists
        if os.path.exists(folder):
            print 'Folder already exists, ignoring'
            return
        os.mkdir(folder)
        user_file = join(folder,self.USER_FILENAME)
        item_file = join(folder,self.ITEM_FILENAME)
        param_file = join(folder,self.PARAM_FILENAME)
        
        # Save the feature arrays
        self._save_array_dict(user_file, self.user_features)
        self._save_array_dict(item_file, self.item_features)
        if (not self.latent_dictionary is None):
            dict_file = join(folder,self.DICT_FILENAME)
            np.savetxt(dict_file, self.latent_dictionary, delimiter=',')
            
        # Save the dictionary       
        with open(param_file,'w') as h:
            for k,v in self.params.iteritems():
                h.write('%s:%s\n' % (k,repr(v)))

def read_array_dict(file, sep =','):
    ''' 
    This function reads a CSV as a dictionary of arrays. That is the format used by the SGD_result class
    for saving user and item features
    '''
    out = dict()
    with open(file, 'r') as h:
        for line in h:
            k, v = line.strip().split(sep,1)
            out[k] = np.array(map(float,v.split(sep)))
    return out

def analyze_dictionary(latent_dictionary, item_features, translation_function = None):
    # Example translation function:
    # categories = pd.read_csv('../categories.csv', sep=',', header = None, index_col = 0)
    # def trans(x):
    #    try:
    #       out = categories.ix[x.split('_')
    #    except:
    #       out = 'unknown'
    #    return out
    # analyze_dictionary(latent_dict, item_features, trans)
    M, K = latent_dictionary.shape
    V = np.dot(latent_dictionary, np.vstack(item_features.values())[:,:K].T)
    for i in range(M):
        print '='*20
        print 'Cluster %i' % i
        aux = np.argsort(V[i])
        print 'Top items: '
        res = [item_features.keys()[i] for i in aux[-10:]]
        if (not translation_function is None):
            res = map(translation_function,res)
        print res    
        print 'Bottom items: '
        res = [item_features.keys()[i] for i in aux[:10]]        
        if (not translation_function is None):
            res = map(translation_function,res)
        print res
        raw_input()    
    
    
# TODO: Function which returns a function that evaluates the gradient updates for different losses and 
# regularization types
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

def ParallelSGD(folder, regexp = None, n_jobs = 5, *args):
    # We assume we have a folder with several files that comprise the dataset we want to 
    # run SGD one
    files = os.listdir(folder)
    # If we pass a regular expression, use it to filter the files
    if (not regexp is None):
        r = re.compile(regexp)
        files = [f for f in files if not r.match(f) is None]
    print files

    def f(x):
        SGF_file(join(folder,x), args)
        
    # Run SGD on parallel on each block
    p = Pool(n_jobs)
    p.map(f, files)
    # Average the results

    
def SGD_file(input_file, k, reg, output_folder = None, indices=[0,1,2], sep=',', num_passes = 1, 
             TOL = 0.001, lr0 = 1e-2, t0 = 100, BACKOFF_RATE = 0.5, MIN_LR = 1e-3, 
             U = None, V=None, mean_reg=0, transform = None, user_bias = False, 
             item_bias = False, user_feat_indices = [], item_feat_indices = [], initialize_with_avg = False,
             use_constant = True, initialize_with_zeros = True, serialize = True):
    '''
    Basic stochastic gradient descent for low-rank factorization
    of sparse matrices. Read files in the format

    Inputs:

    - input_file: A file-like object or a string containint a filename. 
    - k: Dimensionality of the latent space
    - reg: L2 regularization parameter
    - output_folder: Name of the folder where results will be saved to. By default, it is based on the input file name
    - serialize [True]: Whether the results should be serialized or not. 
    - sep [',']: The field delimiter for the input file
    - indices [0,1,2]: The indices for the user, item and ranking columns in the input file
    - U:
    - V:
    - mean_reg [0]: Parameter for the mean-based regularizer (i.e. a term that pushes individual latent space 
    representations to be close to the average) EXPERIMENTAL!
    - num_passes [1]: Maximum number of passes through the dataset
    - TOL
    - lr0
    - t0
    - transform: A function to be applied to the individual rankings (e.g. log)
    - user_bias [False]: Introduce a per-user bias term 
    - item_bias [False]: Introduce a per-item bias term
    - user_feat_indices []: Indices for addtional, per-user features (EXPERIMENTAL)
    - item_feat_indices []: Indices for additional, per-item features (EXPERIMENTAL)
    - initialize_with_zeros [True]
    - initialize_with_avg [False]: If set to true, starts the feature vector for an unseen customer/item using 
      a perturbed version of the running average of such features
    - use_constant = True
    '''
    
    LINES_BETWEEN_INFO = 100000
    LINES_BETWEEN_BACKUP = 1000000
    
    err_old = np.inf
    anneal_rate = 1
    lr = lr0

    # If we do not have any starting values for the customer/item features, initialize them as empty dictionaries
    if (U is None):
        print 'Initializing user latent features'
        U = dict()                
    if (V is None):
        print 'Initializing item latent features'        
        V = dict()

    if (initialize_with_avg):
        print 'Using average-based initialization'

    # Initialize the counters for number of users and items
    N_u = len(U)
    N_v = len(V)

    N_user_feats = len(user_feat_indices)
    N_item_feats = len(item_feat_indices)
    
    # If necessary, update the user and item biases
    if (user_bias is True):
        print 'Initializing user bias'
        ub = dict()
    elif (isinstance(user_bias,dict)):
        print 'Loading user bias for %i users' % len(user_bias)
        ub = user_bias
        user_bias = True
    else:
        ub = None
    if (item_bias is True):        
        print 'Initializing item bias'
        vb = dict()
    elif (isinstance(item_bias,dict)):
        print 'Loading item bias for %i items' % len(item_bias)
        vb = item_bias
        item_bias = True   
    else:
        vb = None
    
    
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
        input_file = 'stream_input'
    if output_folder is None:
        output_folder = 'res_%s' % input_file.split('/')[-1]

    # Constant
    const = 0
    
    line_number = 0
    cum_err = 0
    aux_cum_err = 0
    t_0 = time.time()
    t_block = time.time()
    sc_factor = max(reg,10)
    # Let us trap Ctrl+C so that we can stop the iteration and save the result
    def signal_handler(sig, frame):
        print 'Ctrl+C pressed, stopping the optimization'
        if (not raw_input('Save results? (y/n) ').lower()=='y'):
            # 'Untrap' the signal
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            # Propagate
            raise KeyboardInterrupt
        # We are done. Pack the results and end
        # TODO: Write to different, standard files instead of dumping
        param = {'k':k, 'reg':reg, 'lr':lr0, 'num_passes':num_passes, 'user_bias':user_bias, 
                 'item_bias':item_bias, 'user_feat_indices':user_feat_indices, 
                 'item_feat_indices':item_feat_indices, 
                 'transform':repr(transform), 'indices':indices,
                 'input_file':input_file}
        out = SGD_result(user_features = U, item_features = V, parameters = param)
        outfolder = '%s_%03i.dump' % (output_folder, np.random.randint(999))
        print 'Serializing'
        out.save(outfolder)
        print 'Serialized output to %s' % outfile
        # 'Untrap' the signal
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        # Propagate
        raise KeyboardInterrupt
        
    # Register the handler
    signal.signal(signal.SIGINT, signal_handler)
    
    for t in range(num_passes):
        print 'Pass %i' % t
        if (t>0):
            try:
                h.seek(0)            
            except:
                print 'Can not rewind the file. Is it a piped stream? Then use the LoopedPipe class'
                break
            
        iter_err = 0
        # Process a rating
        for line in h:
            line_number += 1
            try:
                values = line.split(sep)
                u_code = values[indices[0]]   # User
                i_code = values[indices[1]]   # Item
                r = float(values[indices[2]]) # Rating
            except:
                print 'Problem parsing line, ignoring'
                continue
            
            if (user_feat_indices):
                u_feat = [indices[a] for a in user_feat_indices]
            if (item_feat_indices):
                v_feat = [indices[a] for a in item_feat_indices]

            if (not transform is None):
                # TODO: Should we cache this function?
                # Maybe a simple memoization would do
                r = transform(r)

            # Check if it is the first time that we see a customer
            if (not u_code in U):
                u_f = np.zeros(k+1+N_item_feats)
                # Random initialization: We use the current mean feature vector plus some noise
                u_f[:k] = np.random.randn(k)/sc_factor 

                if (initialize_with_avg):
                    u_f[:k] += u_mean
                        
                if (item_feat_indices):
                    u_f[k+1:] = np.random.randn(N_item_feats)/sc_factor
                    #else:
                    #uc = [0]
                    
                U[u_code] =u_f
                # Update counter
                N_u += 1
     

            # Do the same for items
            if (not i_code in V):          
                v_f = np.zeros(k+1+N_user_feats)      
                v_f[:k] = np.random.randn(k)/sc_factor

                if (initialize_with_avg):
                    v_f[:k] += v_mean

                    #                if (item_bias is True):
                    #vb = np.random.randn(1)/sc_factor
                    #else:
                    #vb = [0]
                if (user_feat_indices):
                    v_f[k+1:] = np.random.randn(N_user_feats)/sc_factor                
                    #                else:
#                    vc = [0]
                N_v += 1
                V[i_code] = v_f

            # Prediction for the current data point
            u_f = U[u_code]
            v_f = V[i_code]

            u = u_f[:k]
            ub = u_f[k:k+1]
            uc = u_f[k+1:]

            v = v_f[:k]
            vb = v_f[k:k+1]
            vc = v_f[k+1:]
                                    
            pred = np.dot(u,v) + ub + vb + const
                        
            if (user_feat_indices):
                pred += np.dot(uc,v_feat)
            if (user_feat_indices):
                pred += np.dot(vc,u_feat)
                    
            # Prediction err
            err = r-pred
            err_sq = err*err
            
            # Sanity check
            if (np.isnan(err)):
                print line_number
                print pred
                print err
                print U[u_code]
                print V[i_code]
                break

            # Update cumulative error counters
            cum_err += err_sq
            aux_cum_err += err_sq
            iter_err += err_sq

            ####################
            # Gradient updates
            ####################
            u_update = lr*v*err-reg*lr*u-mean_reg*lr*(u-u_mean)
            v_update = lr*u*err-reg*lr*v-mean_reg*lr*(v-v_mean)

            # Note: The following lines already update the values
            # in the dictionary, since a numpy array is mutable
            u += u_update
            v += v_update            
            if (user_bias is True):
                ub += lr*(err-reg*ub)
            if (item_bias is True):
                vb += lr*(err-reg*vb)

            # Constant
            if (use_constant is True):
                const += lr*(err-reg*const)
                
            # Feature-related weights
            if (user_feat_indices):
                vc += lr*u_feat*err-reg*lr*vc
            if (item_feat_indices):
                uc += lr*v_feat*err-reg*lr*uc
                   
            u_mean += u_update/N_u
            v_mean += v_update/N_u

            #print u_update
            #print v_update             
            #raw_input()

            # Learning rate update            
            lr = lr0/(1+lr0*reg*t)
                      
            # Show info
            if line_number % LINES_BETWEEN_INFO == 0:
                t_current = time.time()
                print ('Instance %i, %s:%s:%f:%f\nMean error: %.3f, ' + 
                'Mean error since last: %.3f, Nu: %i, Ni: %i, Time since last: %.3f, ' +
                'Total time: %.3f, Best constant: %.3f') % \
                (line_number, u_code, i_code, r, pred, cum_err/line_number, 
                 aux_cum_err/LINES_BETWEEN_INFO, N_u, N_v,t_current-t_block,t_current-t_0,
                 const
                )
                # Reset the error-between-updates counter and the timer
                aux_cum_err = 0
                t_block = t_current
                #        print 'Iteration %i\tError %f\tLearning rate %f' % (t,cum_err, lr)

        # Stoppping condition        
        if (iter_err<1e-5 or ((t>0) and (abs((iter_err-err_old)/err_old)<TOL))):
            print 'Convergence reached'         
            print iter_err
            print err_old
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

    # We are done. Pack the results and exit the building
    # TODO: Write the results to a file instead of passing them as an object?
    param = {'k':k, 'reg':reg, 'lr':lr0, 'num_passes':num_passes, 
             'user_bias':user_bias, 'item_bias':item_bias,
             'user_feat_indices':user_feat_indices, 'item_feat_indices':item_feat_indices, 
             'transform':repr(transform),'indices':indices,
             'input_file':input_file}
        #    if (user_bias is True):
        #param.update({'ub':ub})
        #    if (item_bias is True):
        #param.update({'vb':vb})
    out = SGD_result(user_features = U, item_features = V, parameters = param)
    if (serialize):
        out.save(output_folder)
        
    # Cleanup
    # 'Untrap' the signal
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    return out


def SGD_predict(input_file, SGD_res, indices = None, user_feat_indices = None, item_feat_indices = None, sep = ',', output_file = None):
    if (indices is None):
        indices = SGD_res.parameters.get('indices',[0,1,2])
    if (user_feat_indices is None):
        user_feat_indices = SGD_res.parameters.get('user_feat_indices',[])
    if (item_feat_indices is None):
        item_feat_indices = SGD_res.parameters.get('item_feat_indices',[])
    if (output_file is None):
        output_file = input_file + '_pred'
    with open(input_file, 'r') as h_in:
        with open(output_file, 'w') as  h_out:
            line_number = 0
            for line in h_in:
                line_number += 1
                if (line_number % 500000 == 0):
                    sys.stdout.write('.')
                values = line.split(sep)
                u_code = values[indices[0]]   # User
                i_code = values[indices[1]]   # Item
                r = float(values[indices[2]]) # Rating
                if (user_feat_indices):
                    u_feat = [indices[a] for a in user_feat_indices]
                    if (item_feat_indices):
                        v_feat = [indices[a] for a in item_feat_indices]                         
                        if (not transform is None):
                            # TODO: Should we cache this function?
                            # Maybe a simple memoization would do
                            r = transform(r)

                # Prediction for the current data point
                u, ub, uc = SGD_res.user_features[u_code]
                v, vb, vc = SGD_res.item_features[i_code]
                        
                pred = np.dot(u,v) + ub + vb
            
                if (user_feat_indices):
                    pred += np.dot(uc,v_feat)
                if (user_feat_indices):
                    pred += np.dot(vc,u_feat)
                    
                h_out.write('%s,%s,%.3f\n' % (u_code, i_code, pred))    
                
        sys.stdout.write('\n')


######################################################################
def SGD_clustered(input_file, k, M, reg, reg_w=1, output_folder = None, indices=[0,1,2], sep=',', num_passes = 1, 
             TOL = 0.001, lr0 = 1e-2, t0 = 100, BACKOFF_RATE = 0.5, MIN_LR = 1e-3, 
             U = None, V=None, mean_reg=0, transform = None, user_bias = False, 
             item_bias = False, user_feat_indices = [], item_feat_indices = [], initialize_with_avg = False,
             use_constant = True, initialize_with_zeros = True, sc_factor = None, serialize = True,
             extra_pass = 1, kmeans_mode = False, max_lines = None):
    '''
    Stochastic gradient descent for low-rank factorization with clustered
    user features.

    The main idea is that the latent space representation for a user can be written in terms of some basis 
    vectors x_j

    u_i = \sum_j^M w_{ij}x_j

    This expands the number of parameters from (N+C)k [+N+C+1] to NM + (N+M)k [+N+C+1]. The benefit that 
    we obtain is that weights w_{ij} can be readily interpreted as affinities of the user i with respect
    to the different 'types' x_j. Ideally, the w's should be strongly regularized with l1 or similar, and 
    be non-negative (not implemented yet)

    Inputs:

    - input_file: A file-like object or a string containing a filename. 
    - k: Dimensionality of the latent space
    - M: Number of user features centroids
    - reg: L2 regularization parameter
    - reg_w: L1 regularization parameter for the individual weights
    - extra_pass [1]: Number of extra passes through the data, updating only
      the per-user weights
    - kmeans_mode [False]
    - serialize [True]
    - output_folder: Folder where results will be saved to. By default, it is based on the input file name
    - sep [',']: The field delimiter for the input file
    - indices [0,1,2]: The indices for the user, item and ranking columns in the input file
    - U:
    - V:
    - mean_reg [0]: Parameter for the mean-based regularizer (i.e. a term that pushes individual latent space 
    representations to be close to the average) EXPERIMENTAL!
    - num_passes [1]: Maximum number of passes through the dataset
    - sc_factor [10]: Inverse standard deviation for the prior weights
    - TOL
    - lr0
    - t0
    - transform: A function to be applied to the individual rankings (e.g. log)
    - user_bias [False]: Introduce a per-user bias term 
    - item_bias [False]: Introduce a per-item bias term
    - user_feat_indices []: Indices for addtional, per-user features (EXPERIMENTAL)
    - item_feat_indices []: Indices for additional, per-item features (EXPERIMENTAL)
    - initialize_with_zeros [True]
    - initialize_with_avg [False]: If set to true, starts the feature vector for an unseen customer/item using 
      a perturbed version of the running average of such features
    - use_constant = True
    '''
    
    LINES_BETWEEN_INFO = 100000
    LINES_BETWEEN_BACKUP = 1000000

    if (sc_factor is None):
        sc_factor = max(reg,10)
        
    err_old = np.inf
    anneal_rate = 1
    lr = lr0

    # If we do not have any starting values for the customer/item features, initialize them as empty dictionaries
    if (U is None):
        print 'Initializing user latent features'
        U = dict()                
    if (V is None):
        print 'Initializing item latent features'        
        V = dict()

    # Initialize the array of centroids for latent user features
    L = np.random.randn(M,k)/sc_factor
    L_update = np.zeros((M,k))
    
    if (initialize_with_avg):
        print 'Using average-based initialization'

    # Initialize the counters for number of users and items
    N_u = len(U)
    N_v = len(V)

    N_user_feats = len(user_feat_indices)
    N_item_feats = len(item_feat_indices)
    
    # If necesarry, update the user and item biases
    if (user_bias is True):
        print 'Initializing user bias'
        ub = dict()
    elif (isinstance(user_bias,dict)):
        print 'Loading user bias for %i users' % len(user_bias)
        ub = user_bias
        user_bias = True
    else:
        ub = None
    if (item_bias is True):        
        print 'Initializing item bias'
        vb = dict()
    elif (isinstance(item_bias,dict)):
        print 'Loading item bias for %i items' % len(item_bias)
        vb = item_bias
        item_bias = True   
    else:
        vb = None
    
    
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
        input_file = 'stream_input'
    if output_folder is None:
        output_folder = 'res_%s' % input_file.split('/')[-1]

    # Constant
    const = 0

    # Counters for how many points have been assigned to a particular cluster
    counter = np.zeros(M)
    
    line_number = 0
    cum_err = 0
    aux_cum_err = 0
    t_0 = time.time()
    t_block = time.time()

    # Let us trap Ctrl+C so that we can stop the iteration and save the result
    def signal_handler(sig, frame):
        print 'Ctrl+C pressed, stopping the optimization'
        if (not raw_input('Save results? (y/n) ').lower()=='y'):
            # 'Untrap' the signal
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            # Propagate
            raise KeyboardInterrupt
        # We are done. Pack the results and end
        param = {'k':k, 'reg':reg, 'lr':lr0, 'num_passes':num_passes, 'user_bias':user_bias, 'item_bias':item_bias,
                 'user_feat_indices':user_feat_indices, 'item_feat_indices':item_feat_indices, 
                 'transform':repr(transform), 'indices':indices, 'input_file':input_file}
        out = SGD_result(user_features = U, item_features = V,
                         parameters = param, latent_dictionary = L)
        
        outfolder = '%s_%03i.dump' % (output_folder, np.random.randint(999))
        print 'Serializing ...'
        out.save(outfolder)
        print 'Serialized output to %s' % outfolder
        # 'Untrap' the signal
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        # Propagate
        raise KeyboardInterrupt
        
    # Register the handler
    signal.signal(signal.SIGINT, signal_handler)

    update_dictionary = True
    update_item_feats = True

    #if (extra_pass is True):
    total_passes = num_passes + extra_pass
    #num_passes = num_passes +1
        
    for t in range(total_passes):
        print 'Pass %i' % t
        if (t >= num_passes):
            print 'Extra pass! Updating only per-user weights'
            # TODO: Shall we reset the learning rate for this extra passes?
            update_dictionary = False
            update_item_feats = False                
            
        if (t>0):
            try:
                h.seek(0)            
            except:
                print 'Can not rewind the file. Is it a piped stream?'
                break
            
        iter_err = 0
        num_empty_lines = 0
        # Process a rating
        for line in h:
            line = line.strip()
            if (line==''):
                num_empty_lines+=1
                continue
            try:
                values = line.split(sep)
                u_code = values[indices[0]]   # User
                i_code = values[indices[1]]   # Item
                r = float(values[indices[2]]) # Rating
                if (user_feat_indices):
                    u_feat = [indices[a] for a in user_feat_indices]
                if (item_feat_indices):
                    v_feat = [indices[a] for a in item_feat_indices]
                        
                if (not transform is None):
                    # TODO: Should we cache this function?
                    # Maybe a simple memoization would do
                    r = transform(r)
            except:
                print 'Invalid line format, ignoring ...'
                print line
                continue
            
            line_number += 1
            
            

            # Check if it is the first time that we see a customer
            if (not u_code in U):
                u_f = np.zeros(M+1+N_item_feats)
                if (kmeans_mode is True):
                    # Assign to one cluster at random
                    u_f[np.random.randint(M)] = 1
                else:
                    # Random initialization: Perturbed equiprobable
                    u_f[:M] = 1/float(M) + np.random.randn(M)/sc_factor

                if (item_feat_indices):
                    u_f[k+1:] = np.random.randn(N_item_feats)/sc_factor
                    
                U[u_code] =u_f
                N_u += 1                    

            # Do the same for items
            if (not i_code in V):          
                v_f = np.zeros(k+1+N_user_feats)      
                v_f[:k] = np.random.randn(k)/sc_factor
                if (initialize_with_avg):
                    v_f[:k] += v_mean
                if (user_feat_indices):
                    v_f[k+1:] = np.random.randn(N_user_feats)/sc_factor                
                N_v += 1
                V[i_code] = v_f

            # Prediction for the current data point
            u_f = U[u_code]
            v_f = V[i_code]
            v = v_f[:k]
            w = u_f[:M]
            ub = u_f[k:k+1]
            vb = v_f[k:k+1]

            Lv = np.dot(L,v)
            indiv_pred = Lv 
            pred = np.dot(w,Lv) + ub + vb + const
            
            if (user_feat_indices):
                pred += np.dot(uc,v_feat)
            if (user_feat_indices):
                pred += np.dot(vc,u_feat)
                    
            # Prediction err
            err = r-pred
            err_sq = err*err
            
            # Sanity check
            if (np.isnan(err)):
                print line_number
                print r
                print pred
                print err
                print U[u_code]
                print V[i_code]
                print L
                print lr
                return 

            # Update cumulative error counters
            cum_err += err_sq
            aux_cum_err += err_sq
            iter_err += err_sq

            ####################
            # Gradient updates
            ####################

            # k-means like
            if (kmeans_mode is True):
                # Optimal assignment
                err_aux = r - Lv - ub -vb -const
                idx_opt = np.argmin(err_aux)
                w[:] = 0
                w[idx_opt] = 1
                counter[idx_opt] += 1
                if update_dictionary:
                    L[idx_opt,:] += 1/counter[idx_opt]*(v*err_aux[idx_opt]-reg*L[idx_opt,:])
                    #                               L[idx_opt,:] += 1/counter[idx_opt]*(v*err_aux[idx_opt]-reg*L[idx_opt,:])
                
            else:
                ## w has L1 regularization: Do truncated gradient as in (Carpenter, 2008)
                # First, take a step in the gradient without considering regularization
                w += lr*Lv*err
                # Now, 'shrink' the vector
                w[w>0] = np.maximum(0, w[w>0] - lr*reg_w)
                w[w<0] = np.minimum(0, w[w<0] + lr*reg_w)             
                
                if update_dictionary:
                    #                for m in range(M):
                    #    aux = L[m,:]
                    #    aux += lr*w[m]*v*err-reg*lr*aux
                    L += lr*w[:,np.newaxis]*v*err-reg*lr*L
                    
            if update_item_feats:
                v += lr*np.dot(w,L)*err-reg*lr*v-mean_reg*lr*(v-v_mean)

            # Note: The following lines already update the values
            # in the dictionary, since a numpy array is mutable
            if (user_bias is True):
                ub += lr*(err-reg*ub)
            if (item_bias is True):
                vb += lr*(err-reg*vb)

            # Constant
            if (use_constant is True):
                const += lr*(err-reg*const)
                
            # Feature-related weights
            if (user_feat_indices):
                vc += lr*u_feat*err-reg*lr*vc
            if (item_feat_indices):
                uc += lr*v_feat*err-reg*lr*uc

            # Update learning rate
            lr = lr0/np.sqrt(1+lr0*reg*line_number)
            #            print lr
                   
            # Show info
            if line_number % LINES_BETWEEN_INFO == 0:
                t_current = time.time()
                print ('Instance %i, %s:%s:%f:%f\nMean error: %.3f, ' + 
                'Mean error since last: %.3f, Nu: %i, Ni: %i, Time since last: %.3f, ' +
                'Total time: %.3f, Best constant: %.3f, lr: %f, empty: %i') % \
                (line_number, u_code, i_code, r, pred, cum_err/line_number, 
                 aux_cum_err/LINES_BETWEEN_INFO, N_u, N_v,t_current-t_block,t_current-t_0,
                 const, lr, num_empty_lines
                )
                # Reset the error-between-updates counter and the timer
                aux_cum_err = 0
                t_block = t_current
                #        print 'Iteration %i\tError %f\tLearning rate %f' % (t,cum_err, lr)

        # Stoppping condition        
        if (iter_err<1e-5 or ((t>0) and (abs((iter_err-err_old)/err_old)<TOL))):
            print 'Convergence reached'         
            print iter_err
            print err_old
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

    # We are done. Pack the results and exit the building
    param = {'k':k, 'reg':reg, 'lr':lr0, 'num_passes':num_passes, 'user_bias':user_bias, 'item_bias':item_bias,
'user_feat_indices':user_feat_indices, 'item_feat_indices':item_feat_indices, 'transform':repr(transform), 
'indices':indices}
    if (user_bias is True):
        param.update({'ub':ub})
    if (item_bias is True):
        param.update({'vb':vb})
    out = SGD_result(user_features = U, item_features = V, parameters = param, latent_dictionary = L)
    
    if (os.path.exists(output_folder)):
        output_folder = output_folder + '%03i' % np.random.randint(999)

    if (serialize):        
        out.save(output_folder)
        print 'Results saved in %s' % output_folder
    
    return out    

######################################################################    
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

def SGD_test():
    res = SGD_file('test',2,0,lr0 = 0.001, NUM_ITER=100000,TOL=1e-7)
    # Reconstruct the matrix
    U = np.vstack([a[0] for a in res.user_features.values()])
    V = np.vstack([a[0] for a in res.item_features.values()])
    print 'User features'
    print res.user_features
    print 'Item features'
    print res.item_features
    print 'Reconstructed: '
    print np.dot(U,V.T)

    
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

