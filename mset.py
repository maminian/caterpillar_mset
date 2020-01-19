'''
Implements the tools necessary for an "online" 
implementation of MSET with a vector timeseries.
Supports a subset of `simpler` nonlinear operators 
$\otimes$ which are based on pairwise operations of 
all the elements in X and Y.

The basic algorithm is described in the function
online_MSET().

Manuchehr Aminian
Last update: 20 August 2019
'''

import numpy as np
import scipy

#
# These store the cached similarity matrix 
# and its LU factors, which speeds up evaluation
# of new points. Will be faster than "naive"
# reconstruction when there are only a few anomalies.
#
global _D
global _oDD
global _oDD_lufactors

global _mu
global _mus


_D = np.zeros((0,0))
_oDD = np.zeros((0,0))
_oDD_lufactors = None

_mu = np.nan
_mus = []

#
#########################################################
#

def otimes1_ij(x,y):
    '''
    Similarity operator from the Dreissigmeier paper
    
    s(x,y) = 1 - (||x||**2 - ||y||**2)/(||x-y||**2)
    '''
    import numpy as np
    return 1. - (np.linalg.norm(x)**2 - np.linalg.norm(y)**2)/(np.linalg.norm(x-y)**2)
#

def otimes2_ij(x,y):
    '''
    Similarity operator from the Wang paper
    
    s(x,y) = 1 - ||x-y||/(||x|| + ||y||)
    '''
    import numpy as np
    
    if all(x==y):
        # Handling the case of x and y both being the zero vector.
        return 1.
    else:
        return 1. - np.linalg.norm(x-y)/(np.linalg.norm(x) + np.linalg.norm(y))
#

def otimes(X,Y, op=otimes2_ij):
    '''
    otimes operator on matrices; double loop over the columns of X and Y.
    
    Note slightly different convention than in the papers; there, X loops 
        over the rows and Y over the columns.
    '''
    m1,n = np.shape(X)
    m2,p = np.shape(Y)
    
    if m1!=m2:
        raise Exception('dimensionality mismatch between X and Y.')
    else:
        m = m1
    #
    
    Z = np.zeros( (n,p) )
    
    for i in range(n):
        for j in range(p):
            Z[i,j] = op(X[:,i], Y[:,j])
    #
    return Z
#


def op_W(D,P, op=otimes2_ij):
    '''
    Nonlinearly maps the features into the (possibly overcomplete)
    columns of D based on the operator otimes; roughly,
    
    W = np.linalg.solve( otimes(D,D) , otimes(D,P) ),
    
    with the hope that P is well approximated by DW. Note 
    that if the otimes operator is the simple dot product, 
    then DW==P as long as the rank of D is the same as the 
    dimension of the columns of D.
    
    A rough caching is done if mset.py is treated 
    as a module to accelerate repeated calls to this function; 
    the matrix D and the similarity matrix otimes(D,D)
    are stored internally, and op_W uses the stored otimes(D,D) 
    if D is the same as the internally stored _D.
    '''
    import numpy as np
    from scipy import linalg as spla
    
    global _D
    global _oDD
    global _oDD_lufactors
    
    # If D has not changed since the last usage, use 
    # the cached similarity matrix _oDD. Else, recompute 
    # and store.
    if np.shape(_D)==np.shape(D):
        if np.linalg.norm(_D - D)/np.linalg.norm(D) < 1e-4:
            use_cached = True
        else:
            _D = D
            use_cached = False
    else:
        _D = D
        use_cached = False
    #
    if not use_cached:
        # recompute similarity matrix and 
        # its LU factorization to accelerate solves
        # for future iterations.
        oDD = otimes(D,D, op=op)
        _oDD = oDD

        _oDD_lufactors = spla.lu_factor(oDD)
    #
    
    # fix P if it's a one-dimensional array.
    pshape = np.shape(P)
    
    if len(pshape)==1:
        P.shape = (pshape[0],1)
    #
    
    oDP = otimes(D,P, op=op)

    # The actual operation.    
    try:
        W = spla.lu_solve(_oDD_lufactors, oDP)
    except:
        print('LU solver failed; falling back naive solver.')
        W = np.linalg.solve( _oDD, oDP )
    #
    return W
#


def online_mset(Y, op=otimes2_ij, thresh=0.10, output_norms=False, **kwargs):
    '''
    An 'online' version of MSET for a vector-valued 
    timeseries Y, arranged by **columns**.
    
    The algorithm goes as follows:
        1. Initialization.
            a. Initialize the estimate for the data mean with the 
                first column of Y.
            b. Initialize the memory/dictionary/list of exemplars
                "D" with the second column of Y, subtracting the running average.
        2. For the remaining columns of Y, indexed by j,
            a. Subtract the estimate for the mean from Y[:,j], call it y.
            b. Apply the nonlinear mapping of the data onto 
                the basis of D and calculate the prediction, call it yhat.
            c. If ||yhat - y||/||y|| < thresh,
                continue; else, append Y[:,j] to D, 
                mark index j as an anomaly, and update the 
                estimate of the mean with a weighted average of the 
                existing mean and the new data point (should be a simple mean
                of all anomalies to that point).

        3. The output is a binary vector of the same size 
            as np.shape(Y)[1] indicating locations where 
            new dictionary entries were added (understood as anomalies).
    
    Inputs:
        Y: numpy array, shape (d,n); n vectors in dimension d.
    Outputs:
        anomalies: numpy array, shape (n,); True/False 
            vector indicating locations of anomalies; i.e., 
            updates to the memory/dictionary/exemplars.
    Optional inputs:
        *args:
        op: a function implementing the nonlinear similarity between 
            two vectors, which is the basis for most of the 
            corresponding nonlinear operators on matrices X \otimes Y. 
            Default: otimes2_ij, in the mset.py file.
            
        thresh: threshold parameter; if relative error in representing 
            the new datapoint is larger than this, then it is 
            added to the memory.
            Default: 0.1
            
        output_norms: Boolean. If True, then the values of the 
            relative error are output instead of the binary vector 
            of thresholds. Computing (anomalies < thresh) returns the 
            original output.
            Default: False.
            
        **kwargs:
        debug: boolean. If True, a pdb.set_trace() is executed at the 
            top of the code. Default: False
        verbosity: integer. If positive, updates are printed.
            Default: 0
    '''
    import numpy as np
    if kwargs.get('debug',False):
        import pdb
        pdb.set_trace()
    #
    verbosity = kwargs.get('verbosity',0)
    
    global _oDD     # storage for precomputed operator (D \otimes D)
    global _mu      # current estimate for mean
    global _mus     # the history of calculated means; non-essential.
    
    d,n = np.shape(Y)
    
    #anomalies = np.array(n, dtype=bool)
    norms = np.zeros(n, dtype=float)
    
    # track an estimate for the data mean, initialized with the 
    # first data point in Y.
    _mu = np.array( Y[:,0] )
    _mu.shape = (d,1)
    _mus = [ _mu ]
    
    # The second vector in Y will be the first dictionary entry.
    D = np.zeros( (d,1), dtype=float)
    y1 = Y[:,1]
    y1.shape = (d,1)
    D = y1 - _mu
    norms[0] = 0.
    _oDD = otimes(D,D, op=op)
    
    # main loop. 
    for j in range(2,n):
        if verbosity>0: print('Iteration %s : '%str(j).zfill(5), end='')
        
        yorig = np.array( Y[:,j] )
        yorig.shape = (d,1)
        
        ycurr = np.array( Y[:,j] )
        ycurr.shape = (d,1)
        ycurr -= _mu
        
        w = op_W( D, ycurr )
        
        ytil = np.dot(D, w)
        
        rel_err = np.linalg.norm( ytil - ycurr )/np.linalg.norm( ycurr )
        norms[j] = rel_err
        
        if verbosity>0: print('relative error %.2e; '%rel_err, end='')
        
        if rel_err < thresh:
            if verbosity>0: print('continuing.')
            continue
        else:
            if verbosity>0: print('appending datapoint to memory.')
            D = np.hstack( (D, ycurr) )
            
            # update the mean; store a history of previous values.
            nc = len( _mus ) +1
            _mu = ((nc-1.)*_mu + yorig)/(nc)
            _mus.append( _mu )
            
        #
    #
    
    if output_norms:
        # allow the user to play with thresholding
        return norms
    else:
        # apply a thresholding to define anomalies.
        return (norms >= thresh)
    #
#

#########################
#
# whole gamut + visualization
#

def visualize_mset(x,thresh,delay, verbosity=0):
    '''
    Applies the online MSET algorithm to a scalar time series x 
    (expected as a simple row vector/array) and specified 
    threshold and time delay parameters. (the embedded space 
    is always three dimensional here).
    '''
    import tde
    import numpy as np
    from matplotlib import pyplot

    X = tde.tde(x, delay=delay)

    norms = online_mset(X, output_norms=True, thresh=thresh, verbosity=verbosity)

    #############################
    #
    # visualize
    # 
    fig,ax = pyplot.subplots(3,1, 
                        figsize=(12,5), 
                        gridspec_kw={'height_ratios':[3,1,1]}, 
                        sharex=True)

    t = np.arange(len(x))

    t_d = t[2*delay:]

    ax[0].plot(t,x)
    ax[1].scatter(t_d,norms, s=2)

#    ymin = 10**int(np.floor(min(np.log10(norms[norms!=0.]))))
#    ymin = max(10**-5,ymin)
#    ymax = 10**int(np.ceil(max(np.log10(norms[norms!=0.]))))

#    ax[1].set_yscale('log')
    ymin,ymax = 0,0.2
    ax[1].set_ylim([ymin,ymax])
    
    yticks = [ymin,thresh,ymax]
    ax[1].set_yticks( yticks )
#    ax[1].set_yticklabels([r'$10^{%i}$'%np.log10(val) for val in yticks])
    
    ax[1].yaxis.grid()
    gls = ax[1].get_ygridlines()
    gls[1].set_color('r')

    # get locations of anomalies.
    anomalies = (norms>=thresh)
    where = np.where(anomalies)[0]
    where += 2*delay

    anom_windowed = np.convolve(anomalies, np.ones(delay//2)/(delay/2.), mode='same')
    ax[2].plot(t_d, anomalies, c='r')
    ax2r = ax[2].twinx()

    ax2r.plot(t_d, anom_windowed, c='k')

    ax[0].scatter(t[where], x[where], c='r', marker='o', s=50, alpha=0.8, zorder=1000)

    ax[0].set_title('timeseries (blue) with anomalies (red)', fontsize=16)
    ax[1].set_title('normed error in MSET representation', fontsize=16)
    ax[2].set_title('anomaly hits (red) and density (black)', fontsize=16)

    for axi in ax: axi.xaxis.grid()

    fig.tight_layout()

#    fig.show()
    return fig,ax
#

