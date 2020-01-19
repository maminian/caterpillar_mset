# rudimentary time delayed embedding
def tde(signal,delay=6*60,nd=3):
    '''
    rudimentary time-delayed embedding for a scalar signal.
    
    inputs:
        signal  : numpy array, shape (n,)
        delay   : integer; size of delay (default: 6*60)
        nd      : integer; number of delays (default: 3)
        
    outputs:
        tde_signal : numpy array, shape (n-delay*nd,nd+1)
    '''
    import numpy as np
    
    n, = np.shape(signal)   # will throw error if not expected input.
    
    sub_idx = np.arange(n - delay*(nd-1))
    
    inc = np.tile([k*delay for k in range(nd-1,-1,-1)], (len(sub_idx),1))
    
    inc = (inc.T + sub_idx).T
    
    tde_signal = np.array(signal)[inc]
    
    tde_signal = tde_signal.T
    
    return tde_signal
#
