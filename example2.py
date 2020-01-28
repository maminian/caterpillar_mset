'''
Another example of MSET to a scalar time series 
by applying a time delayed embedding. 

This version builds a fixed training set 
chosen from non-anomalous data (decided by eye), 
then manually finds the relative error of that fixed 
model on the entire data set.
'''

import numpy as np
import mset
import tde
from matplotlib import pyplot

n = 1000
thresh = 0.1

t = np.linspace(0, 10*np.pi, n)
x = np.sin(t) + 0.1*np.random.randn(n)

# introduce a new type of signal halfway through.
heavy = (1 + np.tanh( 0.5*(t-5*np.pi) ))/2.
x += heavy*np.sin(t/2) + heavy*(-np.sin(t))

# time delayed embedding based on analytical 
# zero-autocorrelation time of pi/2 for a sinusoid.
# generically this requires you have some knowledge of 
# your problem of interest, or additional observation time 
# sufficient to get a numerical zero-autocorrelation time.
delay = int((np.pi/2)/(t[1] - t[0]))

Y = tde.tde(x, delay=delay)

# train on roughly the first period of the sinusoid.
training = np.arange(4*delay, dtype=int)

print('fitting MSET model to fixed training set')
mset.fit(Y[:,training])

# predict everything.
print('predicting with fixed dictionary')
Yhat = mset.predict(Y, verbosity=1)

print('evaluating error and visualizing.')
err = np.linalg.norm(Y - Yhat, axis=0)
rel_err = err/np.linalg.norm(Yhat, axis=0)

fig,ax = pyplot.subplots(2,1, figsize=(10,6), gridspec_kw={'height_ratios':[2,1]}, sharex=True)
ax[0].plot(t[2*delay:], Y[0,:], label='original')
ax[0].plot(t[2*delay:], Yhat[0,:], label='MSET fit')
ax[0].plot(t[2*delay:][training], Y[0,training], c='g', alpha=0.5, lw=4, label='training data')
ax[0].legend(loc='upper left')
ax[0].set_ylabel(r'$x(t)$', fontsize=14)

ax[1].plot(t[2*delay:], rel_err, c='r')
ax[1].set_xlabel('time', fontsize=14)
ax[1].set_ylabel('rel. error', fontsize=14)

fig.tight_layout()
#fig.savefig('example2_vis.png', dpi=120, bbox_inches='tight')

fig.show()
