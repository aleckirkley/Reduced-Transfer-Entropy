import numpy as np
from collections import Counter
from mpmath import loggamma

def logfact(n):
    """
    log of factorial
    """
    return loggamma(n+1)

def logchoose(n,k):
    """
    log of binomial coefficient
    """
    return logfact(n) - logfact(k) - logfact(n-k)


def TE(x,y,k=1,l=1,reduced=True,norm=True,stirling=False,constant_correction=False):

    """
    Inputs:
    x,y: arrays of discrete or binned time series values
    k,l: lags for x and y respectively
    reduced: boolean for using the finite-size corrected TE. default True
    norm: boolean for mapping TE to [-1,1] (reduced measure) or [0,1] (non-reduced measure). default True
    stirling: whether to stirling approximate log factorials to compute Shannon entropies. default False
    constant_correction: use correction obtained from subtracting off approximate ensemble average. default False

    Returns:
    TE_final: transfer entropy value, incorporating the requested normalization/reduction/approximations
    """

    # construct time delay embeddings for future y, past y, and past x
    mint = max(k,l)
    embedding_array = []
    embedding_array.append(y[mint:])
    for t in range(1,l+1): embedding_array.append(y[mint-t:-t])
    for t in range(1,k+1): embedding_array.append(x[mint-t:-t])
    embedding = np.array(embedding_array).T
    embedding = [tuple(t) for t in embedding.tolist()]

    # compute count distributions (contingency tables) for pairs of series for joint entropy calculations
    n123 = Counter(embedding)
    n12 = Counter([tuple(tt.tolist()) \
                   for tt in np.concatenate([[tuple(list(t)[:-k])]*n123[t] for t in n123])])
    n23 = Counter([tuple(tt.tolist()) \
                   for tt in np.concatenate([[tuple(list(t)[1:])]*n123[t] for t in n123])])
    n2 = Counter([tuple(tt.tolist()) \
                   for tt in np.concatenate([[tuple(list(t)[1:-k])]*n123[t] for t in n123])])

    # apply stirling approximation to factorials if requested
    if stirling:
        def lfact(z): return z*np.log(z)
    else:
        def lfact(z): return logfact(z)

    # compute unreduced transfer entropy using four joint entropy terms    
    TE_raw = sum(lfact(n) for n in n123.values()) + sum(lfact(n) for n in n2.values()) \
           - sum(lfact(n) for n in n12.values()) - sum(lfact(n) for n in n23.values())

    # apply reduction if requested
    C = len(set(x).union(set(y)))
    if reduced:
        correction = sum(logchoose(n+C-1,C-1) for n in n2.values()) \
                    - sum(logchoose(n+C-1,C-1) for n in n23.values())
        TE_final = TE_raw + correction

    elif constant_correction:
        correction = -(C**l)*(C**k-1.)*(C-1)/(2.)
        TE_final = TE_raw + correction

    else:
        TE_final = TE_raw

    # apply normalization if requested
    if norm:
        
        if np.abs(TE_final) < 1e-10: 
            return 0.
        
        CE = sum(lfact(n) for n in n2.values()) - sum(lfact(n) for n in n12.values())
        
        if reduced or constant_correction:
            
            if TE_final > 0:
                ub = CE + correction
                
            else:
                ub = -correction
        else:
            ub = CE
        
        TE_final /= ub

    return TE_final