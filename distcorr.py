import numpy as np
import scipy.spatial
import scipy.stats

def _dcorr(y, n2, A, dcov2_xx):
    """Helper function for distance correlation bootstrapping.
    """
    # Pairwise Euclidean distances
    b = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(y, metric='euclidean'))
    # Double centering
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()
    # Compute squared distance covariances
    dcov2_yy = np.vdot(B, B) / n2
    dcov2_xy = np.vdot(A, B) / n2
    return np.sqrt(dcov2_xy) / np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))


def distcorr(x, y, n_boot=1000, seed=None):
    """Compute the distance correlation between two N-dimensional arrays.
    
    Statistical significance (p-value) is evaluated with a permutation test.
    
    Parameters
    ----------
    x, y : np.ndarray
        Input arrays
    n_boot : int or None
        Number of bootstrap to perform.
        If None, no bootstrapping is performed and the function
        only returns the distance correlation (no p-value).
        Default is 1000 (thus giving a precision of 0.001).
    seed : int or None
        Random state seed.
        
    Returns
    -------
    dcor : float
        Distance correlation (range from 0 to 1).
    pval : float
        P-value.
    
    Notes
    -----
    From Wikipedia:
    
    Distance correlation is a measure of dependence between two paired 
    random vectors of arbitrary, not necessarily equal, dimension. The population 
    distance correlation coefficient is zero if and only if the random vectors 
    are independent. Thus, distance correlation measures both linear and 
    nonlinear association between two random variables or random vectors. 
    This is in contrast to Pearson's correlation, which can only detect 
    linear association between two random variables.
    
    This function uses the Joblib package for parallel bootstrapping.
    
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Distance_correlation
    .. [2] https://gist.github.com/satra/aa3d19a12b74e9ab7941
    .. [3] https://gist.github.com/wladston/c931b1495184fbb99bec
    .. [4] https://joblib.readthedocs.io/en/latest/
    
    Examples
    --------
    1. With two 1D vectors
    
        >>> a = [1, 2, 3, 4, 5]
        >>> b = [1, 2, 9, 4, 4]
        >>> distcorr(a, b, seed=9)
            (0.7626762424168667, 0.334)
    
    2. With two 2D arrays and no p-value
    
        >>> import numpy as np
        >>> np.random.seed(123)
        >>> a = np.random.random((10, 10))
        >>> b = np.random.random((10, 10))
        >>> distcorr(a, b, n_boot=None)
            0.8799633012275321
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    assert x.shape[0] == y.shape[0], 'x and y must have same number of samples'
    
    # Extract number of samples
    n = x.shape[0]
    n2 = n**2
    
    # Process first array to avoid redundancy when performing bootstrap
    a = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(x, metric='euclidean'))
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    dcov2_xx = np.vdot(A, A) / n2
    
    # Process second array and compute final distance correlation
    dcor = _dcorr(y, n2, A, dcov2_xx)

    # Compute p-value using a bootstrap procedure
    if n_boot is not None and n_boot > 1:
        # Define random seed and permutation
        rng = np.random.RandomState(seed)
        bootsam = rng.random_sample((n, n_boot)).argsort(axis=0)
        num_cores = multiprocessing.cpu_count()
        bootstat = [_dcorr(y[bootsam[:, i]], n2, A, dcov2_xx) for i in range(n_boot)]
        mean = np.mean(bootstat)
        stdev = np.std(bootstat-mean, ddof=1)
        pval = scipy.stats.norm.cdf(0, loc=np.abs(dcor-mean), scale=stdev)
        return dcor, pval
    else:
        return dcor
