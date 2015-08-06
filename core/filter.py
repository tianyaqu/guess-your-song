import numpy as np

######################################################################
# Sampling rate, related to the Nyquist conditions, which affects
# the range frequencies we can detect.
DEFAULT_FS = 44100

######################################################################
# Size of the frame window
DEFAULT_WINDOW_SIZE = 4096

######################################################################
# time of the frame duration,in milliseconds
DEFAULT_FRAME_DURATION = 32


def sliding_window(sequence,ws=DEFAULT_WINDOW_SIZE,shift_ratio=1):
    """Returns a generator that will iterate through
    the defined chunks of input sequence.  Input sequence
    must be iterable."""
    shift = int(shift_ratio*ws)
    # Verify the inputs
    try: it = iter(sequence)
    except TypeError:
        raise Exception("**ERROR** sequence must be iterable.")
    if not ((type(ws) == type(0)) and (type(shift) == type(0))):
        raise Exception("**ERROR** type(ws) and type(shift) must be int.")
    if shift > ws:
        raise Exception("**ERROR** shift must not be larger than ws.")
    if ws > len(sequence):
        raise Exception("**ERROR** ws must not be larger than sequence length.")
 
    # Pre-compute number of chunks to emit
    num_frames = ((len(sequence)-ws)/shift)+1
 
    # Do the work
    for i in range(0,num_frames*shift,shift):
        yield sequence[i:i+ws]

def acf(sequence):
    n = len(sequence)
    data = np.asarray(sequence)
    mean = np.mean(data)
    c0 = np.sum((data - mean) ** 2) / float(n)

    def r(h):
        acf_lag = ((data[:n - h] - mean) * (data[h:] - mean)).sum() / float(n) / c0
        return round(acf_lag, 3)
    x = np.arange(n) # Avoiding lag 0 calculation
    acf_coeffs = map(r, x)
    return np.asarray(acf_coeffs)

def median_filt(data, k):
    """Apply a length-k median filter to a 1D array x.
    Boundaries are extended by repeating endpoints.
    """
    assert k % 2 == 1, "Median filter length must be odd."
    x = np.asarray(data)
    assert x.ndim == 1, "Input must be one-dimensional."
    k2 = (k - 1) // 2
    y = np.zeros ((len (x), k), dtype=x.dtype)
    y[:,k2] = x
    for i in range (k2):
        j = k2 - i
        y[j:,i] = x[:-j]
        y[:j,i] = x[0]
        y[:-j,-(i+1)] = x[j:]
        y[-j:,-(i+1)] = x[-1]
    return np.median(y, axis=1)
    
def del_outlier_pitches(pv,thresh=10):
    adjusted = pv - np.median(pv)
    loc = (abs(adjusted) > thresh)
    pv[loc] = 0
    return pv