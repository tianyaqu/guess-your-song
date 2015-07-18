import numpy as np
import matplotlib.pyplot as plt
import pydub

from fingerprint import fingerprint,sliding_window,acf,DEFAULT_FS

def get_raw_from_file(filename):
    container = pydub.AudioSegment.from_file(filename)
    data = np.fromstring(container._data, np.int16)

    channels = []
    for chn in xrange(container.channels):
        channels.append(data[chn::container.channels])
    return channels
    
def generate_hashes(data):
    hashes = fingerprint(data,wsize = 4096,wratio = 0.5)
    return set(hashes)

def file_hashes(filename):
    channels = get_raw_from_file(filename)
    r = set()
    for data in channels:
        r |= generate_hashes(data)
    return r

def find_local_max(sequence):
    return np.r_[True, sequence[1:] > sequence[:-1]] & np.r_[sequence[:-1] > sequence[1:], True]

def find_max(sequence):
    l = find_local_max(sequence)
    # skip the first
    l[0] = False
    try:
        index = sequence[l].argmax()
    except:
        return 0,1
    loc = np.where(l)[0]
    offset = loc[index]
    return sequence[offset],offset
    
def freq_to_notes(freqs):
    log440 = 8.78135971
    notes_array = np.asarray(freqs)
    notes = 12 * (np.log2(notes_array) - log440) + 69
    return notes

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

def cal_energy(sequence):
    data = np.asarray(sequence)
    return 1.0*abs(data).sum()/data.size
    
def extract_pitches(filename,fs=DEFAULT_FS):
    channels = get_raw_from_file(filename)
    #for channel in channels:
    data = channels[0]
    energy = cal_energy(data)
    result = []
    for window in sliding_window(data):
        pitch = 1
        if cal_energy(window) < 0.3 * energy:
            pass
        else:
            r = acf(window)
            value,offset = find_max(r)
            #print value,offset
            if(value < 0.01):
                #yield -1
                pass
            else:
                pitch =  round(1.0*fs/(offset))
        result.append(pitch)
    return result
        
if __name__ == '__main__':
    file = 'c:\\src\\drm.wma'
    noteStrings = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    r = []
    for x in extract_pitches(file):
        r.append(x)
    print r
    filter = median_filt(r,5)

    plt.figure()
    plt.plot(filter)
    #plt.plot(r)
    plt.show()
    
    k = freq_to_notes(filter)
    nn =[]
    for x in k:
        nn.append(noteStrings[int(x)%12])
    print nn
    #file_hashes(file)
    
        