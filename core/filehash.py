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
    index = sequence[l].argmax()
    loc = np.where(l)[0]
    offset = loc[index]
    return sequence[offset],offset
    
def freq_to_pitch(freqs):
    log440 = 8.78135971
    pitches_array = np.array(freqs)
    pitches = 12 * (np.log2(pitches_array) - log440) + 69
    return pitches
    
    
def extract_pitches(filename,fs=DEFAULT_FS):
    channels = get_raw_from_file(filename)
    #for channel in channels:
    for window in sliding_window(channels[0]):
        r = acf(window)
        value,offset = find_max(r)
        #print value,offset
        if(value < 0.01):
            yield -1
        else:
            yield round(1.0*fs/(offset))
        
if __name__ == '__main__':
    file = 'c:\\src\\w.mp3'
    noteStrings = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    r = []
    for x in extract_pitches(file):
        r.append(noteStrings[int(x)%12])
    print r
    #file_hashes(file)
    
        