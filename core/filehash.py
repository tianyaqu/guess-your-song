import numpy as np
import pydub

from fingerprint import fingerprint

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

if __name__ == '__main__':
    file = 'c:\\src\\a.mp3'
    file_hashes(file)
    
        