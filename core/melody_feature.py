import numpy as np
import matplotlib.pyplot as plt
import pydub

from filter import *
from dtw import dtw

def get_raw_from_file(filename):
    container = pydub.AudioSegment.from_file(filename)
    if container.sample_width == 1:
        data = np.fromstring(container._data, np.int8)
    else:
        data = np.fromstring(container._data, np.int16)
    
    data = data - data.mean()
    channels = []
    for chn in xrange(container.channels):
        channels.append(data[chn::container.channels])
    return channels,container.frame_rate
    
def freq_to_notes(freqs):
    log440 = 8.78135971
    notes_array = np.asarray(freqs)
    notes = 12 * (np.log2(notes_array) - log440) + 69
    return notes

def note_segment(sequence,thresh=1):
    x = np.asarray(sequence)
    y =  np.array([])
    start = end = 0
    for index in range(x.size-1):
        if(abs(x[index] - x[index+1]) < thresh):
            end = index + 1
            if(end == x.size-1):
                #print 's',start,index+1,x[start:x.size]
                avg = np.median(x[start:x.size])
                x[start:x.size] = avg
                if(x.size - start >= 1):
                    #print 'append'
                    y = np.append(y,x[start:x.size])
        else:
            #print start,index+1, x[start:index+1]
            avg = np.median(x[start:index+1])
            x[start:index+1] = avg
            if(index+1 - start >= 1):
                #print 'b append'
                y = np.append(y,x[start:index+1])
            start = index + 1

        index += 1

    return y

def cal_energy(sequence):
    data = np.asarray(sequence)
    return data.max()
    #return abs(data).sum()
    #return 1.0*abs(data).sum()/data.size
    
def frame_to_pitch(frame,fs,thresh):
    frame_x = np.asarray(frame)
    invalid = -1
    if cal_energy(frame_x) < thresh:
        return invalid
    else:
        down_limit = 40
        up_limit = 1000
        n1 = int(round(fs*1.0/up_limit))
        n2 = int(round(fs*1.0/down_limit))
        frame_acf = acf(frame_x)
        frame_acf[np.arange(n1)] = invalid
        if n2 < frame_acf.size:
            frame_acf[np.arange(n2,frame_acf.size)] = invalid
        index = frame_acf.argmax()
        #max = frame_x[index]
        #print n1,n2,max,index
        pitch = fs/(index - 1.0)
        return pitch
    
def extract_pitches(filename):
    channels,fs = get_raw_from_file(filename)
    ws = int(round(40*fs/1000.0))
    data = channels[0]
    energy = cal_energy(data)
    thresh = 0.3*energy
    result = []
    for window in sliding_window(data,ws):
        pitch = frame_to_pitch(window,fs,thresh)
        result.append(pitch)
    return result
    
def pitch_vector_distance(pa,pb):
    la = ~np.isnan(pa)
    lb = ~np.isnan(pb)
    x = pa[la]
    y = pb[lb]
    
    dist, cost, path = dtw(x,y)
    return dist

# dump pitch vector to file
def vector_to_file(t,file):
    s = ''.join(map(lambda x:str(x)+',',t))
    with open(file,'wb') as f:
        f.write(s)

def file_to_pitch_vector(file):
    r = extract_pitches(file)
    #t = freq_to_notes(r1)
    r1 = freq_to_notes(r)
    t = median_filt(r1,5)
    
    return t

#pitch based lsh
def plsh(file):
    pv = file_to_pitch_vector(file)
    for x in sliding_window(pv,ws=60):
        note = x[::3]
        #very helpful to set nan as 0
        loc = np.isnan(note)
        note[loc] = 0
        yield note,file

# note based lsh
def nlsh(file):
    pv = file_to_pitch_vector(file)
    for x in sliding_window(pv,ws=60):
        note = x[::3]
        #[::3]
        #very helpful to set nan as 0
        loc = np.isnan(note)
        note[loc] = 0
        #note = x
        yield note,file
            
if __name__ == '__main__':
    from lshash import LSHash

    hash_len = 10
    dm = 20

    lsh = LSHash(hash_len, dm)
    f1 = 'xml.wav'
    f2 = 'mxml2.wma'
    f3 = 'soo.wav'
    f4 = '10-little-indians.wav'
    f5 = 'xyx.wav'
    f6 = '00001.mid'

    pv = []
    for note,name in plsh('wenrou_org.wav'):
        pv.extend(note)
    #seg = note_segment(pv)
    vector_to_file(pv,'f6.txt')
    """
    for note,name in plsh(f1):
        lsh.index(note,extra_data=(name,0.8))
    for note,name in plsh(f3):
        lsh.index(note,extra_data=(name,0.8))
    for note,name in plsh(f4):
        lsh.index(note,extra_data=(name,0.8))
    for note,name in plsh(f5):
        lsh.index(note,extra_data=(name,0.8))
        
    for note,name in plsh(f2):
        q = note
        r = lsh.query(q)
        print '-------------------'
        if(len(r) > 0):
            print len(r)
            print r[0][0]
    
    """
    """
    p1 = file_to_pitch_vector(f1)
    p2 = file_to_pitch_vector(f2)
    p3 = file_to_pitch_vector(f3)
    p4 = file_to_pitch_vector(f4)

    #vector_to_file(p1,'p1.txt')
    #vector_to_file(p2,'p2.txt')
    #vector_to_file(p3,'p3.txt')
    print pitch_vector_distance(p1,p2)
    print pitch_vector_distance(p2,p3)
    print pitch_vector_distance(p2,p4)
    plt.plot(p1,'red')
    plt.plot(p2,'green')
    plt.plot(p3,'blue')
    plt.plot(p4,'yellow')
    plt.show()
    """
    
        