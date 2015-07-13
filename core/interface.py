import sys
import os
sys.path.append('..')
from models import *
from utils.file_matcher import file_filter
from filehash import *

def train_song_prints(path,filters):
    for file, _ in file_filter(path,filters):
        print file
        song = Song(name=file,path=file,desc='')
        song.save()
        hashes = file_hashes(file)
        for x in hashes:
            px = Print(song=song,hash=x[0],offset=x[1])
            px.save()

def fetch_similimar(hashes,n):
    mapper = {}
    largest_count = 0
    largest = 0
    song_id = None
    hash_set = [x[0] for x in hashes]
    print len(hashes)
    xx = {}
    
    for px in Print.objects(hash__in=hash_set):
        for hash,offset in hashes:
            if hash != px.hash:
                continue
            if px.hash not in xx:
                xx[px.hash] = 0
            xx[px.hash] += 1
            diff = offset - px.offset
            song = px.song
            if diff not in mapper:
                mapper[diff] = {}
            if song not in mapper[diff]:
                mapper[diff][song] = 0
            mapper[diff][song] += 1
            
            if mapper[diff][song] > largest_count:
                largest = diff
                largest_count = mapper[diff][song]
                song_id = song
    print xx
    return song_id.name,largest_count

def guess_from_snippet(file):
    hashes = file_hashes(file)
    name,count = fetch_similimar(hashes,0)
    print name,count
        
if __name__ == '__main__':
    connect(MONGO_DB)
    #train_song_prints('c:\\src',['test'])
    guess_from_snippet('c:\\src\\wr.wma')
