from utils.file_matcher import file_filter
from core.melody_feature import *

def db_create(dir,format=['wav']):
    db = {}
    for file_name, _ in file_filter(dir,format):
        pv = file_to_pitch_vector(file_name)
        db[file_name] = pv

    return db

def query(db,pv):
    distance = {}
    for name,pitch in db.items():
        dis = pitch_vector_distance(pitch,pv)
        distance[name] = dis
    
    ordered = sorted(distance.items(), lambda x, y: cmp(x[1], y[1]))
    # name,value pair
    if(ordered):
        return ordered[0]
        
if __name__ == '__main__':
    db = db_create('data/train')
    test_file = 'data/test/mxml2.wma'
    pv = file_to_pitch_vector(test_file)
    name,dis = query(db,pv)
    
    print 'The best match is:'
    print '  name:',name,', distance:',dis
