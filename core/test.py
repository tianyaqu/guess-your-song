from melody_feature import *

if __name__ == '__main__':
    file = 'alphaville-forever_young.mid'
    for k,note in note_from_midi_test(file):
        print k
        name = 'forever_youngy' + str(k)+'.txt'
        vector_to_file(note,name)