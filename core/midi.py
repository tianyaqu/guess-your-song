from mido import MidiFile
    
class ErrorStr(Exception):
    def __init__(self,err_str):
        Exception.__init__(self,err_str)

class NoteSequencer(object):
    def __init__(self,file):
        try:
            self.mid = MidiFile(file)
            self.chs = {}
            self._load(self.mid)
        except Exception,e:
            raise ErrorStr,'midi file not exist,try another.'
    
    def _load(self,mid):
        tempo = 120.0
        scale_coff = 1.04
        last = {}
        for i, track in enumerate(mid.tracks):
            #print('Track {}: {}'.format(i, track.name))
            # track 0 as descriptor
            if(i == 0):
                for message in track:
                    #print message
                    if(message.type == 'set_tempo'):
                        tempo = 60000000.0/message.tempo
                        #how many miliseconds per ticks
                        scale_coff = (60*1000.0)/(tempo*mid.ticks_per_beat)
            
            # extract notes
            # i have manually change the track name to mainx for the main track
            if(track.name == 'mainx' or track.name == 'MAJOR_TRACK'):
                cur_time = 0
                for message in track:
                    cur_time += message.time
                    #print message
                    if(message.type == 'note_on' and message.velocity != 0):
                        if(not last.has_key(message.channel)):
                            last[message.channel] = []

                        ele = dict(note=message.note,start=cur_time,duration=0,open=True)
                        last[message.channel].append(ele)

                    elif((message.type == 'note_on') or message.type == 'note_off'):
                        #print last[message.channel]
                        #print '---',int(cur_time*scale_coff),message
                        ls = [ x for x in last[message.channel] if x['note'] == message.note]
                        len_ls = len(ls)
                        if(len_ls > 1 or len_ls <= 0):
                            #fix me
                            #what if got a key stucked? raise an error or be more fault-tolerable?
                            raise ErrorStr,'note on/off not match.'
                        if(ls[0]['open'] == True):
                            start = int(ls[0]['start']*scale_coff)
                            duration = int(message.time*scale_coff)
                            # append (note,start_time,duration) pair
                            if(not self.chs.has_key(message.channel)):
                                self.chs[message.channel] = []
                            self.chs[message.channel].append((message.note,start,duration))
                            last[message.channel].remove(ls[0])
                        else:
                            raise ErrorStr,'note on/off not match.'
                    elif(message.type == 'end_of_track'):
                        pass
    
    # return (note,start_time,duration) pair sequence
    def get_note_seq(self):
        for channel,note_seq in self.chs.iteritems(): 
            yield channel,note_seq
        
if __name__ == '__main__':
    s = NoteSequencer('eminem-love_the_way_you_lie_feat_rihanna.mid') 
    for k,v in s.get_note_seq():
        print 'chnnel:',k,v