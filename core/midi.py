from mido import MidiFile
    
class ErrorStr(Exception):
    def __init__(self,err_str):
        Exception.__init__(self,err_str)

class NoteSeq(object):
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
        for i, track in enumerate(mid.tracks):
            # track 0 as descriptor
            if(i == 0):
                for message in track:
                    if(message.type == 'set_tempo'):
                        tempo = 60000000.0/message.tempo
                        scale_coff = (60*1000.0)/(tempo*mid.ticks_per_beat)
            
            # extract notes
            else:
                x = dict(chn=0,note=0,start=0,duration=0,open=False)
                for message in track:
                    if(message.type == 'note_on' and message.velocity != 0):
                        x['note'] = message.note
                        x['chn'] = message.channel
                        x['start'] = x['start'] + x['duration']
                        x['open'] = True
                    elif((message.type == 'note_on') or message.type == 'note_off'):
                        if(x['open'] == True and x['chn'] == message.channel and x['note'] == message.note):
                            #in ms
                            duration = int(message.time*scale_coff)
                            x['duration'] = duration
                            x['open'] = False
                            
                            # append (note,start_time,duration) pair
                            if(self.chs.has_key(message.channel)):
                                self.chs[message.channel].append((message.note,x['start'],duration))
                            else:
                                self.chs[message.channel] = []
                                self.chs[message.channel].append((message.note,x['start'],duration))
                        else:
                            raise ErrorStr,'note on/off not match.'
                    elif(message.type == 'end_of_track'):
                        pass
    
    # return (note,start_time,duration) pair sequence
    def get_note_seq(self):
        for channel,note_seq in self.chs.iteritems(): 
            yield channel,note_seq
        
if __name__ == '__main__':
    s = NoteSeq('00003.mid') 
    for k,v in s.get_note_seq():
        print v