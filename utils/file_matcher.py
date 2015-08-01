import os
import fnmatch

def file_filter(path,filters):
    try:
        files = os.listdir(path)
        for filter in filters:
            for f in fnmatch.filter(files, "*.%s" % filter):
                p = os.path.join(path, f)
                yield (p, filter)
    except:
        print "Catch an exception caught,but I don't know what to do"
        
if __name__ == '__main__':
    x = file_filter('c:\\src',['mp3'])
    for name,format in x:
        print name,format