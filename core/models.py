from mongoengine import *
from bson import objectid
import datetime

MONGO_DB = 'songs'

class Song(Document):
    name = StringField(required=True)
    path = StringField(required=True)
    desc = StringField()
    created_at = DateTimeField(default=datetime.datetime.now)

class Print(Document):
    song = ReferenceField(Song)
    hash = StringField(required=True,min_length=20,max_length=20)
    offset = IntField(required=True,default=-1)
    created_at = DateTimeField(default=datetime.datetime.now)