
# coding: utf-8

# In[2]:

import os
def index_hipchat():
    c = 0;
    es = elasticsearch.Elasticsearch()
    for r,d,f in os.walk('/Users/britta/tmp/spark_poc/data/hipchat_export/rooms'):
        for afile in f:
            if afile[-4:] == 'json':
                index(os.path.join(r,afile), es)
                c = c+1


# In[3]:

import json, requests, pprint, random, math, operator, datetime, sys, optparse, time, elasticsearch
def create_index():
    es = elasticsearch.Elasticsearch()
    es.indices.delete("hipchat")
    mapping = {
          "mappings": {
            "_default_": {
              "properties": {
                "message": {
                  "type": "string",
                  "term_vector": "yes"
                },
                "from": {
                  "properties": {
                    "name": {
                      "type": "string",
                      "analyzer": "keyword"
                    },
                    "user_id": {
                      "type": "string"
                    }
                  }
                }
              }
            }
          },
          "settings": {
            "index.number_of_shards": 4
          }
        }
    es.indices.create(index="hipchat",body=mapping)

        


# In[4]:

import json, requests, pprint, random, math, operator, datetime, sys, optparse, time, elasticsearch
def index(filename, es):
    
    path, file = os.path.split(filename)
    folders = path.split('/')
    room = folders[len(folders)-1].replace(' ', '').replace('#', '')
    f = open(filename, 'r')
    try :
        docs=f.read()
        docs = docs.replace('\n',' ')
        for doc in json.loads(docs):
            try :
                es.index(body = json.dumps(doc), index = "hipchat", doc_type = room)
            except  :
                pass
                
    except ValueError:
        print 'not valid json? ' + filename
        print docs
    
    


# In[1]:

create_index()
index_hipchat()


# In[12]:




# In[ ]:



