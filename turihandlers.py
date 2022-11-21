#!/usr/bin/python

from pymongo import MongoClient
import tornado.web

from tornado.web import HTTPError
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from tornado.options import define, options

from basehandler import BaseHandler

import turicreate as tc
import pickle
from bson.binary import Binary
import json
import numpy as np
import base64

class PrintHandlers(BaseHandler):
    def get(self):
        '''Write out to screen the handlers used
        This is a nice debugging example!
        '''
        self.set_header("Content-Type", "application/json")
        self.write(self.application.handlers_string.replace('),','),\n'))

class UploadLabeledDatapointHandler(BaseHandler):
    def post(self):
        '''Save data point and class label to database
        '''
        data = json.loads(self.request.body.decode("utf-8"))

        vals = data['feature']
        fvals = vals
        label = data['label']
        sess  = data['dsid']

        dbid = self.db.labeledinstances.insert_one(
            {"feature":fvals,"label":label,"dsid":sess}
            );
        self.write_json({"id":str(dbid),
            "feature":[str(len(fvals))+" Points Received",
                    "min of: " +str(min(fvals)),
                    "max of: " +str(max(fvals))],
            "label":label})

class RequestNewDatasetId(BaseHandler):
    def get(self):
        '''Get a new dataset ID for building a new dataset
        '''
        a = self.db.labeledinstances.find_one(sort=[("dsid", -1)])
        if a == None:
            newSessionId = 1
        else:
            newSessionId = float(a['dsid'])+1
        self.write_json({"dsid":newSessionId})

class UpdateModelForDatasetId(BaseHandler):
    def get(self):
        '''Train a new model (or update) for given dataset ID
        '''
        dsid = self.get_int_arg("dsid",default=0)

        data = self.get_features_and_labels_as_SFrame(dsid)
        print(data)

        # fit the model to the data
        acc = -1
        best_model = 'unknown'
        if len(data)>0:
            
            model = tc.image_classifier.create(data,target='target',solver='fista')# training
            yhat = model.predict(data)
            self.clf.update({dsid: model})
            acc = sum(yhat==data['target'])/float(len(data))
            # save model for use later, if desired
            model.save('../models/turi_model_dsid%d'%(dsid))
            

        # send back the resubstitution accuracy
        # if training takes a while, we are blocking tornado!! No!!
        self.write_json({"resubAccuracy":acc})

    def get_features_and_labels_as_SFrame(self, dsid):
        # create feature vectors from database
        features=[]
        labels=[]
        targets=[]
        for i, a in enumerate(self.db.labeledinstances.find({"dsid":dsid})):
            image_64_decode = base64.b64decode(a['feature'])
            filename = "temp/%s%s.jpeg"%(a['label'],i)
            image_result = open(filename, 'wb') # create a writable image and write the decoding result
            image_result.write(image_64_decode)
            labels.append(a['label'])

        data = tc.image_analysis.load_images('./temp', with_path=True)
        unique_labels = set(labels)
        for path in data['path']:
            for l in unique_labels:
                if l in path:
                    targets.append(l)
                    break;

        data['target'] = targets

        # convert to dictionary for tc
        # data = {'target':labels, 'sequence':features}
        print(data)

        # send back the SFrame of the data
        return tc.SFrame(data=data)

class PredictOneFromDatasetId(BaseHandler):
    def post(self):
        '''Predict the class of a sent feature vector
        '''
        data = json.loads(self.request.body.decode("utf-8"))    
        fvals = self.get_features_as_SFrame(data['feature'])
        dsid  = data['dsid']

        # load the model from the database (using pickle)
        # we are blocking tornado!! no!!
        if(self.clf.get(dsid) == None):
            print('Loading Model From file')
            try:
                self.clf.update({dsid: tc.load_model('../models/turi_model_dsid%d'%(dsid))})
            except:
                print('No Model Trained for DSID:%d'%(dsid))

        # if the clf still does not contain the dsid ERROR
        if(self.clf.get(dsid) == None):
            predLabel = 'ERROR'
        else:
            predLabel = self.clf.get(dsid).predict(fvals);

        self.write_json({"prediction":str(predLabel)})

    def get_features_as_SFrame(self, vals):
        # create feature vectors from array input
        # convert to dictionary of arrays for tc

        image_64_decode = base64.b64decode(vals)
        filename = "guess/guess.jpeg"
        image_result = open(filename, 'wb') # create a writable image and write the decoding result
        image_result.write(image_64_decode)

        # send back the SFrame of the data
        return tc.image_analysis.load_images('./guess', with_path=True)
