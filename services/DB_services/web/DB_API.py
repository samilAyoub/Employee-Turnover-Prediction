from flask import Flask, request, jsonify
import pymongo
from flask_cors import CORS, cross_origin
from bson.json_util import dumps
from bson.json_util import loads

client_url = 'mongodb+srv://model:DKVMyavEwrK76jw@cluster0-g7co1.mongodb.net'
db_name = 'Employee_attr'
collection_name = 'models'

app = Flask(__name__)
CORS(app)

@app.route('/metadata', methods=['GET'])
def getModelsInfos():

    client = pymongo.MongoClient(client_url)
    
    db = client[db_name]

    collection = db[collection_name]
    projection = {  'model_name': 1, 
                    'predictors':1,
                    'target':1,
                    'f1':1, 
                    'recall': 1, 
                    'precision':1, 
                    'auc': 1, 
                    'train_time':1,
                    'predict_end_point': 1,
                    'test_end_point': 1,
                    'train_end_point': 1
                    }
    cursor = collection.find({}, projection)

    models_metadata  = loads(dumps(cursor))
        
    list(map(lambda item: item.pop('_id'), models_metadata)) #remove id from data

    return jsonify(models_metadata)
    

if __name__ == '__main__':
    app.run('0.0.0.0', 5000)