import pymongo
import pickle
import time

def preprocess(data, predictors_name, target_name):
    predictors = []
    target = []
    list(map(lambda x:  target.append(x.pop(target_name)), data))
    predictors = projection(data, predictors_name)
    return predictors, target


def encode_target(target):
    target_map = {'Yes':1, 'No':0}
    return list(map(lambda item: target_map[item], target))

def decode_target(target):
    target_map = {1:'Yes', 0:'No'}
    return list(map(lambda item: target_map[item], target))

def updateModel(model, model_name, client_url, db_name, collection_name):
    #pickling the model
    pickled_model = pickle.dumps(model)
    
    #saving model to mongoDB
    #creating connection
    client = pymongo.MongoClient(client_url)
    
    #creating or finding database in mongodb
    db = client[db_name]
    
    #creating or finding collection
    collection = db[collection_name]
    info = collection.update_one({ 
        'model_name': model_name
        }, {
            '$set': {
                    'model': pickled_model,
                    }
            }, upsert=False
        )
    
    return True

def updateModelScores(model, model_name, client_url, db_name, collection_name, f1_score, precision_score, recall_score, auc_score, train_time):
    #creating connection
    client = pymongo.MongoClient(client_url)
    
    #creating or finding database in mongodb
    db = client[db_name]
    
    #creating or finding collection
    collection = db[collection_name]
    info = collection.update_one({ 
        'model_name': model_name
        }, {
            '$set': {
                    'f1': f1_score,
                    'precision': precision_score,
                    'reacll': recall_score,
                    'auc': auc_score,
                    'train_time': train_time
                    }
            }, upsert=False
        )

    return True    



def loadByName(model_name, client_url, db_name, collection_name):
    json_data = {}
    
    #loading model from mongoDB
    #creating connection
    client = pymongo.MongoClient(client_url)
    
    #finding database in mongodb
    db = client[db_name]
    
    #finding collection
    collection = db[collection_name]
    data = collection.find({'model_name': model_name})
    
    for i in data:
        json_data = i
    #fetching model from db
    pickled_model = json_data['model']
    model_name = json_data['model_name']
    train_time = json_data['train_time']
    precision = json_data['precision']
    recall = json_data['recall']
    f1 = json_data['f1']
    auc = json_data['auc']    

    response = pickle.loads(pickled_model)

    return response

def getPredTargetNamesName(model_name, client_url, db_name, collection_name):
    #loading model from mongoDB
    #creating connection
    client = pymongo.MongoClient(client_url)
    
    #finding database in mongodb
    db = client[db_name]
    
    #finding collection
    collection = db[collection_name]
    data = collection.find({'model_name': model_name}, {'predictors': 1, 'target': 1})
    json_data = {}
    for i in data:
        json_data = i

    return json_data['predictors'], json_data['target'];

def projection(dictList, predictors_name):
   
    return  [dictProjection(dict, predictors_name) for dict in dictList]


def dictProjection(dict, predictors_name):
    response = {}
    for(key, value) in dict.items():
        if key in predictors_name:
            response[key] = value  
    return response        

