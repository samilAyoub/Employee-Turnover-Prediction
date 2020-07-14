from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import ShuffleSplit
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from flask_cors import CORS, cross_origin
import my_libarary

pipeline = None
model_name = 'GB' 
client_url = 'mongodb+srv://model:DKVMyavEwrK76jw@cluster0-g7co1.mongodb.net'
db_name = 'Employee_attr'
collection_name = 'models'

def loadPipeline():
    global pipeline

    #retrive model from database
    pipeline = my_libarary.loadByName(model_name, client_url, db_name, collection_name)
    


# initialize the flask app and restful api 
app = Flask(__name__)
cors = CORS(app)

@app.route('/gb/predict', methods=['POST'])
def predict():

    predictors_name, target_name = my_libarary.getPredTargetNamesName(model_name, client_url, db_name, collection_name)

    global pipeline

    if (pipeline == None):
        loadPipeline()

    #get data as json
    dictList = request.get_json()

    #projection
    projected = my_libarary.projection(dictList, predictors_name)

    #load json to a pandas dataframe 
    features_df = pd.DataFrame(projected)

    #preprocessing data and get the prediction(1 or 0)
    predictions = pipeline.predict(features_df).tolist() 

    #decode predictions
    predictions_decd = my_libarary.decode_target(predictions)

    #prepare the response
    response = []
    for point, pred in zip(projected, predictions_decd):
            point['Attrition_pred'] = pred
            response.append(point)

    return jsonify(response)

@app.route('/gb/train', methods=['POST'])
def retrain(): 

    global pipeline

    predictors_name, target_name = my_libarary.getPredTargetNamesName(model_name, client_url, db_name, collection_name)

    if (pipeline == None):
        loadPipeline()

    #get data as json
    data = request.get_json()

    predictors_name, target_name = my_libarary.getPredTargetNamesName(model_name, client_url, db_name, collection_name)

    #separate features and target
    features, target = my_libarary.preprocess(data, predictors_name, target_name)

    #encode target
    target_enc = my_libarary.encode_target(target)

    #load json to a pandas dataframe 
    features_df = pd.DataFrame(features)

    #train the module 
    pipeline.fit(features_df, target_enc)

    #save the model on database
    my_libarary.updateModel(pipeline, model_name, client_url, db_name, collection_name)

    #prepare the response
    return jsonify({
            'status': 'the model is fited'
        }) 

@app.route('/gb/test', methods=['POST'])
def test_module():

    global pipeline

    predictors_name, target_name = my_libarary.getPredTargetNamesName(model_name, client_url, db_name, collection_name)

    if (pipeline == None):
        loadPipeline()

    #get data as json
    json = request.get_json()
    data = json['dataset']
    n_splits = json['n_splits']
    stratified = json['stratified']
    shuffle = json['shuffle']
    n_jobs = json['n_jobs']
    

    predictors_name, target_name = my_libarary.getPredTargetNamesName(model_name, client_url, db_name, collection_name)

    #separate features and target
    features, target = my_libarary.preprocess(data, predictors_name, target_name)

    #load json to a pandas dataframe 
    features_df = pd.DataFrame(features)

    #encode target
    target_enc = my_libarary.encode_target(target)

    #cross_validate function and multiple metric evaluation
    cv = 5 #default value
    if(stratified):
        cv = StratifiedKFold(n_splits=n_splits, shuffle=shuffle)
    elif(shuffle):
        cv = ShuffleSplit(n_splits=n_splits)
    else:
        cv = n_splits

    scoring = ['f1','recall','precision', 'roc_auc']
    cv_score  = cross_validate(pipeline, features_df, target_enc, n_jobs=n_jobs,cv=cv,return_train_score=True, scoring=scoring)
    auc = np.mean(cv_score['test_roc_auc'].tolist())
    f1 = np.mean(cv_score['test_f1'].tolist())
    precision = np.mean(cv_score['test_precision'].tolist())
    recall = np.mean(cv_score['test_recall'].tolist())
    train_time = np.mean(cv_score['test_recall'].tolist())
    my_libarary.updateModelScores(model_name,model_name,client_url, db_name, collection_name, f1, precision, recall, auc, train_time )
    
    return jsonify({
       'score_roc_auc': auc,
       'score_f1': f1,
       'score_precision': precision,
       'score_recall': recall,
       'train_time': train_time
        }) 


if __name__ == '__main__':
    app.run('0.0.0.0', 5001)
    

