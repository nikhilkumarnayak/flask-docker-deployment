from flask import Flask,request,jsonify
import pandas as pd
import numpy as np
import pickle
import flasgger
from flasgger import Swagger


app = Flask(__name__)
Swagger(app)

## Read the pickle file and load it
pickle_in = open('classifier.pkl','rb')
classifier = pickle.load(pickle_in)

@app.route('/')
def welcome():
    return "Welcome All"

@app.route('/predict')
def predict_note_authentication():
    
    """Let's Authenticate the Bank Note
    This is using docstrings for specifications.
    ---
    parameters:
        - name: variance
          in: query
          type: number
          required: true
        - name: skewness
          in: query
          type: number
          required: true
        - name: curtosis
          in: query
          type: number
          required: true
        - name: entropy
          in: query
          type: number
          required: true
    responses:
        200:
            description: The output values
    
    """
    variance = request.args.get('variance')
    skewness = request.args.get('skewness')
    curtosis = request.args.get('curtosis')
    entropy = request.args.get('entropy')
    prediction = classifier.predict([[variance,skewness,curtosis,entropy]])
    print("The Predicted value is " + str(prediction))
    json_file = {}
    json_file['prediction'] = str(prediction)
    return jsonify(json_file)
    

@app.route('/predict_file',methods=["POST"])
def predict_note_file():
    
    """Let's Authenticate the Bank Note
    This is using docstrings for specifications.
    ---
    parameters:
        - name: file
          in: formData
          type: file
          required: true
    responses:
        200:
            description: The output values
    
    """
    df_test = pd.read_csv(request.files.get("file"))
    prediction = classifier.predict(df_test)
    print("The Predicted value of csv is " + str(list(prediction)))
    json_file = {}
    json_file['prediction'] = str(list(prediction))
    return jsonify(json_file)

    
if __name__ == '__main__':
    app.run()