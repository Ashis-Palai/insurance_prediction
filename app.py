from flask import Flask, jsonify, request
import numpy as np
import joblib
import pandas as pd
import numpy as np
from sklearn import linear_model
from bs4 import BeautifulSoup
from flask_restful import Resource
from collections.abc import Mapping
import csv
import SimpleHTTPServer, SocketServer


# https://www.tutorialspoint.com/flask
import flask
app = Flask(__name__)


###################################################app



@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/index')
def index():
    return flask.render_template('test.html')


@app.route('/predict', methods=['POST','GET'])
def predict():
    file = request.files['myfile']
    data = pd.read_csv(file)

    clf = joblib.load('Porto_final_model.pkl')
    X_Test_final=data.drop(['id'],axis=1)
    Final_features = ['ps_car_08_cat', 'ps_ind_07_bin', 'ps_ind_04_cat', 'ps_car_04_cat', 'ps_ind_16_bin', 'ps_ind_12_bin', 'ps_ind_18_bin', 'ps_car_10_cat', 'ps_ind_03', 'ps_car_06_cat', 'ps_ind_17_bin', 'ps_car_01_cat', 'ps_car_15', 'ps_car_12', 'ps_ind_09_bin', 'ps_ind_06_bin', 'ps_ind_05_cat', 'ps_reg_01', 'ps_ind_15', 'ps_reg_02', 'ps_car_13', 'ps_car_07_cat', 'ps_ind_02_cat', 'ps_ind_08_bin', 'ps_ind_01', 'ps_car_02_cat', 'ps_car_09_cat', 'ps_car_11_cat']
    X_Test_final_with_imp_features=X_Test_final[Final_features]
    sub = pd.DataFrame()
    sub['id'] = data['id']
    sub['target'] = clf.predict(X_Test_final_with_imp_features)

    data_small = sub
    return flask.render_template('table.html', tables=[data_small.to_html()], titles=[''])
    
    #return 'Hare Krishna Hare Krishna Krishna Krishna Hare Hare \n Hare Rama Hare Rama Rama Rama Hare Hare'
# def predict():
#     f  = request.files['myfile']
#     df = pd.read_csv(f.stream).read()
#     # df = pd.read_csv(request.files["myfile"])
#     # x = pd.read_csv(recieved_file)
#     return df.head(10)

    
# 
#     clf = joblib.load('Porto_final_model.pkl')
# 
#     recieved_file = request.files
#     review_text = clean_text(to_predict_list['review_text'])
#     pred = clf.predict(count_vect.transform([review_text]))
#     if pred[0]:
#         prediction = "Positive"
#     else:
#         prediction = "Negative"

#     return jsonify({'prediction': prediction})


PORT = 8000
httpd = SocketServer.TCPServer(("", PORT), SimpleHTTPServer.SimpleHTTPRequestHandler)
httpd.allow_reuse_address = True
httpd.serve_forever()

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=PORT)
