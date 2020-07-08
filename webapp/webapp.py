import datetime
import os
import numpy as np

#from webapp import app, api, db

#from webapp import app

from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy
import tensorflow as tf
from astrophpredict import AstrophPrediction
from tensorflow.contrib.keras.api.keras import backend as K

import pymysql
pymysql.install_as_MySQLdb()
import MySQLdb

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def main():
   #K.clear_session()
   prediction = AstrophPrediction()
   

   now = datetime.datetime.now()
   timeString = now.strftime("%Y-%m-%d %H:%M")
   cpuCount = os.cpu_count()
   templateData = {
      'title' : 'Web App for classifying abstracts on statistics',
      'time': timeString,
      'cpucount' : cpuCount,
      'tfversion' : tf.__version__
      }
   
   if request.method == 'GET':
      return render_template('main.html', **templateData)
   elif request.method == 'POST':
      resultText = "You wrote: " + request.form['myTextArea']

      probability = prediction.predict([request.form['myTextArea']])[0] 
      ind = np.argmax(probability)
      label = prediction.target_fullname[ind]


      results = {'text' : resultText, 
                 'label' : label}
      return render_template('main.html', results=results, **templateData)

if __name__ == "__main__":
   app.run(host='0.0.0.0', port=5000, debug=True, threaded=False)
