import json
import os

from loadmodel import loadmodel
from loadmodel_3cnn import loadmodel_3cnn
from loadmodel_2gan import loadmodel_2gan
from loadmodel_3gan import loadmodel_3gan

from flask import Flask, render_template, request
from flask.wrappers import Response
from werkzeug.utils import secure_filename
from werkzeug.wrappers import response

app = Flask(__name__)  # reference this file

app.config['UPLOAD_PATH'] = 'static/uploads'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':

        # per_val = request.form['per_val']

        model_name = request.form['model_name']
        f = request.files['file']

        data_open = open('mydata.json')
        data_json = json.load(data_open)
        filename = data_json['file_name']

        print("Already Have File: ", filename)
        if(f.filename == ""):

            filename = data_json['file_name']

        else:
            filename = secure_filename(f.filename)

            # save the file to directory
            f.save(os.path.join(app.config['UPLOAD_PATH'], filename))

        name = {}
        name = {
            'file_name': filename,

            'model_name': model_name
        }
        # to store the filename to json file (mydata.json)
        with open('mydata.json', 'w') as j:
            json.dump(name, j)

        per_val = ''
        prediction = ''
        confidence = ''
        crack = 0
        wheeze = 0
        normal = 0

        if(model_name == 'CNN2'):
            result = loadmodel(filename)  # run loadmodel.py
            print(result)   # ABnormal, Very high , Percentage
            per_val = result[2]
            prediction = result[0]
            confidence = result[1]

            return render_template('uploaded.html', filename=filename, per_val=per_val, prediction=prediction, confidence=confidence)
        elif(model_name == 'CNN3'):
            result = loadmodel_3cnn(filename)  # run loadmodel.py
            print(result)   # Crack,Normal Wheeze , Very high , Percentage

            prediction = result[0]
            confidence = result[1]
            crack = result[2][0]
            normal = result[2][1]
            wheeze = result[2][2]

            return render_template('uploaded_cnn.html', filename=filename, crack=crack, normal=normal, wheeze=wheeze, prediction=prediction, confidence=confidence)
        elif(model_name == 'GAN2'):
            result = loadmodel_2gan(filename)  # run loadmodel.py
            print(result)   # ABnormal, Very high , Percentage
            per_val = result[2]
            prediction = result[0]
            confidence = result[1]
            return render_template('uploaded.html', filename=filename, per_val=per_val, prediction=prediction, confidence=confidence)
        else:
            result = loadmodel_3gan(filename)  # run loadmodel.py
            print(result)   # Crack,Normal Wheeze , Very high , Percentage

            prediction = result[0]
            confidence = result[1]
            crack = result[2][0]
            normal = result[2][1]
            wheeze = result[2][2]

            return render_template('uploaded_cnn.html', filename=filename, crack=crack, normal=normal, wheeze=wheeze, prediction=prediction, confidence=confidence)


@app.route('/', methods=["back"])
def go_back():
    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
