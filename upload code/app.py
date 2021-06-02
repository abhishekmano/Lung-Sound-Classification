import json
import os

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

app = Flask(__name__) #reference this file

app.config['UPLOAD_PATH'] = 'static/uploads'

@app.route('/')

def index ():
    return render_template('index.html')

@app.route('/', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      filename = secure_filename(f.filename)
      
      # to store the filename to json file (mydata.json)
      name = {}
      name = {'file_name' : filename}

      with open('mydata.json','w') as j:
          json.dump(name, j) 

      # save the file to directory  
      f.save(os.path.join(app.config['UPLOAD_PATH'], filename))

      return render_template('uploaded.html', filename=filename) 

@app.route('/', methods = ["back"])
def go_back():
    return render_template('index.html')

if __name__ == "__main__" :
    app.run(debug=True)

