import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template

from flask import render_template

filename = 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

app = Flask(__name__, template_folder='templates')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])

def predict():
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = loaded_model.predict(final_features)
    output = prediction[0]
    return render_template('index.html', prediction_text='The player will be in the NBA in 5 years : {}'.format(output))

if __name__ == "__main__":
    app.run(port = 5000 ,debug=True)