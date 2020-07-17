import numpy as np
from flask import Flask, render_template, url_for, request, jsonify
import pickle

a = Flask(__name__, static_url_path='/static')


cv = pickle.load(open('transform.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))


@a.route('/')
@a.route('/index')
def index():
    return render_template('index.html')

@a.route('/index',methods=['POST'])
def predict():

    if request.method == 'POST':
        msg = request.form['data']
        data = [msg]
        vect = cv.transform(data).toarray()
        prediction = model.predict(vect)
    return render_template('index.html', predictionn = prediction[0])

if __name__ == "__main__":
    a.run(debug=True)