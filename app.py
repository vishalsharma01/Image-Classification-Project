import flask
import pickle
from flask import render_template
from flask import request
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import requests
from io import StringIO
from io import BytesIO


app = flask.Flask(__name__)

classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('model.h5')

@app.route('/')
def page():
    return render_template('page.html')


@app.route('/result_image', methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        result = flask.request.form
    url = result['url']
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img = img.resize((200,200))
    img = np.array(img, dtype = "float32")
    img /= 255

    pred = model.predict(np.reshape(img,(1,200,200,3)))
    value = np.max(pred)
    value = round(value * 100, 2)
    pred = np.argmax(pred)
    pred = classes[pred]

    return render_template('result.html', pred = pred, image_url = url, value = value)




if __name__ == '__main__':
    '''Connects to the server'''

    HOST = '127.0.0.1'
    PORT = 4100

    app.run(HOST, PORT, debug=True)
