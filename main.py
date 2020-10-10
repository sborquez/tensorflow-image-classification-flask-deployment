from flask import Flask, render_template, request, jsonify
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import cv2
import numpy as np

from flask_cors import CORS, cross_origin



def get_model(model_name, get_labels=False):
    models = {
        "flower": {
            "path": "flower.model",
            "label_names": ["daisy", "dandelon", "roses", "sunflowers", "tulips"]
        }
    }
    model_path = models[model_name]["path"]
    model = load_model(model_path)
    if get_labels:
        model_labels = models[model_name]["label_names"]
        return model, model_labels
    return model


# Process image and predict label
def predict(model_name, image_path):
    # Read image
    # Preprocess image
    print(image_path)
    image = cv2.imread(image_path)
    image = cv2.resize(image, (199, 199))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    model, model_labels = get_model(model_name, get_labels=True)
    res = model.predict(image)
    label = np.argmax(res)
    print("Label", label)
    label_name = model_labels[label]
    print("Label name:", label_name)
    prediction = {
        "label": int(label),
        "label_name": label_name
    }
    print("prediction:", prediction)
    return prediction


# Initializing flask application
SECRET_KEY = 'you-will-never-guess'
app = Flask(__name__)
app.config['SECRET_KEY'] = SECRET_KEY
cors = CORS(app)

@app.route("/ok")
def ok():
    return """
        Application is working
    """


@app.route("/")
def index():
    return render_template("index.html")

# About page with render template
@app.route("/about")
def about():
    return render_template("about.html")

# Process images
class UploadForm(FlaskForm):
    photo = FileField()#validators=[FileAllowed(photos, 'Image only!'), FileRequired('File was empty!')])
    submit = SubmitField('Upload')

@app.route("/predict", methods=["POST", "GET"])
def process_predict():
    error = ""
    form = form = UploadForm()

    if request.method == 'POST' and 'photo' in request.files:
        image = request.files["photo"]
        model_name = request.files.get("model", 'flower')
        image.save("img.jpg")
        print(model_name)
        resp = predict(model_name, "img.jpg")
        resp = jsonify(resp)
        resp.status_code = 200
        print(f"resp:", resp)
        return resp
    else:
        return render_template('predict.html', form=form, message=error)


if __name__ == "__main__":
    app.run(debug=True)
