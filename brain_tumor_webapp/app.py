from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model("model/Brain_tumor_simple_1.keras")
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_url = None
    confidence_score = None
    raw_output = None
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Preprocess image
            img = image.load_img(filepath, target_size=(64, 64))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Make prediction
            result = model.predict(img_array)
            score = float(result[0][0])
            confidence_score = round(score * 100, 2) if score > 0.5 else round((1 - score) * 100, 2)
            prediction = "Tumor detected!" if score > 0.5 else "No Tumor detected."
            image_url = filepath
            raw_output = f"{score:.4f}"

    return render_template(
        'index.html',
        prediction=prediction,
        image_url=image_url,
        confidence_score=confidence_score,
        raw_output=raw_output
    )

if __name__ == '__main__':
    app.run(debug=True)
