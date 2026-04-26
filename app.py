from flask import Flask, render_template, request
import os
from flask import Flask, request, render_template, url_for
from werkzeug.utils import secure_filename
import os
import torch
from other_operations import model_predict, generate_synthetic_images

app = Flask(__name__)

# Define the path for image uploads
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_images', methods=['GET', 'POST'])
def generate_images():
    if request.method == 'POST':
        input_file = request.files.get('input_image')
        num_images = int(request.form.get('num_images', 1))

        if not input_file:
            return render_template('generate_images.html', error="No file uploaded")

        filename = secure_filename(input_file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        input_file.save(save_path)

        generated = generate_synthetic_images(num_images)

        return render_template(
            'generate_images.html',
            uploaded_image=f"uploads/{filename}",
            generated_images=generated
        )

    return render_template('generate_images.html')

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        file = request.files['image']
        if not file:
            return render_template('prediction.html', error="Please upload an image.")

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        predicted_label,confidence=model_predict(filepath)

        return render_template('prediction.html',
                               predicted_class=predicted_label,
                               confidence_score=round(confidence, 2),
                               image_path=f"uploads/{filename}")

    return render_template('prediction.html')

@app.route('/results')
def results():
    return render_template('results.html')

if __name__ == "__main__":
    app.run(debug=True)
