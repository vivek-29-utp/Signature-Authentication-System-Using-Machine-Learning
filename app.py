from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import cv2
from utils.preprocess import preprocess_image
from model.siamese_network import get_siamese_model
from model.feature_extractor import get_embedding
import numpy as np
from PIL import Image
import tensorflow as tf

import os
import logging
from flask import send_from_directory

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

static_path = os.path.abspath('static').replace('\\', '/')
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(static_path, 'uploads').replace('\\', '/')

# Explicit static routes
@app.route('/img/<path:filename>')
def serve_img(filename):
    return send_from_directory(os.path.join(static_path, ''), filename)

@app.route('/css/<path:filename>')
def serve_css(filename):
    return send_from_directory(os.path.join(static_path, ''), filename)

@app.route('/js/<path:filename>')
def serve_js(filename):
    return send_from_directory(os.path.join(static_path, ''), filename)

@app.before_request
def log_static_requests():
    if request.path.startswith('/assets/') or request.path.startswith('/static/'):
        print(f"File request: {request.path}")
        full_path = os.path.join(static_path, request.path[1:]).replace('\\', '/')
        print(f"Resolved path: {full_path}")
        print(f"File exists: {os.path.exists(full_path)}")

# Load model with memory optimization
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

try:
    model = get_siamese_model()
    model.load_weights('model/siamese_model.h5')
except Exception as e:
    print(f"Error loading model: {e}")
    # Try with smaller batch size
    tf.keras.backend.clear_session()
    model = get_siamese_model()
    model.load_weights('model/siamese_model.h5')

def verify_signature(uploaded_path, reference_path):
    print(f"Verifying signature: {uploaded_path} vs {reference_path}")
    img1 = preprocess_image(uploaded_path)
    img2 = preprocess_image(reference_path)
    emb1 = get_embedding(img1)
    emb2 = get_embedding(img2)
    distance = np.linalg.norm(emb1 - emb2)
    result = "Genuine" if distance < 0.5 else "Forged"
    print(f"Verification result: {result} (distance: {distance:.4f})")
    return result

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    uploaded_img = None
    reference_img = None

    print(f"\nReceived {request.method} request")
    
    if request.method == "POST":
        print("Processing signature verification request")
        person_id = request.form["person_id"]
        file = request.files["signature"]
        print(f"Person ID: {person_id}, File: {file.filename}")

        if file:
            # Validate file extension
            allowed_extensions = {'png', 'jpg', 'jpeg'}
            if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
                result = "Unsupported file format. Please upload PNG or JPG images."
            else:
                filename = f"{person_id}_uploaded.png"
                upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                
                # Ensure upload directory exists
                # Ensure upload directory exists
                os.makedirs(os.path.dirname(upload_path), exist_ok=True)
                print(f"Upload path: {upload_path}")
                
                try:
                    print(f"Attempting to save file to: {upload_path}")
                    file.save(upload_path)
                    print("File save completed")
                    print(f"Saved file size: {os.path.getsize(upload_path)} bytes")
                    
                    # Verify file was saved and is not empty
                    if not os.path.exists(upload_path):
                        result = f"File not found at {upload_path} after save attempt"
                    elif os.path.getsize(upload_path) == 0:
                        result = "File saved but is empty (0 bytes)"
                        print(f"Saved file size: {os.path.getsize(upload_path)} bytes")
                    else:
                        # Log file details
                        print(f"Uploaded file path: {upload_path}")
                        print(f"Uploaded file size: {os.path.getsize(upload_path)} bytes")
                        
                        # Verify image can be read
                        test_img = cv2.imread(upload_path)
                        if test_img is None:
                            result = "Uploaded file is not a valid image"
                        else:
                            reference_path = f"static/reference_signatures/{person_id}_real.png"
                            if not os.path.exists(reference_path):
                                result = "Reference signature not found."
                            else:
                                result = verify_signature(upload_path, reference_path)
                                uploaded_img = url_for('static', filename=f"uploads/{filename}")
                                reference_img = url_for('static', filename=f"reference_signatures/{person_id}_real.png")
                except Exception as e:
                    result = f"Error processing file: {str(e)}"

    return render_template("index.html", result=result, uploaded_img=uploaded_img, reference_img=reference_img)

@app.before_request
def log_request_info():
    print("Static files path:", app.static_folder)

if __name__ == "__main__":
    app.run(debug=True)