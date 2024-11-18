from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory, session
import os
import cv2
from PIL import Image
import numpy as np
import tensorflow as tf
import random
import shutil
import time

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from functools import wraps
from datetime import timedelta
import logging

# Configure logging to exclude PIL debug messages
logging.basicConfig(level=logging.INFO)
logging.getLogger('PIL').setLevel(logging.INFO)

app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=30)

# Disable auto-reload
app.config['DEBUG'] = False
app.config['USE_RELOADER'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USERS_DIR = os.path.join(BASE_DIR, "users")
os.makedirs(USERS_DIR, exist_ok=True)

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
@login_required
def index():
    matrix_exists = os.path.exists(os.path.join('static', 'confusion_matrix.png'))
    return render_template('index.html', 
                         prediction=request.args.get('prediction', ''),
                         username=session.get('username'),
                         show_matrix=matrix_exists,
                         matrix_path='confusion_matrix.png' if matrix_exists else None)

@app.route('/generate_confusion_matrix', methods=['POST'])
@login_required
def generate_confusion_matrix():
    try:
        plt.switch_backend('Agg')
        
        # Get predictions and calculate accuracy
        y_pred = np.argmax(model.predict(val_generator), axis=1)
        y_true = val_generator.classes
        accuracy = np.mean(y_pred == y_true) * 100
        cm = confusion_matrix(y_true, y_pred)
        
        # Generate square confusion matrix plot
        plt.figure(figsize=(20, 20))  # Square figure size
        cmd = ConfusionMatrixDisplay(cm, display_labels=val_generator.class_indices.keys())
        cmd.plot(values_format='d', xticks_rotation=45)  # Added value format and rotated labels
        plt.tight_layout(pad=2.0)  # Adjust padding
        plt.title(f"Confusion Matrix\nValidation Accuracy: {accuracy:.2f}%", pad=20, fontsize=14)
        
        # Save the plot with higher DPI
        static_dir = os.path.join(os.getcwd(), 'static')
        os.makedirs(static_dir, exist_ok=True)
        matrix_path = 'confusion_matrix.png'
        plt.savefig(os.path.join(static_dir, matrix_path), dpi=300, bbox_inches='tight')
        plt.close()
        
        return render_template('index.html', 
                             username=session.get('username'),
                             show_matrix=True,
                             matrix_path=matrix_path,
                             model_accuracy=f"{accuracy:.2f}")
    except Exception as e:
        flash(f'Error generating confusion matrix: {str(e)}', 'warning')
        return redirect(url_for('index'))

def create_user_dataset_structure(username):
    """Create dataset folders for user"""
    emotions = ['happy', 'sad', 'angry', 'disgust', 'fear', 'surprise', 'neutral', 'contempt']
    user_dataset = os.path.join(USERS_DIR, username, 'dataset')
    
    for emotion in emotions:
        emotion_dir = os.path.join(user_dataset, emotion)
        os.makedirs(emotion_dir, exist_ok=True)

@app.route('/classify_image', methods=['POST'])
def classify_image():
    try:
        data = request.get_json()
        filename = data.get('filename')
        emotion = data.get('emotion')
        username = session.get('username')

        # Add debug logging
        print(f"Received request: filename={filename}, emotion={emotion}, username={username}")

        if not all([filename, emotion, username]):
            return jsonify({'success': False, 'error': 'Missing parameters'})

        # Source path from photos folder
        source_path = os.path.join(USERS_DIR, username, 'photos', filename)
        print(f"Source path: {source_path}")
        
        if not os.path.exists(source_path):
            return jsonify({'success': False, 'error': f'Source file not found: {source_path}'})

        # Target path in dataset folder
        target_dir = os.path.join(USERS_DIR, username, 'dataset', emotion.lower())
        os.makedirs(target_dir, exist_ok=True)
        print(f"Target directory: {target_dir}")

        # Create unique filename
        base, ext = os.path.splitext(filename)
        target_filename = f"{base}_{int(time.time())}{ext}"
        target_path = os.path.join(target_dir, target_filename)
        print(f"Target path: {target_path}")

        # Move file with explicit error handling
        try:
            shutil.move(source_path, target_path)
            print(f"File moved successfully from {source_path} to {target_path}")
        except Exception as move_error:
            print(f"Error moving file: {str(move_error)}")
            return jsonify({'success': False, 'error': f'Failed to move file: {str(move_error)}'})

        return jsonify({
            'success': True,
            'message': f'Image moved to {emotion} folder',
            'new_path': target_path
        })

    except Exception as e:
        print(f"Error in classify_image: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/register', methods=['GET', 'POST'])
def register():
    app.logger.debug(f"Register route called with method: {request.method}")
    if request.method == 'GET':
        app.logger.debug("Rendering register template")
        return render_template('register.html')
    
    # Handle POST request
    username = request.form.get('username')
    password = request.form.get('password')

    if not username or not password:
        return jsonify({"error": "Username and password are required"}), 400

    if not username.isalnum():
        return jsonify({"error": "Username can only contain letters and numbers"}), 400

    # Create user directory
    user_dir = os.path.join(USERS_DIR, username)
    if os.path.exists(user_dir):
        return jsonify({"error": "User already exists"}), 400

    try:
        os.makedirs(user_dir)
        user_file = os.path.join(user_dir, "user.txt")
        
        # Create dataset structure
        create_user_dataset_structure(username)
        
        # Save user data
        with open(user_file, 'w') as f:
            f.write(f"Username:{username}\nPassword:{password}")

        return jsonify({"message": "Registration successful"}), 200
    except Exception as e:
        return jsonify({"error": f"Registration failed: {str(e)}"}), 500

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')
    
    username = request.form.get('username')
    password = request.form.get('password')

    if not username or not password:
        return jsonify({"error": "Username and password are required"}), 400

    user_dir = os.path.join(USERS_DIR, username)
    user_file = os.path.join(user_dir, "user.txt")

    if not os.path.exists(user_file):
        return jsonify({"error": "User does not exist"}), 404

    # Odczyt danych użytkownika
    with open(user_file, 'r') as f:
        stored_password = f.readlines()[1].split(":")[1].strip()

    if stored_password != password:
        return jsonify({"error": "Invalid password"}), 400

    session['username'] = username
    return jsonify({"message": "Login successful"}), 200

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if 'text' in request.form:
        text = request.form['text']
        # Tutaj dodaj logikę przewidywania dla tekstu
        predictions = [{"emotion": "happy", "probability": 0.8}]  # Przykład
        return jsonify({"predictions": predictions})
    return jsonify({"error": "No text provided"}), 400

@app.route('/upload_and_predict', methods=['POST'])
@login_required
def upload_and_predict():
    if 'file' not in request.files:
        flash('No file part', 'danger')
        return redirect(url_for('index'))
    
    files = request.files.getlist('file')  # Changed to getlist
    uploaded_files = []
    predictions_list = []
    
    user_folder = os.path.join(USERS_DIR, session['username'], 'photos')
    os.makedirs(user_folder, exist_ok=True)
    
    for file in files:
        if file.filename != '':
            file_path = os.path.join(user_folder, file.filename)
            file.save(file_path)
            uploaded_files.append(file.filename)
            
            # Get predictions for this image
            img_array = preprocess_image(file_path)
            predictions = model.predict(img_array)[0]
            
            # Sort emotions by probability
            emotions = list(train_generator.class_indices.keys())
            emotion_predictions = [
                {"emotion": emotion, "probability": float(prob)}
                for emotion, prob in zip(emotions, predictions)
            ]
            # Sort by probability in descending order
            emotion_predictions.sort(key=lambda x: x["probability"], reverse=True)
            predictions_list.append(emotion_predictions)
    
    if uploaded_files:
        return render_template('photos.html',
                             username=session.get('username'),
                             photos=uploaded_files,
                             predictions=predictions_list)
    
    flash('No files were uploaded', 'warning')
    return redirect(url_for('index'))

@app.route('/upload_multiple', methods=['POST'])
@login_required
def upload_multiple_images():
    if 'files' not in request.files:
        flash('No files part', 'danger')
        return redirect(url_for('index'))
    
    files = request.files.getlist('files')
    
    if not files:
        flash('No files selected', 'danger')
        return redirect(url_for('index'))
    
    user_folder = os.path.join(USERS_DIR, session['username'], 'photos')
    os.makedirs(user_folder, exist_ok=True)
    
    for file in files:
        if file.filename == '':
            continue
        
        file_path = os.path.join(user_folder, file.filename)
        file.save(file_path)
    
    flash(f'{len(files)} images uploaded successfully!', 'success')
    return redirect(url_for('index'))

@app.route('/photos')
@login_required
def photos():
    return render_template('photos.html', username=session.get('username'))

@app.route('/users/<username>/photos/<filename>')
@login_required
def user_photo(username, filename):
    if username != session.get('username'):
        return redirect(url_for('login'))
    return send_from_directory(os.path.join(USERS_DIR, username, 'photos'), filename)

# Update paths to be relative to the application directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "emotions", "trening")
VAL_DIR = os.path.join(BASE_DIR, "emotions", "walidacja")
TEST_DIR = os.path.join(BASE_DIR, "emotions", "test")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "emotion_detection_model.h5")

# Replace the existing path definitions with these new variables
train_dir = TRAIN_DIR
val_dir = VAL_DIR
test_dir = TEST_DIR
model_path = MODEL_PATH

def ensure_directories():
    """Ensure all required directories exist"""
    directories = [
        os.path.join(BASE_DIR, 'static'),
        os.path.join(BASE_DIR, 'templates'),
        os.path.join(BASE_DIR, 'uploads'),
        os.path.join(BASE_DIR, 'users'),
        MODEL_DIR
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

# Hyperparameters
batch_size = 64
img_height, img_width = 96, 96
epochs = 100
learning_rate = 0.0005

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Define the CNN model
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(train_generator.num_classes, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def load_or_create_model():
    try:
        if os.path.exists(MODEL_PATH):
            print(f"Loading model from: {MODEL_PATH}")
            model = tf.keras.models.load_model(MODEL_PATH)
            model.compile(
                optimizer=Adam(learning_rate=learning_rate),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            return model
        else:
            print("No pre-trained model found. Creating a new one.")
            return create_model()
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Creating a new model instead.")
        return create_model()

# Initialize the model
model = load_or_create_model()

def preprocess_image(file_path):
    img = image.load_img(file_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_emotion(model, img_array):
    prediction = model.predict(img_array)[0]
    top_indices = prediction.argsort()[-2:][::-1]
    top_emotions = [(list(train_generator.class_indices.keys())[i], prediction[i]) for i in top_indices]
    return top_emotions

@app.route('/train', methods=['POST'])
@login_required
def train_model():
    if not os.path.exists(TRAIN_DIR) or not os.path.exists(VAL_DIR):
        flash('Training and validation directories not found!', 'danger')
        return redirect(url_for('index'))

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

    try:
        history = model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=[early_stopping, reduce_lr]
        )

        # Ensure model directory exists
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        # Save the model
        model.save(MODEL_PATH)
        flash('Model trained and saved successfully!', 'success')
    except Exception as e:
        flash(f'Error during training: {str(e)}', 'danger')
    
    return redirect(url_for('index'))

@app.route('/classify_all_images', methods=['POST'])
def classify_all_images():
    try:
        data = request.get_json()
        classifications = data.get('classifications', [])
        username = session.get('username')

        if not username or not classifications:
            return jsonify({'success': False, 'error': 'Invalid request'})

        results = []
        for item in classifications:
            filename = item.get('filename')
            emotion = item.get('emotion')
            
            source_path = os.path.join(USERS_DIR, username, 'photos', filename)
            target_dir = os.path.join(USERS_DIR, username, 'dataset', emotion.lower())
            os.makedirs(target_dir, exist_ok=True)

            base, ext = os.path.splitext(filename)
            target_filename = f"{base}_{int(time.time())}{ext}"
            target_path = os.path.join(target_dir, target_filename)

            try:
                shutil.move(source_path, target_path)
                results.append({
                    'filename': filename,
                    'success': True
                })
            except Exception as e:
                results.append({
                    'filename': filename,
                    'success': False,
                    'error': str(e)
                })

        return jsonify({
            'success': True,
            'results': results
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == '__main__':
    # Ensure all required directories exist
    ensure_directories()
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)