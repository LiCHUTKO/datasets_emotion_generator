from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory, session
import os
import cv2
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from functools import wraps
from datetime import timedelta
import logging

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=30)

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
    # Generate confusion matrix
    y_pred = np.argmax(model.predict(val_generator), axis=1)
    y_true = val_generator.classes
    cm = confusion_matrix(y_true, y_pred)
    cmd = ConfusionMatrixDisplay(cm, display_labels=val_generator.class_indices.keys())
    cmd.plot()
    plt.title("Confusion Matrix")

    # Ensure the static directory exists
    static_dir = os.path.join(os.getcwd(), 'static')
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)

    plt.savefig(os.path.join(static_dir, 'confusion_matrix.png'))
    plt.close()  # Close the plot to free memory

    prediction = request.args.get('prediction', '')

    return render_template('index.html', prediction=prediction, username=session.get('username'))

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

# Define paths
train_dir = r"emotions\trening"
val_dir = r"emotions\walidacja"
test_dir = r"emotion_detection-main\emotions"
model_path = r"emotion_detection_model_face.h5"

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
        if os.path.exists(model_path):
            print("Loading model from disk.")
            model = tf.keras.models.load_model(model_path)
            model.compile(optimizer=Adam(learning_rate=learning_rate),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
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
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=[early_stopping, reduce_lr]
    )

    model.save(model_path)
    flash('Model trained and saved successfully!', 'success')
    return redirect(url_for('index'))

@app.route('/upload', methods=['POST'])
@login_required
def upload_and_predict():
    if 'file' not in request.files:
        flash('No file part', 'danger')
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        flash('No selected file', 'danger')
        return redirect(url_for('index'))
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        img_array = preprocess_image(file_path)
        top_emotions = predict_emotion(model, img_array)
        emotion_message = f"Predicted emotions: {top_emotions[0][0]} ({top_emotions[0][1]*100:.2f}%), {top_emotions[1][0]} ({top_emotions[1][1]*100:.2f}%)"
        flash(emotion_message, 'success')
        return redirect(url_for('index', prediction=emotion_message))
    flash('File upload failed', 'danger')
    return redirect(url_for('index'))

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == '__main__':
    # Create required directories
    for directory in ['static', 'templates', 'uploads', 'users']:
        os.makedirs(directory, exist_ok=True)
    
    # Enable debug mode
    app.debug = True
    app.run(host='0.0.0.0', port=5000)