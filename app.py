from flask import Flask, render_template, request, redirect, url_for, flash
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

app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'uploads'

# Define paths
train_dir = r"C:\Users\klesz\Desktop\emotion_detection-main\emotion_detection-main\emotions\trening"
val_dir = r"C:\Users\klesz\Desktop\emotion_detection-main\emotion_detection-main\emotions\walidacja"
test_dir = r"C:\Users\klesz\Desktop\emotion_detection-main\emotion_detection-main\emotions"
model_path = r"C:\Users\klesz\Desktop\emotion_detection-main\emotion_detection-main\emotion_detection_model_face.h5"

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

@app.route('/')
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

    return render_template('index.html', prediction=prediction)

@app.route('/train', methods=['POST'])
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

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)