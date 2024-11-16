import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk
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

# Define paths
train_dir = "C:/tensorflow_env/emotions/trening"
val_dir = "C:/tensorflow_env/emotions/walidacja"
test_dir = "C:/tensorflow_env/emotions/test"
model_path = "emotion_detection_model.h5"

# Hyperparameters
batch_size = 64
img_height, img_width = 96, 96
epochs = 100
learning_rate = 0.0005

# Initialize main window
root = tk.Tk()
root.title("Emotion Detection")
root.geometry("400x400")
root.configure(bg="#f2f2f2")

# Define header frame
header_frame = tk.Frame(root, bg="#4a4a4a", height=80)
header_frame.pack(fill="x")

header_label = tk.Label(header_frame, text="Emotion Detection", font=("Helvetica", 18), bg="#4a4a4a", fg="white")
header_label.pack(pady=20)

# Define main content frame
content_frame = tk.Frame(root, bg="#f2f2f2")
content_frame.pack(expand=True, fill="both")

# Style for buttons
style = ttk.Style()
style.configure("TButton", font=("Helvetica", 12), padding=10)

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

# Load or initialize the model
def load_or_create_model():
    if os.path.exists(model_path):
        print("Loading model from disk.")
        model = tf.keras.models.load_model(model_path)
        # Recompile the model to ensure the optimizer is set up correctly
        model.compile(optimizer=Adam(learning_rate=learning_rate),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model
    else:
        print("No pre-trained model found. Creating a new one.")
        return create_model()


# Initialize the model
model = load_or_create_model()

# Function definitions
def preprocess_image(file_path):
    img = image.load_img(file_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_emotion(model, img_array):
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction, axis=1)[0]
    return list(train_generator.class_indices.keys())[class_idx]

def upload_and_predict():
    file_path = filedialog.askopenfilename()
    if file_path:
        img_array = preprocess_image(file_path)
        emotion = predict_emotion(model, img_array)
        messagebox.showinfo("Prediction", f"Predicted emotion: {emotion}")

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
    print(f"Model saved to {model_path}")

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()

def show_confusion_matrix():
    y_pred = np.argmax(model.predict(val_generator), axis=1)
    y_true = val_generator.classes
    cm = confusion_matrix(y_true, y_pred)
    cmd = ConfusionMatrixDisplay(cm, display_labels=val_generator.class_indices.keys())
    cmd.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

def start_webcam():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, (img_height, img_width)) / 255.0
            face = np.expand_dims(face, axis=0)
            emotion = predict_emotion(model, face)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, f'Emotion: {emotion}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Webcam - Press Q to exit', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# Create buttons
ttk.Button(content_frame, text="Train Model", command=train_model).pack(pady=10)
ttk.Button(content_frame, text="Upload Image", command=upload_and_predict).pack(pady=10)
ttk.Button(content_frame, text="Start Webcam", command=start_webcam).pack(pady=10)
ttk.Button(content_frame, text="Show Test Results", command=show_confusion_matrix).pack(pady=10)

root.mainloop()
