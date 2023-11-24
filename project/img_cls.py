'''
This example uses a linear Support Vector Machine (SVM) as the classifier, 
but you can experiment with other classifiers or neural network architectures depending on your specific requirements.
'''
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load pre-trained MobileNetV2 model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Create a new model with the base model's convolutional layers
model = tf.keras.models.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
])

# Load and preprocess your dataset
def load_and_preprocess_images(directory, label):
    images = []
    labels = []
    for filename in os.listdir(directory):
        img_path = os.path.join(directory, filename)
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = preprocess_input(img_array)
        images.append(img_array)
        labels.append(label)
    return np.array(images), np.array(labels)

# Replace 'path/to/positive' and 'path/to/negative' with your dataset paths
positive_images, positive_labels = load_and_preprocess_images('path/to/positive', label=1)
negative_images, negative_labels = load_and_preprocess_images('path/to/negative', label=0)

# Concatenate positive and negative samples
all_images = np.concatenate([positive_images, negative_images])
all_labels = np.concatenate([positive_labels, negative_labels])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(all_images, all_labels, test_size=0.2, random_state=42)

# Extract features using the model
X_train_features = model.predict(X_train)
X_test_features = model.predict(X_test)

# Train a Support Vector Machine (SVM) classifier
classifier = SVC(kernel='linear')
classifier.fit(X_train_features, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test_features)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
