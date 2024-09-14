import os
import numpy as np
import cv2
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

def load_data_from_folder(folder_path, label):
    data = []
    labels = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (128, 128))
            img = img.flatten()
            data.append(img)
            labels.append(label)
    return data, labels

def load_dataset():
    train_normal_data, train_normal_labels = load_data_from_folder('E:\\pneumonia_detection\\dataset\\chest_xray\\train\\NORMAL', 0)
    train_pneumonia_data, train_pneumonia_labels = load_data_from_folder('E:\\pneumonia_detection\\dataset\\chest_xray\\train\\PNEUMONIA', 1)
    
    val_normal_data, val_normal_labels = load_data_from_folder('E:\\pneumonia_detection\\dataset\\chest_xray\\val\\NORMAL', 0)
    val_pneumonia_data, val_pneumonia_labels = load_data_from_folder('E:\\pneumonia_detection\\dataset\\chest_xray\\val\\PNEUMONIA', 1)

    test_normal_data, test_normal_labels = load_data_from_folder('E:\\pneumonia_detection\\dataset\\chest_xray\\test\\NORMAL', 0)
    test_pneumonia_data, test_pneumonia_labels = load_data_from_folder('E:\\pneumonia_detection\\dataset\\chest_xray\\test\\PNEUMONIA', 1)

    # Combine training and validation data
    X_train = train_normal_data + train_pneumonia_data + val_normal_data + val_pneumonia_data
    y_train = train_normal_labels + train_pneumonia_labels + val_normal_labels + val_pneumonia_labels

    X_test = test_normal_data + test_pneumonia_data
    y_test = test_normal_labels + test_pneumonia_labels

    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

# Load dataset
X_train, y_train, X_test, y_test = load_dataset()

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model and scaler
joblib.dump(model, 'model/model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')

# Evaluate model on test data
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
