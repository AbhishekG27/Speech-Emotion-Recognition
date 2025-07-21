import os
import librosa
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# Define paths
DATA_PATH = '/Users/rahul.v/Documents/VS_Code/mini_project/audio'

# Define emotions
emotion_labels = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

# Function to load data and extract features
def load_data(data_path):
    features = []
    for subdir, _, files in os.walk(data_path):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(subdir, file)
                try:
                    emotion = emotion_labels[file.split('-')[2]]
                    signal, sample_rate = librosa.load(file_path, sr=22050)
                    mfccs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=13)
                    mfccs_mean = np.mean(mfccs.T, axis=0)
                    features.append([mfccs_mean, emotion])
                except Exception as e:
                    print(f"Error processing {file}: {str(e)}")
    return features

# Load and process data
print("Loading data...")
data = load_data(DATA_PATH)

# Convert to DataFrame
df = pd.DataFrame(data, columns=['feature', 'emotion'])

# Encode labels
X = np.array(df['feature'].tolist())
y = np.array(df['emotion'].tolist())
le = LabelEncoder()
y = le.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the model
model = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=300, random_state=42)

# Train the model
print("Training the model...")
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Function to predict emotion
def predict_emotion(file_path):
    signal, sample_rate = librosa.load(file_path, sr=22050)
    mfccs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    mfccs_scaled = scaler.transform([mfccs_mean])
    emotion_pred = model.predict(mfccs_scaled)
    return le.inverse_transform(emotion_pred)[0]


# After training your model
joblib.dump(model, 'emotion_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(le, 'label_encoder.joblib')


# Example usage (uncomment and modify path as needed)
# print(predict_emotion('path_to_test_audio.wav'))