import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import numpy as np

# Load the data
data = pd.read_excel('angles.xlsx')

# Define the pose labels
pose_labels = {
    'adhomukhasvanasana': 0,
    'ashtanga namaskara': 1,
    'ashwasanchalanasana': 2,
    'bhujangasana': 3,
    'chaturanga dandasana': 4,
    'hastauttanasana': 5,
    'padangusthasana': 6,
    'pranamsana': 7
}

# Map the pose names to their corresponding labels
data['Pose'] = data['Pose'].map(pose_labels)

# Define the angle columns
angle_columns = ['Left Elbow', 'Right Elbow', 'Left Shoulder', 'Right Shoulder', 'Left Hip', 'Right Hip', 'Left Knee', 'Right Knee']

# Features and labels
X = data['Pose'].values.reshape(-1, 1)
y = data[angle_columns]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model for all angles simultaneously
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
r2_score = model.score(X_test, y_test)
print('Overall R^2 score:', r2_score)

# Save the model
joblib.dump(model, 'pose_to_angles_predictor.pkl')
