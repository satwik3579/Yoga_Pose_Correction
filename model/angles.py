import joblib

# Define the pose labels
pose_labels = {
    'adhomukhasvanasana': 0,
    'ashtanga namaskara': 1,
    'ashtanga namaskara': 2,
    'bhujangasana': 3,
    'chaturanga dandasana': 4,
    'hastauttanasana': 5,
    'padangusthasana': 6,
    'pranamsana': 7
}

# Define the angle columns
angle_columns = ['Left Elbow', 'Right Elbow', 'Left Shoulder', 'Right Shoulder', 'Left Hip', 'Right Hip', 'Left Knee', 'Right Knee']

# Load the model
model = joblib.load('pose_to_angles_predictor.pkl')

# Function to predict angles for a given pose
def predict_angles(pose_name):
    pose_label = pose_labels[pose_name]
    predicted_angles = model.predict([[pose_label]])[0]
    angles = dict(zip(angle_columns, predicted_angles))
    return angles

# Example usage
selected_pose_name = 'adhomukhasvanasana'  # Replace with the pose name you want to predict angles for
predicted_angles = predict_angles(selected_pose_name)
print('Predicted Angles for Pose', selected_pose_name, ':', predicted_angles)
