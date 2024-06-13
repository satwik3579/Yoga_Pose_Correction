from flask import Flask, render_template, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib

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


# Load the model
model = joblib.load('model\pose_to_angles_predictor.pkl')



mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

angle_columns = ['Left Elbow', 'Right Elbow', 'Left Shoulder', 'Right Shoulder', 'Left Hip', 'Right Hip', 'Left Knee', 'Right Knee']


def predict_angles(pose_name):
    pose_label = pose_labels[pose_name]
    predicted_angles = model.predict([[pose_label]])[0]
    angles = dict(zip(angle_columns, predicted_angles))
    return angles



# Function to read reference angles from an Excel file
def read_reference_angles(pose_name):
    reference_angles = predict_angles(pose_name)
    reference_angles = {key.lower().replace(' ', '_'): value for key, value in reference_angles.items() if key != 'pose'}
    print(reference_angles)
    return reference_angles

# Example file path
file_path = 'model\output_angles.xlsx'

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def capture_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        return None
    cap.release()
    return frame

def process_frame(frame, reference_angles):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        angles = {
            'left_elbow': calculate_angle(
                [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y],
                [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y],
                [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]),
            'right_elbow': calculate_angle(
                [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y],
                [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y],
                [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]),
            'left_shoulder': calculate_angle(
                [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y],
                [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y],
                [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]),
            'right_shoulder': calculate_angle(
                [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y],
                [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y],
                [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]),
            'left_hip': calculate_angle(
                [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y],
                [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y],
                [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]),
            'right_hip': calculate_angle(
                [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y],
                [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y],
                [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]),
            'left_knee': calculate_angle(
                [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y],
                [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y],
                [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]),
            'right_knee': calculate_angle(
                [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y],
                [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y],
                [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]),
        }
        flag = True
        wrong_joints = []
        for part, angle in angles.items():
            if not compare_angles(angle, part, reference_angles):
                flag = False
                wrong_joints.append(part)
        if flag:
            return "Pose is correct"
        else:
            return f"Incorrect pose, adjust the following joints: {', '.join(wrong_joints)}"
    else:
        return "No pose detected"

def compare_angles(calculated_angle, body_part, reference_angles):
    reference_angle = reference_angles[body_part]
    if abs(calculated_angle - reference_angle) > 15:
        return False
    return True




app = Flask(__name__)






# Route for the main page
@app.route('/')
def index():
    return render_template('index1.html')

# Routes for each button
@app.route('/page1')
def page1():
    return render_template('page1.html')

@app.route('/page2')
def page2():
    return render_template('page2.html')

@app.route('/page3')
def page3():
    return render_template('page3.html')

@app.route('/page4')
def page4():
    return render_template('page4.html')

@app.route('/page5')
def page5():
    return render_template('page5.html')

@app.route('/page6')
def page6():
    return render_template('page6.html')

@app.route('/page7')
def page7():
    return render_template('page7.html')

@app.route('/page8')
def page8():
    return render_template('page8.html')


@app.route('/analyze_pose', methods=['POST'])
def analyze_pose():
    pose_name = request.json.get('pose_name')
    print(pose_name)
    reference_angles = read_reference_angles(pose_name)
    frame = capture_image()
    if frame is not None:
        result = process_frame(frame, reference_angles)
        return jsonify({'result': result})
    else:
        return jsonify({'result': 'Error capturing image'}), 500




if __name__ == '__main__':
    app.run(debug=True)
