import os
import cv2 as opencv
import mediapipe as mp
import numpy as np
import openpyxl

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# Define a function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point
    c = np.array(c)  # End point
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# Extract angles for key points
def extract_angles(landmarks):
    # Define key points
    key_points = {
        'left_shoulder': [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y],
        'left_elbow': [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y],
        'left_wrist': [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y],
        'right_shoulder': [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y],
        'right_elbow': [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y],
        'right_wrist': [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y],
        'left_hip': [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y],
        'left_knee': [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y],
        'left_ankle': [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y],
        'right_hip': [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y],
        'right_knee': [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y],
        'right_ankle': [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y],
    }
    
    # Calculate angles
    angles = {
        'left_elbow': calculate_angle(key_points['left_shoulder'], key_points['left_elbow'], key_points['left_wrist']),
        'right_elbow': calculate_angle(key_points['right_shoulder'], key_points['right_elbow'], key_points['right_wrist']),
        'left_shoulder': calculate_angle(key_points['left_hip'], key_points['left_shoulder'], key_points['left_elbow']),
        'right_shoulder': calculate_angle(key_points['right_hip'], key_points['right_shoulder'], key_points['right_elbow']),
        'left_hip': calculate_angle(key_points['left_shoulder'], key_points['left_hip'], key_points['left_knee']),
        'right_hip': calculate_angle(key_points['right_shoulder'], key_points['right_hip'], key_points['right_knee']),
        'left_knee': calculate_angle(key_points['left_hip'], key_points['left_knee'], key_points['left_ankle']),
        'right_knee': calculate_angle(key_points['right_hip'], key_points['right_knee'], key_points['right_ankle']),
    }
    
    return angles

# Process an image and return the angles
def process_image(image_path):
    try:
        image = opencv.imread(image_path)
        if image is None:
            return None  # Unable to read the image
        image = opencv.cvtColor(image, opencv.COLOR_BGR2RGB)
        results = pose.process(image)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            angles = extract_angles(landmarks)
            return angles
        else:
            return None
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

# Create a new Excel workbook and write the results
def write_to_excel(file_name, data):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(['Serial No', 'Left Elbow', 'Right Elbow', 'Left Shoulder', 'Right Shoulder', 'Left Hip', 'Right Hip', 'Left Knee', 'Right Knee', 'Pose'])

    serial_no = 1
    for image, angles in data.items():
        row = [serial_no]
        for joint in ['left_elbow', 'right_elbow', 'left_shoulder', 'right_shoulder', 'left_hip', 'right_hip', 'left_knee', 'right_knee']:
            row.append(angles[joint])
        row.append(image.split(os.sep)[-2])  # Pose name
        ws.append(row)
        serial_no += 1

    wb.save(file_name)

# Main function to process images and write results to Excel
def main(folder_path, excel_file):
    data = {}
    processed_images = 0
    unsuitable_images = 0

    for subdir, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(subdir, file)
                angles = process_image(image_path)
                if angles:
                    data[image_path] = angles
                    processed_images += 1
                else:
                    os.remove(image_path)
                    unsuitable_images += 1

    write_to_excel(excel_file, data)
    print(f"Total images read: {processed_images + unsuitable_images}")
    print(f"Total suitable images processed: {processed_images}")
    print(f"Total unsuitable images deleted: {unsuitable_images}")

# Example usage
main('DATASET', 'angles.xlsx')  # Replace 'DATASET' with your folder containing subfolders of images, and 'angles.xlsx' with the desired Excel file name
